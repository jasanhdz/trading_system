import numpy as np
import xgboost as xgb
import joblib
import pandas as pd
from typing import Dict, Optional, Union, Tuple
import logging

logger = logging.getLogger("MetaModel")

class MetaLabelingModel:
    """
    El 'Juez Supremo' del sistema.
    
    No predice la dirección del mercado.
    Predice si la señal generada por el Comité (Ensemble) será exitosa o no.
    
    Target: 1 si el trade fue rentable, 0 si fue pérdida.
    """
    
    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu
        self.model = None
        
        # Configuración específica para clasificación binaria (Rentable vs No Rentable)
        self.params = {
            'objective': 'binary:logistic',
            'eval_metric': ['logloss', 'auc'],
            'booster': 'gbtree',
            'tree_method': 'hist',
            'device': 'cuda' if use_gpu else 'cpu',
            'max_depth': 4, # Menos profundidad para evitar overfitting en meta-features
            'learning_rate': 0.03,
            'n_estimators': 500,
            'subsample': 0.7,
            'colsample_bytree': 0.7,
            'scale_pos_weight': 1.0, # Ajustar si hay desbalance de clases (muchos trades malos vs buenos)
            'verbosity': 1
        }
        
    def generate_meta_features(
        self, 
        ensemble_probs: np.ndarray, 
        market_features: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Construye los features para el Meta-Modelo.
        
        Args:
            ensemble_probs: (N, 3) Probabilidades del comité [Neutral, Long, Short]
            market_features: DataFrame con features de mercado (ATR, Volatilidad, Hora, etc.)
            
        Returns:
            DataFrame con meta-features.
        """
        meta_df = pd.DataFrame(index=market_features.index)
        
        # 1. Features del Comité
        # Confianza máxima (¿Qué tan seguro está el comité?)
        meta_df['committee_confidence'] = np.max(ensemble_probs, axis=1)
        
        # Entropía (Incertidumbre del comité)
        # -sum(p * log(p))
        eps = 1e-8
        meta_df['committee_entropy'] = -np.sum(ensemble_probs * np.log(ensemble_probs + eps), axis=1)
        
        # Dirección predicha (0, 1, 2)
        meta_df['committee_signal'] = np.argmax(ensemble_probs, axis=1)
        
        # Fuerza de la señal (Long - Short)
        meta_df['signal_strength'] = ensemble_probs[:, 1] - ensemble_probs[:, 2]
        
        # 2. Features de Contexto de Mercado (Extraídos de market_features)
        # Asumimos que market_features ya tiene columnas relevantes.
        # Seleccionamos las más importantes para "filtrar" trades.
        
        relevant_cols = [
            'atr', 'atr_pct', 'volatility', 'volume_ma_ratio', 
            'hour', 'day_of_week', 'spread', 'funding_rate'
        ]
        
        for col in relevant_cols:
            if col in market_features.columns:
                meta_df[f'ctx_{col}'] = market_features[col]
            # Si no existe, intentamos calcular o ignorar
            
        return meta_df

    def train(
        self, 
        ensemble_probs_train: np.ndarray,
        market_features_train: pd.DataFrame,
        trade_outcomes_train: np.ndarray, # 1 (Win) o 0 (Loss)
        ensemble_probs_val: np.ndarray,
        market_features_val: pd.DataFrame,
        trade_outcomes_val: np.ndarray
    ):
        """
        Entrena el Meta-Modelo.
        
        Args:
            trade_outcomes: Array binario. 1 si el trade propuesto por el comité hubiera ganado, 0 si no.
        """
        logger.info("🧠 Generando Meta-Features...")
        X_train = self.generate_meta_features(ensemble_probs_train, market_features_train)
        X_val = self.generate_meta_features(ensemble_probs_val, market_features_val)
        
        logger.info(f"   Meta-Features shape: {X_train.shape}")
        
        dtrain = xgb.DMatrix(X_train, label=trade_outcomes_train)
        dval = xgb.DMatrix(X_val, label=trade_outcomes_val)
        
        evals = [(dtrain, 'train'), (dval, 'eval')]
        
        # Ajustar scale_pos_weight dinámicamente
        # Si hay pocos ganadores (1s), aumentamos el peso de los 1s
        n_pos = np.sum(trade_outcomes_train == 1)
        n_neg = np.sum(trade_outcomes_train == 0)
        if n_pos > 0:
            self.params['scale_pos_weight'] = n_neg / n_pos
            logger.info(f"⚖️ Auto-balanced scale_pos_weight: {self.params['scale_pos_weight']:.2f}")
        
        logger.info("🚀 Entrenando Juez Supremo...")
        self.model = xgb.train(
            params=self.params,
            dtrain=dtrain,
            num_boost_round=self.params['n_estimators'],
            evals=evals,
            early_stopping_rounds=50,
            verbose_eval=50
        )
        
        return self.model

    def predict_veto(
        self, 
        ensemble_probs: np.ndarray, 
        market_features: pd.DataFrame,
        threshold: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evalúa las señales del comité.
        
        Returns:
            veto_mask: Boolean array. True si el trade debe ser VETADO (rechazado).
            approval_prob: Probabilidad de aprobación (0.0 a 1.0).
        """
        if not self.model:
            raise ValueError("Meta-Model no entrenado")
            
        X_meta = self.generate_meta_features(ensemble_probs, market_features)
        dtest = xgb.DMatrix(X_meta)
        
        # Probabilidad de que el trade sea EXITOSO (Clase 1)
        approval_prob = self.model.predict(dtest)
        
        # Si la probabilidad de éxito es menor al umbral -> VETO
        veto_mask = approval_prob < threshold
        
        return veto_mask, approval_prob

    def save(self, path: str):
        joblib.dump(self.model, path)
        
    def load(self, path: str):
        self.model = joblib.load(path)
