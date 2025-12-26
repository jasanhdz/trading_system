import numpy as np
import xgboost as xgb
import joblib
from pathlib import Path
from typing import Dict, Optional, Union

class XGBoostTradingModel:
    """
    Wrapper para XGBoost optimizado para trading.
    Se especializa en capturar relaciones no lineales en datos tabulares.
    """
    def __init__(
        self,
        model_params: Optional[Dict] = None,
        use_gpu: bool = True
    ):
        self.use_gpu = use_gpu
        
        # Configuración por defecto "Institutional Grade"
        default_params = {
            'objective': 'multi:softprob', # Probabilidades para 3 clases
            'num_class': 3,
            'eval_metric': ['mlogloss', 'merror'],
            'booster': 'gbtree',
            'tree_method': 'hist', # Más rápido
            'device': 'cuda' if use_gpu else 'cpu',
            'max_depth': 6,
            'learning_rate': 0.05,
            'n_estimators': 1000,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1, # L1 reg
            'reg_lambda': 1.0, # L2 reg
            'early_stopping_rounds': 50,
            'verbosity': 1
        }
        
        if model_params:
            default_params.update(model_params)
            
        self.params = default_params
        self.model = None
        
    def prepare_data(self, X: np.ndarray, y: np.ndarray) -> xgb.DMatrix:
        """
        Convierte tensores 3D (Batch, Seq, Feat) a 2D para XGBoost.
        Estrategia: Flatten de la última ventana o ingeniería de features agregados.
        Por simplicidad y potencia: Usamos la última fila de la secuencia + 
        algunas estadísticas de la ventana (mean, std).
        """
        # X shape: (Batch, Seq_Len, Features) OR (Batch, Features)
        
        if X.ndim == 2:
            # Already flattened or just features
            X_flat = X
        else:
            # 1. Último estado (lo más importante)
            last_step = X[:, -1, :] # (Batch, Features)
            X_flat = last_step
        
        return xgb.DMatrix(X_flat, label=y)

    def train(self, X_train, y_train, X_val, y_val):
        """Entrena el modelo con Early Stopping."""
        dtrain = self.prepare_data(X_train, y_train)
        dval = self.prepare_data(X_val, y_val)
        
        evals = [(dtrain, 'train'), (dval, 'eval')]
        
        # Separar params que van al constructor vs train
        train_params = self.params.copy()
        n_estimators = train_params.pop('n_estimators')
        early_stopping = train_params.pop('early_stopping_rounds')
        
        self.model = xgb.train(
            params=train_params,
            dtrain=dtrain,
            num_boost_round=n_estimators,
            evals=evals,
            early_stopping_rounds=early_stopping,
            verbose_eval=100
        )
        
        return self.model

    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Retorna probabilidades y predicción."""
        if not self.model:
            raise ValueError("Modelo no entrenado")
            
        # Preparar datos (sin label)
        # X shape: (Batch, Seq, Feat) -> Flatten
        if X.ndim == 2:
            X_flat = X
        else:
            X_flat = X[:, -1, :]
            
        dtest = xgb.DMatrix(X_flat)
        
        # Predicción (Softmax probs)
        probs = self.model.predict(dtest) # (Batch, 3)
        
        return {
            'logits': probs, # XGBoost devuelve probs directas con softprob, no logits crudos
            'probs': probs
        }

    def save(self, path: str):
        joblib.dump(self.model, path)
        
    def load(self, path: str):
        self.model = joblib.load(path)
