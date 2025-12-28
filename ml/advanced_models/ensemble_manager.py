import torch
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import logging

# Importar arquitecturas
from ml.advanced_models.improved_architecture import DeepTemporalNet
from ml.advanced_models.tcn_model import TCNTradingModel
from ml.advanced_models.transformer_model import TradingTransformer
from ml.advanced_models.tabular_model import XGBoostTradingModel

logger = logging.getLogger("EnsembleManager")

class EnsembleManager:
    """
    Orquestador del 'Comité de Sabios'.
    Gestiona múltiples modelos heterogéneos y combina sus predicciones.
    """
    
    def __init__(self, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.models: Dict[str, Any] = {}
        self.weights: Dict[str, float] = {}
        self.configs: Dict[str, Dict] = {}
        
    def load_model(self, name: str, model_type: str, model_path: str, config_path: str, weight: float = 1.0):
        """
        Carga un modelo individual al ensemble.
        
        Args:
            name: Identificador único (ej. 'lstm_v1')
            model_type: 'lstm', 'tcn', 'transformer', 'xgboost'
            model_path: Ruta al archivo de pesos (.pt o .joblib)
            config_path: Ruta al json de configuración
            weight: Peso de voto en el ensemble
        """
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
            
        # Cargar config
        with open(config_path, 'r') as f:
            config = json.load(f)
            
        self.configs[name] = config
        self.weights[name] = weight
        
        # Instanciar arquitectura según tipo
        try:
            if model_type == 'lstm':
                # Extraer params relevantes del config
                # Asumimos que config tiene la estructura de production_training_results.json
                mc = config.get('model_config', config)
                model = DeepTemporalNet(
                    input_dim=mc.get('input_dim', 99), # Fallback si no está en config
                    hidden_dim=mc['hidden_dim'],
                    lstm_layers=mc['lstm_layers'],
                    dropout=mc.get('dropout', 0.2),
                    num_classes=3
                ).to(self.device)
                model.load_state_dict(torch.load(path, map_location=self.device))
                model.eval()
                self.models[name] = model
                
            elif model_type == 'tcn':
                mc = config.get('model_config', config)
                model = TCNTradingModel(
                    input_dim=mc.get('input_dim', 99),
                    num_channels=mc.get('num_channels', [64, 128, 256]),
                    kernel_size=mc.get('kernel_size', 3),
                    dropout=mc.get('dropout', 0.2)
                ).to(self.device)
                model.load_state_dict(torch.load(path, map_location=self.device))
                model.eval()
                self.models[name] = model
                
            elif model_type == 'transformer':
                mc = config.get('model_config', config)
                model = TradingTransformer(
                    input_dim=mc.get('input_dim', 99),
                    d_model=mc.get('d_model', 128),
                    nhead=mc.get('nhead', 4),
                    num_layers=mc.get('num_layers', 3)
                ).to(self.device)
                model.load_state_dict(torch.load(path, map_location=self.device))
                model.eval()
                self.models[name] = model
                
            elif model_type == 'xgboost':
                model = XGBoostTradingModel(use_gpu=(self.device.type == 'cuda'))
                model.load(str(path))
                self.models[name] = model
                
            else:
                raise ValueError(f"Unknown model type: {model_type}")
                
            logger.info(f"✅ Loaded {name} ({model_type}) - Weight: {weight}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load {name}: {e}")
            raise e

    def predict(self, x_tensor: torch.Tensor, x_numpy: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Genera predicción combinada.
        
        Args:
            x_tensor: Input para modelos PyTorch (Batch, Seq, Feat)
            x_numpy: Input para XGBoost (Batch, Seq, Feat) [Opcional, si es None se convierte x_tensor]
        """
        x_tensor = x_tensor.to(self.device)
        if x_numpy is None:
            x_numpy = x_tensor.cpu().numpy()
            
        individual_probs = {}
        weighted_probs_sum = torch.zeros(x_tensor.size(0), 3).to(self.device)
        total_weight = 0.0
        
        # Recolectar votos
        for name, model in self.models.items():
            weight = self.weights[name]
            total_weight += weight
            
            if isinstance(model, XGBoostTradingModel):
                # XGBoost output
                out = model.predict(x_numpy)
                probs = torch.from_numpy(out['probs']).to(self.device)
            else:
                # PyTorch output
                with torch.no_grad():
                    out = model(x_tensor)
                    # Convert logits to probs
                    probs = torch.softmax(out['logits'], dim=1)
            
            individual_probs[name] = probs
            weighted_probs_sum += probs * weight
            
        # Normalizar ensemble probs
        ensemble_probs = weighted_probs_sum / total_weight
        
        # Decisión final (Argmax)
        ensemble_class = torch.argmax(ensemble_probs, dim=1)
        
        # Confianza (Probabilidad de la clase elegida)
        confidence, _ = torch.max(ensemble_probs, dim=1)
        
        return {
            'ensemble_probs': ensemble_probs, # (Batch, 3)
            'ensemble_class': ensemble_class, # (Batch, )
            'confidence': confidence,         # (Batch, )
            'individual_votes': individual_probs
        }
        
    def get_consensus_level(self, prediction_result: Dict) -> float:
        """
        Calcula qué tan de acuerdo están los modelos.
        0.0 = Desacuerdo total
        1.0 = Unanimidad
        """
        votes = prediction_result['individual_votes']
        if not votes:
            return 0.0
            
        # Matriz de votos (Num_Models, 3)
        vote_matrix = torch.stack(list(votes.values()))
        
        # Desviación estándar entre las probabilidades de los modelos
        # Si todos dicen lo mismo, std es bajo.
        # Usamos 1 - std_promedio como métrica de consenso (simplificada)
        std_dev = torch.std(vote_matrix, dim=0).mean().item()
        
        # Normalizar un poco (std maximo es ~0.5)
        consensus = max(0.0, 1.0 - (std_dev * 2))
        return consensus
