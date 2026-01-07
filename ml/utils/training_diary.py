# ml/utils/training_diary.py
"""
TrainingDiary - Sistema de bitácora para métricas de entrenamiento MLOps.
Registra accuracy, F1-Score y metadata de cada run para análisis histórico.
"""
import json
import pandas as pd
from datetime import datetime
from pathlib import Path

# Ruta donde guardaremos el diario
DIARY_PATH = Path(__file__).resolve().parents[3] / "data" / "training_diary.json"

class TrainingDiary:
    def __init__(self):
        self.path = DIARY_PATH
        self._ensure_file()

    def _ensure_file(self):
        if not self.path.parent.exists():
            self.path.parent.mkdir(parents=True)
        if not self.path.exists():
            with open(self.path, 'w') as f:
                json.dump([], f)

    def log_entry(self, symbol: str, model_type: str, version: str, metrics: dict):
        """Registra una nueva entrada en la bitácora."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "model_type": model_type,
            "version": version,
            "metrics": metrics
        }

        try:
            with open(self.path, 'r') as f:
                history = json.load(f)
            
            history.append(entry)

            with open(self.path, 'w') as f:
                json.dump(history, f, indent=2)
            
            print(f"📓 Diary updated for {symbol} ({model_type}): Acc={metrics.get('accuracy', 0):.4f}")
            
        except Exception as e:
            print(f"❌ Failed to write to training diary: {e}")

    def get_summary(self, limit=20):
        """Devuelve un DataFrame de Pandas para análisis fácil."""
        if not self.path.exists():
            return pd.DataFrame()
        
        with open(self.path, 'r') as f:
            data = json.load(f)
            
        flat_data = []
        for d in data:
            row = {
                "ts": d['timestamp'],
                "symbol": d['symbol'],
                "model": d['model_type'],
                "version": d['version'],
                **d['metrics']
            }
            flat_data.append(row)
            
        df = pd.DataFrame(flat_data)
        if not df.empty:
            df['ts'] = pd.to_datetime(df['ts'])
            df = df.sort_values('ts', ascending=False)
        return df.head(limit)

    def get_symbol_history(self, symbol: str, model_type: str = None):
        """Obtiene el historial de un símbolo específico."""
        df = self.get_summary(limit=1000)
        if df.empty:
            return df
        
        mask = df['symbol'] == symbol
        if model_type:
            mask &= df['model'] == model_type
        return df[mask].head(20)
