import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path

class MLUtils:
    @staticmethod
    def load_models():
        """Charger tous les modèles"""
        models = {}
        base_dir = Path(__file__).parent
        
        # Liste des modèles à charger
        model_files = {
            'churn_model': 'churn_model.pkl',
            'churn_scaler': 'churn_scaler.pkl',
            'clustering_model': 'clustering_model.pkl',
            'clustering_scaler': 'clustering_scaler.pkl',
            'satisfaction_model': 'satisfaction_model.pkl',
            'satisfaction_scaler': 'satisfaction_scaler.pkl'
        }
        
        for name, filename in model_files.items():
            filepath = base_dir / 'saved_models' / filename
            if filepath.exists():
                models[name] = joblib.load(filepath)
                print(f"✓ {name} chargé")
            else:
                print(f"⚠ {name} non trouvé à {filepath}")
        
        return models