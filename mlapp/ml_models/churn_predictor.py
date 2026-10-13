import pandas as pd
import numpy as np
import joblib
from pathlib import Path

class ChurnPredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.load_model()
    
    def load_model(self):
        """Charger le modèle et le scaler"""
        base_dir = Path(__file__).parent
        model_path = base_dir / 'saved_models' / 'churn_model.pkl'
        scaler_path = base_dir / 'saved_models' / 'churn_scaler.pkl'
        
        if model_path.exists():
            self.model = joblib.load(model_path)
        if scaler_path.exists():
            self.scaler = joblib.load(scaler_path)
    
    def predict(self, input_data):
        """Prédire le churn"""
        try:
            # Convertir en DataFrame
            df = pd.DataFrame([input_data])
            
            # Sélectionner les features utilisées par le modèle
            features = [
                'gender', 'Age', 'abonoment_type',
                'avg_time_in_gym', 'avg_time_check_in', 'avg_time_check_out',
                'drink_abo', 'uses_sauna'
            ]
            
            # Vérifier que toutes les features sont présentes
            for feature in features:
                if feature not in df.columns:
                    df[feature] = 0  # Valeur par défaut
            
            # Réorganiser les colonnes
            df = df[features]
            
            # Normalisation si scaler disponible
            if self.scaler:
                df_scaled = self.scaler.transform(df)
            else:
                df_scaled = df.values
            
            # Prédiction
            prediction = self.model.predict(df_scaled)[0]
            probability = self.model.predict_proba(df_scaled)[0]
            
            return {
                'prediction': int(prediction),
                'probability': probability.tolist(),
                'label': 'À risque' if prediction == 1 else 'Fidèle',
                'confidence': float(np.max(probability))
            }
        except Exception as e:
            raise Exception(f"Erreur lors de la prédiction du churn: {str(e)}")