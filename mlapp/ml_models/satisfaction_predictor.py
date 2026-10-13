import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

class SatisfactionPredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.load_model()
    
    def load_model(self):
        """Charger le modèle de satisfaction"""
        base_dir = Path(__file__).parent
        model_path = base_dir / 'saved_models' / 'satisfaction_model.pkl'
        scaler_path = base_dir / 'saved_models' / 'satisfaction_scaler.pkl'
        
        if model_path.exists():
            self.model = joblib.load(model_path)
        if scaler_path.exists():
            self.scaler = joblib.load(scaler_path)
        else:
            self.scaler = MinMaxScaler()
    
    def predict(self, input_data):
        """Prédire le niveau de satisfaction"""
        try:
            # Convertir en DataFrame
            df = pd.DataFrame([input_data])
            
            # Features pour la satisfaction
            features = [
                'gender', 'Age', 'abonoment_type', 'visit_per_week',
                'days_per_week', 'avg_time_in_gym', 'avg_time_check_in',
                'avg_time_check_out', 'drink_abo', 'personal_training', 'uses_sauna'
            ]
            
            # Vérifier les features
            for feature in features:
                if feature not in df.columns:
                    df[feature] = 0
            
            df = df[features]
            
            # Normalisation
            if hasattr(self.scaler, 'transform'):
                df_scaled = self.scaler.transform(df)
            else:
                df_scaled = df.values
            
            # Prédiction
            prediction = self.model.predict(df_scaled)[0]
            probability = self.model.predict_proba(df_scaled)[0]
            
            # Niveaux de satisfaction
            levels = {
                0: 'Basse',
                1: 'Moyenne',
                2: 'Haute'
            }
            
            return {
                'level': int(prediction),
                'level_name': levels.get(prediction, 'Inconnu'),
                'probability': probability.tolist(),
                'score': round(np.max(probability) * 100, 2),
                'recommendations': self.get_recommendations(prediction)
            }
        except Exception as e:
            raise Exception(f"Erreur lors de la prédiction de satisfaction: {str(e)}")
    
    def get_recommendations(self, level):
        """Recommandations basées sur le niveau de satisfaction"""
        if level == 0:  # Basse
            return [
                'Offrir une séance d\'essai gratuite',
                'Proposer un plan d\'entraînement personnalisé',
                'Réduire les tarifs pour fidélisation'
            ]
        elif level == 1:  # Moyenne
            return [
                'Suggérer des cours collectifs',
                'Offrir des avantages sur les boissons',
                'Programmer un bilan avec un coach'
            ]
        else:  # Haute
            return [
                'Proposer un abonnement premium',
                'Inviter à des événements exclusifs',
                'Offrir un mois gratuit pour parrainage'
            ]