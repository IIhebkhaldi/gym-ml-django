import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

class ClusteringPredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.load_model()
    
    def load_model(self):
        """Charger le modèle de clustering"""
        base_dir = Path(__file__).parent
        model_path = base_dir / 'saved_models' / 'clustering_model.pkl'
        scaler_path = base_dir / 'saved_models' / 'clustering_scaler.pkl'
        
        if model_path.exists():
            self.model = joblib.load(model_path)
        if scaler_path.exists():
            self.scaler = joblib.load(scaler_path)
        else:
            self.scaler = MinMaxScaler()
    
    def predict(self, input_data):
        """Assigner un cluster au client"""
        try:
            # Convertir en DataFrame
            df = pd.DataFrame([input_data])
            
            # Features pour le clustering (à adapter selon votre modèle)
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
            
            # Prédiction du cluster
            cluster = int(self.model.predict(df_scaled)[0])
            
            # Noms des clusters
            cluster_names = {
                0: 'Clients Occasionnels',
                1: 'Clients Réguliers',
                2: 'Clients Premium',
                3: 'Clients Inactifs'
            }
            
            return {
                'cluster': cluster,
                'cluster_name': cluster_names.get(cluster, f'Cluster {cluster}'),
                'description': self.get_cluster_description(cluster)
            }
        except Exception as e:
            raise Exception(f"Erreur lors du clustering: {str(e)}")
    
    def get_cluster_description(self, cluster):
        """Description des clusters"""
        descriptions = {
            0: 'Visites peu fréquentes, temps limité en salle',
            1: 'Visites régulières, bonne assiduité',
            2: 'Fréquentation élevée, utilisation des services premium',
            3: 'Visites rares, risque dabandon'
        }
        return descriptions.get(cluster, 'Cluster non défini')