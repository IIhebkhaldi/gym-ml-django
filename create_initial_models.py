#!/usr/bin/env python
"""
Script pour créer les modèles initiaux
Exécutez: python create_initial_models.py
"""

import sys
import os

# Ajouter le projet Django au path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configurer Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')

import django
django.setup()

# Copiez-collez VOTRE CODE PYTHON COMPLET ICI
# Tout le code que vous m'avez envoyé au début

def main():
    """Copiez-collez votre code ici"""
    
    # === DÉBUT DE VOTRE CODE ===
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    
    # Charger les données
    Gym = pd.read_csv('gym_membership.csv')  # Mettez le bon chemin
    df_Gym = Gym.copy()
    
    # ... COLLEZ TOUT VOTRE CODE ICI ...
    # Tout le code de nettoyage, préparation, entraînement
    
    # À la fin, sauvegardez les modèles :
    import joblib
    
    # Créer le dossier
    os.makedirs('config/ml_models/saved_models', exist_ok=True)
    
    # Sauvegarder
    joblib.dump(best_xgb, 'mlapp/ml_models/saved_models/churn_model.pkl')
    # ... etc pour tous les modèles
    
    print("✅ Modèles créés et sauvegardés!")

if __name__ == "__main__":
    main()