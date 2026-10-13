import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path
from django.conf import settings

def train_and_save_models():
    """
    Fonction pour entraîner et sauvegarder les modèles directement
    Utilise le code de votre fichier Python converti
    """
    print("🔧 Entraînement des modèles ML...")
    
    # ==================== CHARGEMENT DES DONNÉES ====================
    # Chemin vers votre fichier CSV (ajustez le chemin)
    csv_path = os.path.join(settings.BASE_DIR, 'data', 'gym_membership.csv')
    
    if not os.path.exists(csv_path):
        print(f"❌ Fichier CSV non trouvé: {csv_path}")
        return False
    
    # Charger les données
    df_Gym = pd.read_csv(csv_path)
    
    # ==================== PRÉTRAITEMENT (votre code) ====================
    # Supprimer colonnes avec valeurs manquantes
    cols_to_drop = ['fav_drink', 'fav_group_lesson', 'name_personal_trainer']
    df_Gym = df_Gym.drop(columns=[c for c in cols_to_drop if c in df_Gym.columns])
    
    # Encodage LabelEncoder
    from sklearn.preprocessing import LabelEncoder
    cat_cols = ['gender', 'abonoment_type', 'days_per_week', 'avg_time_check_in', 'avg_time_check_out']
    cat_cols = [c for c in cat_cols if c in df_Gym.columns]
    
    le = LabelEncoder()
    for col in cat_cols:
        df_Gym[col] = le.fit_transform(df_Gym[col].astype(str))
    
    # Colonnes booléennes
    bool_cols = ['attend_group_lesson', 'drink_abo', 'personal_training', 'uses_sauna']
    for col in bool_cols:
        if col in df_Gym.columns:
            df_Gym[col] = df_Gym[col].astype(int)
    
    # ==================== DÉFINIR LE CHURN ====================
    # Créer la variable churn comme dans votre code
    df_Gym['churn'] = (
        (df_Gym['visit_per_week'] < 2) | 
        ((df_Gym['visit_per_week'] < 3) & 
         (df_Gym['attend_group_lesson'] == 0) & 
         (df_Gym['personal_training'] == 0))
    ).astype(int)
    
    # ==================== MODÈLE 1: CHURN (XGBoost) ====================
    print("🤖 Entraînement du modèle Churn (XGBoost)...")
    
    # Features pour churn
    features_churn = [
        'gender', 'Age', 'abonoment_type',
        'avg_time_in_gym', 'avg_time_check_in', 'avg_time_check_out',
        'drink_abo', 'uses_sauna'
    ]
    
    # Vérifier que toutes les features existent
    features_churn = [f for f in features_churn if f in df_Gym.columns]
    
    X = df_Gym[features_churn]
    y = df_Gym['churn']
    
    # Train/test split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Standardisation
    from sklearn.preprocessing import StandardScaler
    scaler_churn = StandardScaler()
    X_train_scaled = scaler_churn.fit_transform(X_train)
    
    # Entraînement XGBoost
    from xgboost import XGBClassifier
    
    # Calcul du scale_pos_weight pour déséquilibre
    neg, pos = y_train.value_counts()
    scale_pos_weight = neg / pos
    
    xgb_model = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        gamma=0.3,
        reg_alpha=0.5,
        reg_lambda=0.5,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric='logloss'
    )
    
    xgb_model.fit(X_train_scaled, y_train)
    
    # ==================== MODÈLE 2: CLUSTERING (K-Means) ====================
    print("🤖 Entraînement du modèle Clustering (K-Means)...")
    
    # Features pour clustering
    clustering_features = df_Gym.select_dtypes(include=[np.number]).columns.tolist()
    clustering_features = [f for f in clustering_features if f not in ['churn']]
    
    X_clustering = df_Gym[clustering_features].copy()
    
    # Normalisation
    from sklearn.preprocessing import MinMaxScaler
    scaler_clustering = MinMaxScaler()
    X_clustering_scaled = scaler_clustering.fit_transform(X_clustering)
    
    # K-Means
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    kmeans.fit(X_clustering_scaled)
    
    # ==================== MODÈLE 3: SATISFACTION (XGBoost) ====================
    print("🤖 Entraînement du modèle Satisfaction...")
    
    # Créer un score de satisfaction (comme dans votre code)
    # CHANGE: Base 20 pour permettre d'avoir des scores < 50 (Low)
    base_score = 20  # était 50
    
    if 'visit_per_week' in df_Gym.columns:
        visit_score = df_Gym['visit_per_week'] / 7.0 * 30
    else:
        visit_score = 15
    
    if 'avg_time_in_gym' in df_Gym.columns:
        # avg_time_in_gym est en minutes
        time_score = np.clip(df_Gym['avg_time_in_gym'] / 180, 0, 1) * 20
    else:
        time_score = 10
    
    # Services (20%)
    services_cols = ['attend_group_lesson', 'personal_training', 'drink_abo', 'uses_sauna']
    services_count = sum([df_Gym[col] for col in services_cols if col in df_Gym.columns])
    services_score = (services_count / 4.0) * 20
        
    # Premium (10%)
    if 'abonoment_type' in df_Gym.columns:
        premium_score = df_Gym['abonoment_type'] * 10
    else:
        premium_score = 0
    
    if 'churn' in df_Gym.columns:
        churn_score = (1 - df_Gym['churn']) * 20
    else:
        churn_score = 10
    
    satisfaction_score = base_score + visit_score + time_score + services_score + premium_score + churn_score
    satisfaction_score = np.clip(satisfaction_score, 0, 100)
    
    # Catégoriser en 3 niveaux
    df_Gym['Satisfaction_Level'] = pd.cut(
        satisfaction_score,
        bins=[0, 50, 75, 100],
        labels=[0, 1, 2],  # 0=Low, 1=Medium, 2=High
        include_lowest=True
    )
    
    # Features pour satisfaction
    satisfaction_features = clustering_features.copy()
    
    X_satisfaction = df_Gym[satisfaction_features].copy()
    y_satisfaction_raw = df_Gym['Satisfaction_Level'].astype(int)
    
    # Supprimer les NaN
    mask = ~y_satisfaction_raw.isna()
    X_satisfaction = X_satisfaction[mask]
    y_satisfaction_raw = y_satisfaction_raw[mask]
    
    # Label Encoder pour y
    le_sat = LabelEncoder()
    y_satisfaction = le_sat.fit_transform(y_satisfaction_raw)
    
    print(f"📊 Classes satisfaction: {np.unique(y_satisfaction)}")
    
    # Normalisation
    scaler_satisfaction = MinMaxScaler()
    X_satisfaction_scaled = scaler_satisfaction.fit_transform(X_satisfaction)
    
    # Entraînement XGBoost pour satisfaction
    xgb_satisfaction = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=42,
        eval_metric='mlogloss',
        use_label_encoder=False
    )
    
    xgb_satisfaction.fit(X_satisfaction_scaled, y_satisfaction)
    
    # ==================== SAUVEGARDE DES MODÈLES ====================
    print("💾 Sauvegarde des modèles...")
    
    # Créer le dossier pour les modèles
    model_dir = os.path.join(settings.BASE_DIR, 'mlapp', 'ml_models', 'saved_models')
    os.makedirs(model_dir, exist_ok=True)
    
    # Sauvegarder les modèles et scalers
    joblib.dump(xgb_model, os.path.join(model_dir, 'churn_model.pkl'))
    joblib.dump(scaler_churn, os.path.join(model_dir, 'churn_scaler.pkl'))
    
    joblib.dump(kmeans, os.path.join(model_dir, 'clustering_model.pkl'))
    joblib.dump(scaler_clustering, os.path.join(model_dir, 'clustering_scaler.pkl'))
    
    joblib.dump(xgb_satisfaction, os.path.join(model_dir, 'satisfaction_model.pkl'))
    joblib.dump(scaler_satisfaction, os.path.join(model_dir, 'satisfaction_scaler.pkl'))
    
    print(f"✅ Modèles sauvegardés dans: {model_dir}")
    
    # ==================== ÉVALUATION ====================
    print("\n📊 ÉVALUATION DES MODÈLES:")
    
    # Évaluer le modèle churn
    from sklearn.metrics import accuracy_score
    X_test_scaled = scaler_churn.transform(X_test)
    y_pred = xgb_model.predict(X_test_scaled)
    acc_churn = accuracy_score(y_test, y_pred)
    print(f"  • Churn Model Accuracy: {acc_churn:.4f}")
    
    # Évaluer le clustering
    from sklearn.metrics import silhouette_score
    silhouette = silhouette_score(X_clustering_scaled, kmeans.labels_)
    print(f"  • Clustering Silhouette Score: {silhouette:.4f}")
    
    # Évaluer la satisfaction
    if len(X_satisfaction_scaled) > 0:
        y_sat_pred = xgb_satisfaction.predict(X_satisfaction_scaled)
        acc_sat = accuracy_score(y_satisfaction, y_sat_pred)
        print(f"  • Satisfaction Model Accuracy: {acc_sat:.4f}")
    
    return True

if __name__ == "__main__":
    train_and_save_models()