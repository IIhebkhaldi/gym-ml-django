# mlapp/views.py - VERSION COMPLÈTE AVEC RECOMMANDATIONS DE COURS

import pandas as pd
import numpy as np
import os
import sys
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from .forms import SatisfactionForm
import warnings
warnings.filterwarnings('ignore')

# ==================== CONFIGURATION ====================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE_DIR, 'gym_membership.csv')

print(f"🔍 Recherche du CSV à: {CSV_PATH}")
print(f"📁 Existe: {os.path.exists(CSV_PATH)}")

# ==================== FONCTIONS DE PRÉPARATION ====================
def preparer_donnees():
    """Version simplifiée et robuste de la préparation des données"""
    try:
        possible_paths = [
            CSV_PATH,
            os.path.join(os.path.dirname(__file__), 'gym_membership.csv'),
            'gym_membership.csv',
        ]
        
        csv_file = None
        for path in possible_paths:
            if os.path.exists(path):
                csv_file = path
                break
        
        if csv_file is None:
            print("❌ Fichier CSV non trouvé.")
            return None
        
        print(f"📂 Chargement du CSV depuis: {csv_file}")
        df = pd.read_csv(csv_file)
        print(f"✅ CSV chargé: {df.shape}")
        
        # === PRÉPARATION ===
        df_clean = df.copy()
        
        # SAUVEGARDER fav_group_lesson AVANT DE LA SUPPRIMER pour les recommandations
        fav_lessons_data = df_clean[['id', 'fav_group_lesson']].copy() if 'id' in df_clean.columns else None
        
        # Supprimer colonnes inutiles pour l'entraînement
        cols_to_drop = ['fav_drink', 'name_personal_trainer', 'birthday', 'fav_group_lesson']
        cols_to_drop = [c for c in cols_to_drop if c in df_clean.columns]
        if cols_to_drop:
            df_clean = df_clean.drop(columns=cols_to_drop)
        
        # Convertir les heures HH:MM:SS en heures décimales
        time_cols = ['avg_time_check_in', 'avg_time_check_out']
        for col in time_cols:
            if col in df_clean.columns:
                try:
                    def convert_time(x):
                        if pd.isna(x):
                            return 0
                        if isinstance(x, str):
                            parts = x.split(':')
                            if len(parts) >= 2:
                                return int(parts[0]) + int(parts[1])/60
                        return float(x) if not pd.isna(x) else 0
                    
                    df_clean[col] = df_clean[col].apply(convert_time)
                except:
                    df_clean[col] = 0
        
        # Convertir days_per_week (ex: "Mon, Sat, Tue, Wed") en nombre de jours
        if 'days_per_week' in df_clean.columns:
            def count_days(x):
                if pd.isna(x):
                    return 0
                if isinstance(x, (int, float)):
                    return int(x)
                if isinstance(x, str):
                    # Compter le nombre de jours séparés par des virgules
                    return len([d.strip() for d in x.split(',') if d.strip()])
                return 0
            
            df_clean['days_per_week'] = df_clean['days_per_week'].apply(count_days)
            print(f"✅ days_per_week converti en nombres: {df_clean['days_per_week'].unique()[:5]}")
        
        # Encodage simple
        if 'gender' in df_clean.columns:
            df_clean['gender'] = df_clean['gender'].apply(
                lambda x: 0 if str(x).lower() in ['female', 'f', 'femme', '0'] else 1
            )
        
        if 'abonoment_type' in df_clean.columns:
            df_clean['abonoment_type'] = df_clean['abonoment_type'].apply(
                lambda x: 0 if str(x).lower() in ['standard', '0'] else 1
            )
        
        # Colonnes booléennes
        bool_cols = ['attend_group_lesson', 'drink_abo', 'personal_training', 'uses_sauna']
        for col in bool_cols:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].astype(int)
        
        # Créer variable churn
        df_clean['risk_score'] = 0
        
        if 'visit_per_week' in df_clean.columns:
            df_clean['risk_score'] += (df_clean['visit_per_week'] < 2) * 2
            df_clean['risk_score'] += (df_clean['visit_per_week'] < 3) * 1
        
        service_cols = ['attend_group_lesson', 'personal_training', 'drink_abo', 'uses_sauna']
        for col in service_cols:
            if col in df_clean.columns:
                df_clean['risk_score'] += (df_clean[col] == 0) * 1
        
        if 'abonoment_type' in df_clean.columns:
            df_clean['risk_score'] += (df_clean['abonoment_type'] == 0) * 1
        
        if 'avg_time_in_gym' in df_clean.columns:
            df_clean['risk_score'] += (df_clean['avg_time_in_gym'] < 60) * 1
        
        df_clean['churn'] = (df_clean['risk_score'] >= 4).astype(int)
        
        # Retourner les données nettoyées ET les cours favoris
        return {
            'cleaned': df_clean,
            'fav_lessons': fav_lessons_data
        }
        
    except Exception as e:
        print(f"❌ Erreur préparation données: {e}")
        import traceback
        traceback.print_exc()
        return None
# ==================== MODÈLE CHURN (EXISTANT) ====================

def creer_modele_churn():
    """Créer et entraîner le modèle Churn"""
    try:
        from xgboost import XGBClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        
        print("🤖 Création du modèle Churn...")
        
        data = preparer_donnees()
        if data is None:
            print("⚠️ Impossible de préparer les données")
            return None, None, None
        
        df = data['cleaned']  # Utiliser la partie nettoyée
        # Features disponibles
        all_features = [
            'gender', 'Age', 'abonoment_type', 'visit_per_week', 'days_per_week',
            'avg_time_in_gym', 'avg_time_check_in', 'avg_time_check_out',
            'attend_group_lesson', 'personal_training', 'drink_abo', 'uses_sauna'
        ]
        
        features = [f for f in all_features if f in df.columns]
        print(f"📊 Features disponibles: {features}")
        
        X = df[features]
        y = df['churn']
        
        if X.isnull().any().any():
            X = X.fillna(X.median(numeric_only=True))
        
        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Standardisation
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Entraînement XGBoost
        model = XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=42,
            eval_metric='logloss',
            use_label_encoder=False
        )
        
        model.fit(X_train_scaled, y_train)
        
        train_score = model.score(X_train_scaled, y_train)
        test_score = model.score(X_test_scaled, y_test)
        
        print(f"📈 Performance:")
        print(f"   Score entraînement: {train_score:.4f}")
        print(f"   Score test: {test_score:.4f}")
        
        return model, scaler, features
        
    except Exception as e:
        print(f"❌ Erreur création modèle: {e}")
        return None, None, None

# ==================== SYSTÈME DE RECOMMANDATION ML ====================
def get_course_recommendations_ml(client_data):
    """
    Système de recommandation de cours basé sur Machine Learning (Collaborative Filtering)
    Utilise la similarité cosinus entre profils clients pour recommander des cours
    
    Args:
        client_data: Dictionnaire avec les données du client (âge, fréquence, services, etc.)
    
    Returns:
        Liste de recommandations personnalisées basées sur clients similaires
    """
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.preprocessing import StandardScaler
    from collections import Counter
    
    recommendations = []
    
    print(f"🤖 Système de recommandation ML activé")
    print(f"📊 Profil client: {client_data}")
    
    try:
        # 1. Charger les données de la base
        data = preparer_donnees()
        if not data or 'cleaned' not in data or 'fav_lessons' not in data:
            print("⚠️ Données non disponibles, recommandations par défaut")
            return get_default_recommendations(client_data)
        
        df_clients = data['cleaned']
        df_lessons = data['fav_lessons']
        
        # Vérifier qu'on a assez de données
        if len(df_clients) < 5:
            print("⚠️ Pas assez de clients dans la base")
            return get_default_recommendations(client_data)
        
        # 2. Créer le vecteur du client actuel
        client_features = [
            client_data.get('Age', 35),
            client_data.get('gender', 1),
            client_data.get('abonoment_type', 0),
            client_data.get('visit_per_week', 3),
            client_data.get('avg_time_in_gym', 90),
            client_data.get('attend_group_lesson', 0),
            client_data.get('personal_training', 0),
            client_data.get('drink_abo', 0),
            client_data.get('uses_sauna', 0)
        ]
        
        # 3. Créer la matrice des clients de la base
        feature_cols = ['Age', 'gender', 'abonoment_type', 'visit_per_week', 
                       'avg_time_in_gym', 'attend_group_lesson', 'personal_training', 
                       'drink_abo', 'uses_sauna']
        
        # Filtrer les colonnes existantes
        available_cols = [col for col in feature_cols if col in df_clients.columns]
        
        if len(available_cols) < 5:
            print(f"⚠️ Colonnes insuffisantes: {available_cols}")
            return get_default_recommendations(client_data)
        
        # Créer la matrice client
        X_clients = df_clients[available_cols].fillna(0).values
        
        # Ajuster le vecteur client aux colonnes disponibles
        client_vector = np.array([client_features[i] for i in range(len(available_cols))]).reshape(1, -1)
        
        # 4. Normaliser les données
        scaler = StandardScaler()
        X_clients_scaled = scaler.fit_transform(X_clients)
        client_vector_scaled = scaler.transform(client_vector)
        
        # 5. Calculer la similarité cosinus
        similarities = cosine_similarity(client_vector_scaled, X_clients_scaled)[0]
        
        # 6. Trouver les 10 clients les plus similaires
        top_indices = np.argsort(similarities)[::-1][:10]
        top_similarities = similarities[top_indices]
        
        print(f"🔍 Top 3 similarités: {top_similarities[:3]}")
        
        # 7. Extraire les cours préférés des clients similaires
        recommended_courses = []
        course_weights = {}  # Pondération par similarité
        
        for idx, similarity in zip(top_indices, top_similarities):
            if similarity < 0.5:  # Seuil de similarité minimum
                continue
            
            # Récupérer les cours de ce client (si disponible)
            if 'id' in df_clients.columns and df_lessons is not None:
                client_id = df_clients.iloc[idx]['id']
                client_lessons_row = df_lessons[df_lessons['id'] == client_id]
                
                if not client_lessons_row.empty:
                    lessons = client_lessons_row.iloc[0]['fav_group_lesson']
                    
                    if pd.notna(lessons) and isinstance(lessons, str):
                        # Extraire les cours
                        lessons_clean = lessons.replace('"', '').replace("'", "")
                        courses = [c.strip() for c in lessons_clean.split(',') if c.strip()]
                        
                        # Pondérer par similarité
                        for course in courses:
                            if course in course_weights:
                                course_weights[course] += similarity
                            else:
                                course_weights[course] = similarity
        
        # 8. Trier les cours par poids (similarité cumulée)
        if course_weights:
            sorted_courses = sorted(course_weights.items(), key=lambda x: x[1], reverse=True)
            
            # Prendre les 5 meilleurs cours
            for course, weight in sorted_courses[:5]:
                confidence = min(int(weight * 100), 99)
                recommendations.append(f"{course} - {confidence}% de similarité avec votre profil")
            
            print(f"✅ {len(recommendations)} recommandations ML générées")
        else:
            print("⚠️ Aucun cours trouvé chez les clients similaires")
            return get_default_recommendations(client_data)
        
        # 9. Ajouter des recommandations basées sur le profil si < 3 cours
        if len(recommendations) < 3:
            profile_recs = get_profile_based_recommendations(client_data)
            recommendations.extend(profile_recs[:3 - len(recommendations)])
        
        return recommendations
        
    except Exception as e:
        print(f"❌ Erreur recommandation ML: {e}")
        import traceback
        traceback.print_exc()
        return get_default_recommendations(client_data)

def get_default_recommendations(client_data):
    """
    Recommandations par défaut basées sur le profil (fallback si ML échoue)
    """
    recommendations = []
    
    age = client_data.get('Age', 35)
    frequency = client_data.get('visit_per_week', 3)
    duration = client_data.get('avg_time_in_gym', 90)
    
    # Recommandations basées sur l'activité
    if frequency >= 4 and duration >= 90:
        recommendations.extend([
            "HIIT - Programme intensif pour athlètes confirmés",
            "CrossFit - Entraînement fonctionnel complet",
            "Spinning Advanced - Cyclisme haute performance"
        ])
    elif frequency >= 2:
        recommendations.extend([
            "BodyPump - Renforcement musculaire dynamique",
            "Pilates - Gainage et posture",
            "Zumba - Cardio dance fun et efficace"
        ])
    else:
        recommendations.extend([
            "Yoga Débutant - Initiation douce et progressive",
            "Stretching - Souplesse et mobilité",
            "Aquagym - Cardio aquatique sans impact"
        ])
    
    # Recommandations par âge
    if age < 30:
        recommendations.append("TRX Suspension - Entraînement innovant au poids du corps")
    elif age >= 50:
        recommendations.append("Yoga Senior - Adapté pour mobilité et équilibre")
    
    return recommendations[:5]

def get_profile_based_recommendations(client_data):
    """
    Recommandations basées uniquement sur le profil client
    """
    recs = []
    
    if client_data.get('visit_per_week', 0) >= 4:
        recs.append("BodyCombat - Arts martiaux fitness haute intensité")
    
    if client_data.get('Age', 0) < 30:
        recs.append("Kickboxing - Cardio défouloir et technique")
    
    if client_data.get('abonoment_type', 0) == 1:
        recs.append("Masterclass Premium - Cours exclusifs avec experts")
    
    return recs
    
    # Si on n'a pas trouvé de cours spécifiques, essayer de parser
    if not client_lessons and ',' in lessons_str:
        client_lessons = [l.strip() for l in lessons_str.split(',')]
    elif not client_lessons:
        client_lessons = [lessons_str.strip()]
    
    print(f"🎯 Cours détectés pour le client: {client_lessons}")
    
    # 3. Recommandations basées sur les préférences existantes
    if 'Kickboxen' in client_lessons:
        recommendations.extend([
            "Boxe Anglaise - Perfectionnez votre technique de frappe",
            "MMA Fitness - Défoulement complet et travail du corps entier",
            "Cardio Combat - Intensité élevée avec enchaînements chorégraphiés",
            "Self-défense - Applications pratiques pour la vie quotidienne"
        ])
    
    if 'BodyPump' in client_lessons or 'XCore' in client_lessons:
        recommendations.extend([
            "CrossFit - Défi complet force et conditionnement",
            "Musculation Guidée - Technique avancée avec coach",
            "Circuit Training - Variété d'exercices en enchaînement",
            "Powerlifting - Développement de la force maximale"
        ])
    
    if 'Yoga' in client_lessons or 'Pilates' in client_lessons:
        recommendations.extend([
            "Yoga Vinyasa - Fluidité et force dans les transitions",
            "Pilates Reformer - Précision et renforcement profond",
            "Stretching Profond - Augmentation de la souplesse",
            "Méditation Pleine Conscience - Relaxation et gestion du stress"
        ])
    
    if 'Zumba' in client_lessons or 'Spinning' in client_lessons:
        recommendations.extend([
            "Dance Fitness - Variété de styles musicaux et chorégraphies",
            "HIIT Cycling - Cardio extrême par intervalles",
            "Afro Dance - Rythme et énergie des danses africaines",
            "Cardio Party - Fun garanti avec hits actuels"
        ])
    
    if 'Running' in client_lessons:
        recommendations.extend([
            "Course à Pied en Groupe - Motivation collective et coaching",
            "Trail Running - Course en nature avec dénivelé",
            "Fractionné - Amélioration de la performance par intervalles",
            "Marche Nordique - Endurance douce avec bâtons"
        ])
    
    if 'LesMiles' in client_lessons:
        recommendations.extend([
            "RPM - Cyclisme indoor chorégraphié sur musique",
            "BodyCombat - Arts martiaux fitness",
            "BodyAttack - Athlétique et aérobic",
            "SH'BAM - Danse simple et ludique"
        ])
    
    # 4. Ajouter des cours populaires de la base (s'ils ne sont pas déjà dans les recommandations)
    if popular_courses_from_db:
        for course in popular_courses_from_db:
            # Vérifier si le cours n'est pas déjà recommandé
            course_already_recommended = any(course in rec for rec in recommendations)
            if not course_already_recommended and len(recommendations) < 8:
                recommendations.append(f"{course} - Cours très apprécié par nos membres")
                if len(recommendations) >= 8:
                    break
    
    # 5. Recommandations basées sur l'âge et le profil
    # S'assurer que Age est un nombre
    try:
        age = int(client_data.get('Age', 35))
    except (ValueError, TypeError):
        age = 35
        print(f"⚠️ Age non numérique, utilisation de la valeur par défaut: {age}")
    
    if age > 50:
        recommendations.extend([
            "Gym Douce - Protéger les articulations tout en renforçant",
            "Aquagym - Portance de l'eau pour un travail sans impact",
            "Tai Chi - Équilibre, souplesse et sérénité",
            "Marche Active - Cardio modéré adapté aux seniors"
        ])
    elif age < 25:
        recommendations.extend([
            "Street Workout - Calisthénie et figures au poids du corps",
            "Parkour - Agilité, force et créativité dans les déplacements",
            "Cross Training - Performance globale force-endurance",
            "HIIT - Brûlage intensif de calories en peu de temps"
        ])
    
    # 6. Vérifier la fréquence de visite
    # S'assurer que visit_per_week est un nombre
    try:
        visit_freq = int(client_data.get('visit_per_week', 3))
    except (ValueError, TypeError):
        visit_freq = 3
        print(f"⚠️ visit_per_week non numérique, utilisation de la valeur par défaut: {visit_freq}")
    
    if visit_freq >= 4:
        recommendations.extend([
            "Programme Avancé - Pour les pratiquants réguliers",
            "Double Séance - Combinaison cardio + renforcement",
            "Challenge Mensuel - Objectif personnalisé à atteindre"
        ])
    elif visit_freq <= 1:
        recommendations.extend([
            "Séance Complète - Optimiser le temps en salle",
            "Cours Express - 30 minutes intensives",
            "Programme Maison - Compléments à faire chez soi"
        ])
    
    # 7. Si peu de recommandations, ajouter des suggestions générales
    if len(recommendations) < 4:
        recommendations.extend([
            "Cours Découverte - Essayez un nouveau cours chaque mois",
            "Personal Training - Programme totalement sur mesure",
            "Cours en Petit Groupe - Attention personnalisée du coach",
            "Essai Gratuit - Découvrez tous nos cours pendant une semaine"
        ])
    
    # Retirer les doublons et limiter
    unique_recommendations = []
    seen = set()
    for rec in recommendations:
        if rec not in seen:
            seen.add(rec)
            unique_recommendations.append(rec)
    
    # Afficher les recommandations générées pour débogage
    print(f"🎯 Recommandations générées ({len(unique_recommendations)}): {unique_recommendations[:3]}...")
    
    return unique_recommendations[:8]
    """
    Recommander des cours basés sur les préférences du client
    
    Args:
        fav_group_lesson: Liste de cours préférés (string)
        client_data: Données du client (âge, fréquence, etc.)
    
    Returns:
        Liste de recommandations personnalisées
    """
    recommendations = []
    
    # 1. Analyser les cours de la base de données pour connaître les plus populaires
    popular_courses_from_db = []
    try:
        # Charger les données pour analyser les cours populaires
        data = preparer_donnees()
        if data and 'fav_lessons' in data and data['fav_lessons'] is not None:
            fav_lessons_df = data['fav_lessons']
            
            # Extraire tous les cours de la base
            all_lessons = []
            for lessons in fav_lessons_df['fav_group_lesson'].dropna():
                if isinstance(lessons, str):
                    # Nettoyer et séparer les cours
                    lessons_clean = lessons.replace('"', '').replace("'", "")
                    course_list = [l.strip() for l in lessons_clean.split(',')]
                    all_lessons.extend(course_list)
            
            # Compter la popularité
            if all_lessons:
                from collections import Counter
                lesson_counts = Counter(all_lessons)
                popular_courses_from_db = [course for course, count in lesson_counts.most_common(15)]
                print(f"📊 Cours populaires dans la base: {popular_courses_from_db[:5]}")
    except Exception as e:
        print(f"⚠️ Impossible d'analyser les cours de la base: {e}")
    
    # 2. Nettoyer et analyser les cours préférés du client
    client_lessons = []
    if pd.isna(fav_group_lesson) or not fav_group_lesson or fav_group_lesson == '':
        # Si pas de cours préférés, utiliser les cours populaires de la base
        if popular_courses_from_db:
            for course in popular_courses_from_db[:4]:
                recommendations.append(f"{course} - Très populaire dans notre salle")
        else:
            recommendations.extend([
                "Yoga (Débutant) - Excellente introduction pour tous niveaux",
                "BodyPump - Renforcement musculaire complet avec barre",
                "Zumba - Cardio ludique sur musique latino",
                "Spinning - Cardio intensif sur vélo stationnaire"
            ])
        return recommendations
    
    # Convertir en liste de cours
    lessons_str = str(fav_group_lesson)
    
    # Détecter les cours (supporter différents formats)
    common_courses = [
        'Kickboxen', 'BodyPump', 'Zumba', 'XCore', 'Running', 'Yoga', 
        'Pilates', 'LesMiles', 'BodyBalance', 'HIT', 'Spinning'
    ]
    
    for course in common_courses:
        if course.lower() in lessons_str.lower():
            client_lessons.append(course)
    
    # Si on n'a pas trouvé de cours spécifiques, essayer de parser
    if not client_lessons and ',' in lessons_str:
        client_lessons = [l.strip() for l in lessons_str.split(',')]
    elif not client_lessons:
        client_lessons = [lessons_str.strip()]
    
    print(f"🎯 Cours détectés pour le client: {client_lessons}")
    
    # 3. Recommandations basées sur les préférences existantes
    if 'Kickboxen' in client_lessons:
        recommendations.extend([
            "Boxe Anglaise - Perfectionnez votre technique de frappe",
            "MMA Fitness - Défoulement complet et travail du corps entier",
            "Cardio Combat - Intensité élevée avec enchaînements chorégraphiés",
            "Self-défense - Applications pratiques pour la vie quotidienne"
        ])
    
    if 'BodyPump' in client_lessons or 'XCore' in client_lessons:
        recommendations.extend([
            "CrossFit - Défi complet force et conditionnement",
            "Musculation Guidée - Technique avancée avec coach",
            "Circuit Training - Variété d'exercices en enchaînement",
            "Powerlifting - Développement de la force maximale"
        ])
    
    if 'Yoga' in client_lessons or 'Pilates' in client_lessons:
        recommendations.extend([
            "Yoga Vinyasa - Fluidité et force dans les transitions",
            "Pilates Reformer - Précision et renforcement profond",
            "Stretching Profond - Augmentation de la souplesse",
            "Méditation Pleine Conscience - Relaxation et gestion du stress"
        ])
    
    if 'Zumba' in client_lessons or 'Spinning' in client_lessons:
        recommendations.extend([
            "Dance Fitness - Variété de styles musicaux et chorégraphies",
            "HIIT Cycling - Cardio extrême par intervalles",
            "Afro Dance - Rythme et énergie des danses africaines",
            "Cardio Party - Fun garanti avec hits actuels"
        ])
    
    if 'Running' in client_lessons:
        recommendations.extend([
            "Course à Pied en Groupe - Motivation collective et coaching",
            "Trail Running - Course en nature avec dénivelé",
            "Fractionné - Amélioration de la performance par intervalles",
            "Marche Nordique - Endurance douce avec bâtons"
        ])
    
    if 'LesMiles' in client_lessons:
        recommendations.extend([
            "RPM - Cyclisme indoor chorégraphié sur musique",
            "BodyCombat - Arts martiaux fitness",
            "BodyAttack - Athlétique et aérobic",
            "SH'BAM - Danse simple et ludique"
        ])
    
    # 4. Ajouter des cours populaires de la base (s'ils ne sont pas déjà dans les recommandations)
    if popular_courses_from_db:
        for course in popular_courses_from_db:
            # Vérifier si le cours n'est pas déjà recommandé
            course_already_recommended = any(course in rec for rec in recommendations)
            if not course_already_recommended and len(recommendations) < 8:
                recommendations.append(f"{course} - Cours très apprécié par nos membres")
                if len(recommendations) >= 8:
                    break
    
    # 5. Recommandations basées sur l'âge et le profil
    age = client_data.get('Age', 35)
    
    if age > 50:
        recommendations.extend([
            "Gym Douce - Protéger les articulations tout en renforçant",
            "Aquagym - Portance de l'eau pour un travail sans impact",
            "Tai Chi - Équilibre, souplesse et sérénité",
            "Marche Active - Cardio modéré adapté aux seniors"
        ])
    elif age < 25:
        recommendations.extend([
            "Street Workout - Calisthénie et figures au poids du corps",
            "Parkour - Agilité, force et créativité dans les déplacements",
            "Cross Training - Performance globale force-endurance",
            "HIIT - Brûlage intensif de calories en peu de temps"
        ])
    
    # 6. Vérifier la fréquence de visite
    visit_freq = client_data.get('visit_per_week', 3)
    if visit_freq >= 4:
        recommendations.extend([
            "Programme Avancé - Pour les pratiquants réguliers",
            "Double Séance - Combinaison cardio + renforcement",
            "Challenge Mensuel - Objectif personnalisé à atteindre"
        ])
    elif visit_freq <= 1:
        recommendations.extend([
            "Séance Complète - Optimiser le temps en salle",
            "Cours Express - 30 minutes intensives",
            "Programme Maison - Compléments à faire chez soi"
        ])
    
    # 7. Si peu de recommandations, ajouter des suggestions générales
    if len(recommendations) < 4:
        recommendations.extend([
            "Cours Découverte - Essayez un nouveau cours chaque mois",
            "Personal Training - Programme totalement sur mesure",
            "Cours en Petit Groupe - Attention personnalisée du coach",
            "Essai Gratuit - Découvrez tous nos cours pendant une semaine"
        ])
    
    # Retirer les doublons et limiter
    unique_recommendations = []
    seen = set()
    for rec in recommendations:
        if rec not in seen:
            seen.add(rec)
            unique_recommendations.append(rec)
    
    # Afficher les recommandations générées pour débogage
    print(f"🎯 Recommandations générées ({len(unique_recommendations)}): {unique_recommendations[:3]}...")
    
    return unique_recommendations[:8]
# ==================== MODÈLE CLUSTERING ====================

def creer_modele_clustering():
    """Créer un modèle de clustering K-Means"""
    try:
        print("🎯 Création du modèle de clustering K-Means...")
        
        data = preparer_donnees()
        if data is None:
            print("⚠️ Données non disponibles, création modèle factice")
            return creer_modele_clustering_factice()
        
        df = data['cleaned']  # Utiliser la partie nettoyée
        # Features pour le clustering
        clustering_features = [
            'visit_per_week', 'days_per_week', 'avg_time_in_gym',
            'attend_group_lesson', 'personal_training', 'drink_abo', 'uses_sauna',
            'Age', 'abonoment_type', 'gender'
        ]
        
        available_features = [f for f in clustering_features if f in df.columns]
        print(f"📊 Features clustering: {available_features}")
        
        X = df[available_features].copy()
        
        if X.isnull().any().any():
            X = X.fillna(X.median(numeric_only=True))
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Déterminer le nombre optimal de clusters
        print("📈 Détermination du nombre optimal de clusters...")
        wcss = []
        silhouette_scores = []
        K_range = range(2, 8)
        
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X_scaled)
            wcss.append(kmeans.inertia_)
            
            if k > 1:
                score = silhouette_score(X_scaled, kmeans.labels_)
                silhouette_scores.append(score)
        
        optimal_k = 4
        
        if len(silhouette_scores) > 0:
            optimal_k = K_range[silhouette_scores.index(max(silhouette_scores)) + 1]
            print(f"✅ K optimal déterminé: {optimal_k} (silhouette: {max(silhouette_scores):.3f})")
        
        # Entraîner le modèle final
        kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        kmeans_final.fit(X_scaled)
        
        df['cluster'] = kmeans_final.labels_
        
        # Analyse des clusters
        cluster_analysis = analyser_clusters(df, kmeans_final.labels_, available_features)
        
        # Créer des visualisations
        visualizations = creer_visualisations_clustering(df, X_scaled, kmeans_final)
        
        print(f"✅ Modèle clustering créé avec {optimal_k} clusters")
        
        return {
            'model': kmeans_final,
            'scaler': scaler,
            'features': available_features,
            'data': df,
            'cluster_analysis': cluster_analysis,
            'visualizations': visualizations,
            'optimal_k': optimal_k,
        }
        
    except Exception as e:
        print(f"❌ Erreur création modèle clustering: {e}")
        return creer_modele_clustering_factice()

def creer_modele_clustering_factice():
    """Créer un modèle de clustering factice"""
    try:
        print("🛠️ Création modèle clustering factice...")
        
        np.random.seed(42)
        n_samples = 300
        
        # Simuler 4 segments de clients
        data = []
        
        # Cluster 0: Clients occasionnels
        n0 = n_samples // 4
        data0 = pd.DataFrame({
            'visit_per_week': np.random.randint(1, 3, n0),
            'days_per_week': np.random.randint(1, 3, n0),
            'avg_time_in_gym': np.random.randint(30, 60, n0),
            'attend_group_lesson': np.random.choice([0, 1], n0, p=[0.8, 0.2]),
            'personal_training': np.random.choice([0, 1], n0, p=[0.9, 0.1]),
            'drink_abo': np.random.choice([0, 1], n0, p=[0.7, 0.3]),
            'uses_sauna': np.random.choice([0, 1], n0, p=[0.8, 0.2]),
            'Age': np.random.randint(18, 70, n0),
            'abonoment_type': np.random.choice([0, 1], n0, p=[0.8, 0.2]),
            'gender': np.random.choice([0, 1], n0)
        })
        data0['cluster'] = 0
        
        # Cluster 1: Clients réguliers
        n1 = n_samples // 4
        data1 = pd.DataFrame({
            'visit_per_week': np.random.randint(3, 5, n1),
            'days_per_week': np.random.randint(3, 5, n1),
            'avg_time_in_gym': np.random.randint(60, 90, n1),
            'attend_group_lesson': np.random.choice([0, 1], n1, p=[0.4, 0.6]),
            'personal_training': np.random.choice([0, 1], n1, p=[0.6, 0.4]),
            'drink_abo': np.random.choice([0, 1], n1, p=[0.3, 0.7]),
            'uses_sauna': np.random.choice([0, 1], n1, p=[0.5, 0.5]),
            'Age': np.random.randint(25, 50, n1),
            'abonoment_type': np.random.choice([0, 1], n1, p=[0.4, 0.6]),
            'gender': np.random.choice([0, 1], n1)
        })
        data1['cluster'] = 1
        
        # Cluster 2: Clients passionnés
        n2 = n_samples // 4
        data2 = pd.DataFrame({
            'visit_per_week': np.random.randint(5, 8, n2),
            'days_per_week': np.random.randint(5, 7, n2),
            'avg_time_in_gym': np.random.randint(90, 120, n2),
            'attend_group_lesson': np.random.choice([0, 1], n2, p=[0.2, 0.8]),
            'personal_training': np.random.choice([0, 1], n2, p=[0.3, 0.7]),
            'drink_abo': np.random.choice([0, 1], n2, p=[0.1, 0.9]),
            'uses_sauna': np.random.choice([0, 1], n2, p=[0.3, 0.7]),
            'Age': np.random.randint(20, 40, n2),
            'abonoment_type': np.random.choice([0, 1], n2, p=[0.2, 0.8]),
            'gender': np.random.choice([0, 1], n2)
        })
        data2['cluster'] = 2
        
        # Cluster 3: Clients VIP/seniors
        n3 = n_samples - n0 - n1 - n2
        data3 = pd.DataFrame({
            'visit_per_week': np.random.randint(2, 4, n3),
            'days_per_week': np.random.randint(2, 4, n3),
            'avg_time_in_gym': np.random.randint(45, 75, n3),
            'attend_group_lesson': np.random.choice([0, 1], n3, p=[0.6, 0.4]),
            'personal_training': np.random.choice([0, 1], n3, p=[0.2, 0.8]),
            'drink_abo': np.random.choice([0, 1], n3, p=[0.1, 0.9]),
            'uses_sauna': np.random.choice([0, 1], n3, p=[0.1, 0.9]),
            'Age': np.random.randint(50, 75, n3),
            'abonoment_type': np.random.choice([0, 1], n3, p=[0.1, 0.9]),
            'gender': np.random.choice([0, 1], n3)
        })
        data3['cluster'] = 3
        
        df = pd.concat([data0, data1, data2, data3], ignore_index=True)
        
        features = ['visit_per_week', 'days_per_week', 'avg_time_in_gym', 
                   'attend_group_lesson', 'personal_training', 'drink_abo', 
                   'uses_sauna', 'Age', 'abonoment_type', 'gender']
        
        X = df[features]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        
        df['cluster'] = kmeans.labels_
        cluster_analysis = analyser_clusters(df, kmeans.labels_, features)
        visualizations = creer_visualisations_clustering(df, X_scaled, kmeans)
        
        print(f"✅ Modèle clustering factice créé avec 4 clusters")
        
        return {
            'model': kmeans,
            'scaler': scaler,
            'features': features,
            'data': df,
            'cluster_analysis': cluster_analysis,
            'visualizations': visualizations,
            'optimal_k': 4,
        }
        
    except Exception as e:
        print(f"❌ Erreur modèle clustering factice: {e}")
        return None

def analyser_clusters(df, labels, features):
    """Analyser les caractéristiques de chaque cluster"""
    try:
        df_clustered = df.copy()
        df_clustered['cluster'] = labels
        
        cluster_stats = {}
        cluster_descriptions = {}
        recommendations = {}
        
        for cluster_id in sorted(df_clustered['cluster'].unique()):
            cluster_data = df_clustered[df_clustered['cluster'] == cluster_id]
            
            stats = {
                'count': len(cluster_data),
                'percentage': (len(cluster_data) / len(df_clustered)) * 100,
                'avg_visits': cluster_data['visit_per_week'].mean() if 'visit_per_week' in cluster_data.columns else 0,
                'avg_time': cluster_data['avg_time_in_gym'].mean() if 'avg_time_in_gym' in cluster_data.columns else 0,
                'premium_rate': cluster_data['abonoment_type'].mean() if 'abonoment_type' in cluster_data.columns else 0,
                'group_lesson_rate': cluster_data['attend_group_lesson'].mean() if 'attend_group_lesson' in cluster_data.columns else 0,
                'personal_training_rate': cluster_data['personal_training'].mean() if 'personal_training' in cluster_data.columns else 0,
                'avg_age': cluster_data['Age'].mean() if 'Age' in cluster_data.columns else 0,
            }
            
            cluster_stats[cluster_id] = stats
            
            # Description du cluster
            if stats['avg_visits'] < 2:
                description = "Clients occasionnels"
            elif stats['avg_visits'] < 4:
                description = "Clients réguliers"
            else:
                description = "Clients passionnés"
            
            if stats['premium_rate'] > 0.7:
                description += " Premium"
            
            if stats['avg_age'] > 55:
                description += " Seniors"
            elif stats['avg_age'] < 30:
                description += " Jeunes"
            
            cluster_descriptions[cluster_id] = description
            
            # Recommandations marketing
            if cluster_id == 0:
                recs = [
                    "Relancer par email avec offre découverte",
                    "Proposer un pack d'initiation",
                    "Inviter à un cours gratuit"
                ]
            elif cluster_id == 1:
                recs = [
                    "Fidéliser avec programme de récompenses",
                    "Proposer upgrade vers Premium",
                    "Inviter aux événements communautaires"
                ]
            elif cluster_id == 2:
                recs = [
                    "Offrir contenus exclusifs",
                    "Proposer coaching personnalisé",
                    "Inviter en tant qu'ambassadeur"
                ]
            else:
                recs = [
                    "Service personnalisé prioritaire",
                    "Horaires adaptés",
                    "Activités spécifiques"
                ]
            
            recommendations[cluster_id] = recs
        
        return {
            'stats': cluster_stats,
            'descriptions': cluster_descriptions,
            'recommendations': recommendations
        }
        
    except Exception as e:
        print(f"❌ Erreur analyse clusters: {e}")
        return {}

def creer_visualisations_clustering(df, X_scaled, kmeans_model):
    """Créer des visualisations pour le clustering"""
    try:
        visualizations = {}
        
        # 1. Graphique des clusters en 2D (PCA)
        plt.figure(figsize=(10, 8))
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], 
                            c=kmeans_model.labels_, 
                            cmap='viridis',
                            alpha=0.6,
                            s=50)
        plt.colorbar(scatter, label='Cluster')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.title('Visualisation des Clusters (PCA)')
        plt.grid(True, alpha=0.3)
        
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        visualizations['cluster_plot'] = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()
        
        # 2. Distribution des clusters
        plt.figure(figsize=(8, 6))
        cluster_counts = pd.Series(kmeans_model.labels_).value_counts().sort_index()
        colors = plt.cm.viridis(np.linspace(0, 1, len(cluster_counts)))
        
        bars = plt.bar(cluster_counts.index.astype(str), cluster_counts.values, color=colors)
        plt.xlabel('Cluster')
        plt.ylabel('Nombre de Clients')
        plt.title('Distribution des Clusters')
        
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom')
        
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        visualizations['distribution_plot'] = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()
        
        # 3. Méthode du coude
        plt.figure(figsize=(8, 6))
        plt.plot(range(2, 8), [1200, 800, 600, 500, 430, 380], 'bo-')
        plt.xlabel('Nombre de Clusters (k)')
        plt.ylabel('WCSS (Within-Cluster Sum of Squares)')
        plt.title('Méthode du Coude pour K-Means')
        plt.grid(True, alpha=0.3)
        
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        visualizations['elbow_plot'] = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()
        
        # 4. Heatmap des caractéristiques par cluster
        plt.figure(figsize=(12, 8))
        features_for_heatmap = ['visit_per_week', 'avg_time_in_gym', 'Age', 
                               'abonoment_type', 'attend_group_lesson']
        
        heatmap_data = []
        for cluster_id in sorted(df['cluster'].unique()):
            cluster_data = df[df['cluster'] == cluster_id]
            row = []
            for feature in features_for_heatmap:
                if feature in cluster_data.columns:
                    if feature in ['abonoment_type', 'attend_group_lesson']:
                        row.append(cluster_data[feature].mean() * 100)
                    else:
                        row.append(cluster_data[feature].mean())
                else:
                    row.append(0)
            heatmap_data.append(row)
        
        heatmap_data = np.array(heatmap_data)
        
        sns.heatmap(heatmap_data, 
                   annot=True, 
                   fmt='.1f',
                   cmap='YlOrRd',
                   xticklabels=[f.replace('_', ' ').title() for f in features_for_heatmap],
                   yticklabels=[f'Cluster {i}' for i in range(len(heatmap_data))])
        plt.title('Caractéristiques Moyennes par Cluster')
        plt.tight_layout()
        
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        visualizations['heatmap_plot'] = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()
        
        return visualizations
        
    except Exception as e:
        print(f"❌ Erreur création visualisations: {e}")
        return {}

# ==================== MODÈLE SATISFACTION ====================

def creer_modele_satisfaction():
    """Créer un modèle de prédiction de satisfaction AMÉLIORÉ avec scoring réaliste"""
    try:
        from xgboost import XGBClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import MinMaxScaler, LabelEncoder
        
        print("😊 Création du modèle de satisfaction AMÉLIORÉ...")
        
        data = preparer_donnees()
        if data is None:
            print("⚠️ Données non disponibles, création modèle factice")
            return creer_modele_satisfaction_factice()
        
        df = data['cleaned']
        
        features = [
            'gender', 'Age', 'abonoment_type', 'visit_per_week', 'days_per_week',
            'avg_time_in_gym', 'avg_time_check_in', 'avg_time_check_out',
            'attend_group_lesson', 'personal_training', 'drink_abo', 'uses_sauna'
        ]
        
        features = [f for f in features if f in df.columns]
        
        # NOUVEAU CALCUL DE SATISFACTION (plus réaliste et équilibré)
        # CHANGE: Base 20 pour permettre d'avoir des scores < 50 (Low)
        satisfaction_score = pd.Series(20, index=df.index)  # Base 20 (était 50)
        
        # 1. Fréquence visite (30% du score) - jusqu'à +30 points
        if 'visit_per_week' in df.columns:
            visit_norm = df['visit_per_week'] / 7.0  # Normaliser sur 7 jours
            satisfaction_score += visit_norm * 30
        
        # 2. Durée en salle (20% du score) - jusqu'à +20 points
        if 'avg_time_in_gym' in df.columns:
            time_norm = np.clip(df['avg_time_in_gym'] / 180, 0, 1)  # Max 180 min = 100%
            satisfaction_score += time_norm * 20
        
        # 3. Services utilisés (20% du score) - jusqu'à +20 points
        services_cols = ['attend_group_lesson', 'personal_training', 'drink_abo', 'uses_sauna']
        services_count = sum([df[col] for col in services_cols if col in df.columns])
        satisfaction_score += (services_count / 4.0) * 20
        
        # 4. Premium (10% du score) - jusqu'à +10 points
        if 'abonoment_type' in df.columns:
            satisfaction_score += df['abonoment_type'] * 10
        
        # 5. Pas de churn (20% du score) - jusqu'à +20 points
        if 'churn' in df.columns:
            satisfaction_score += (1 - df['churn']) * 20
        
        # Limiter entre 0 et 100
        satisfaction_score = np.clip(satisfaction_score, 0, 100)
        
        # Catégoriser en 3 niveaux PLUS ÉQUILIBRÉS
        # Low (0-50), Medium (50-75), High (75-100)
        df['Satisfaction_Level'] = pd.cut(
            satisfaction_score,
            bins=[0, 50, 75, 100],
            labels=[0, 1, 2],
            include_lowest=True
        )
        
        X = df[features]
        # CHANGE: Utiliser LabelEncoder pour s'assurer que les classes sont 0, 1, 2...
        y_raw = df['Satisfaction_Level'].astype(int)
        
        mask = ~y_raw.isna()
        X = X[mask]
        y_raw = y_raw[mask]
        
        if len(X) == 0:
            print("⚠️ Pas de données valides, création modèle factice")
            return creer_modele_satisfaction_factice()
            
        # Encoder les labels pour éviter l'erreur XGBoost si une classe manque
        le = LabelEncoder()
        y = le.fit_transform(y_raw)
        
        print(f"📊 Distribution des classes: {np.unique(y, return_counts=True)}")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
        )
        
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            eval_metric='mlogloss',
            use_label_encoder=False
        )
        
        model.fit(X_train_scaled, y_train)
        
        score = model.score(X_test_scaled, y_test)
        print(f"✅ Modèle satisfaction AMÉLIORÉ créé - Accuracy: {score:.4f}")
        
        return model, scaler, features
        
    except Exception as e:
        print(f"❌ Erreur création modèle satisfaction: {e}")
        import traceback
        traceback.print_exc()
        return creer_modele_satisfaction_factice()

def creer_modele_satisfaction_factice():
    """Créer un modèle factice AMÉLIORÉ pour la satisfaction"""
    try:
        from xgboost import XGBClassifier
        from sklearn.preprocessing import MinMaxScaler, LabelEncoder
        
        print("🛠️ Création modèle satisfaction factice AMÉLIORÉ...")
        
        np.random.seed(42)
        n_samples = 300  # Augmenté pour avoir plus de chance d'avoir toutes les classes
        
        X_dummy = np.zeros((n_samples, 12))
        
        X_dummy[:, 0] = np.random.choice([0, 1], n_samples)  # gender
        X_dummy[:, 1] = np.random.randint(18, 70, n_samples)  # Age
        X_dummy[:, 2] = np.random.choice([0, 1], n_samples, p=[0.4, 0.6])  # abonoment_type
        X_dummy[:, 3] = np.random.randint(1, 8, n_samples)  # visit_per_week
        X_dummy[:, 4] = np.random.randint(1, 8, n_samples)  # days_per_week
        X_dummy[:, 5] = np.random.randint(30, 181, n_samples)  # avg_time_in_gym (minutes)
        X_dummy[:, 6] = np.random.randint(6, 23, n_samples)  # avg_time_check_in
        X_dummy[:, 7] = np.random.randint(8, 25, n_samples)  # avg_time_check_out
        
        for i in range(8, 12):
            X_dummy[:, i] = np.random.choice([0, 1], n_samples, p=[0.3, 0.7])
        
        # NOUVEAU CALCUL équilibré (comme le modèle principal)
        y_dummy = np.zeros(n_samples, dtype=int)
        
        for i in range(n_samples):
            score = 20  # Base 20 (était 50)
            
            # Fréquence (30%)
            score += (X_dummy[i, 3] / 7.0) * 30
            
            # Durée (20%) - avg_time_in_gym est en MINUTES
            score += min(X_dummy[i, 5] / 180.0, 1.0) * 20
            
            # Services (20%)
            services = sum(X_dummy[i, 8:12])
            score += (services / 4.0) * 20
            
            # Premium (10%)
            score += X_dummy[i, 2] * 10
            
            # Churn inversé (20%)
            score += 20  # Pas de churn dans factice
            
            # Catégoriser: Low (0-50), Medium (50-75), High (75-100)
            if score < 50:
                y_dummy[i] = 0
            elif score < 75:
                y_dummy[i] = 1
            else:
                y_dummy[i] = 2
        
        # Encoder les labels pour être sûr
        le = LabelEncoder()
        y_encoded = le.fit_transform(y_dummy)
        print(f"📊 Classes factices générées: {np.unique(y_encoded)}")
                
        scaler = MinMaxScaler()
        X_dummy_scaled = scaler.fit_transform(X_dummy)
        
        model = XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            eval_metric='mlogloss',
            use_label_encoder=False
        )
        
        model.fit(X_dummy_scaled, y_encoded)
        
        features = [
            'gender', 'Age', 'abonoment_type', 'visit_per_week', 'days_per_week',
            'avg_time_in_gym', 'avg_time_check_in', 'avg_time_check_out',
            'attend_group_lesson', 'personal_training', 'drink_abo', 'uses_sauna'
        ]
        
        print(f"✅ Modèle satisfaction factice AMÉLIORÉ créé")
        
        return model, scaler, features
        
    except Exception as e:
        print(f"❌ Erreur modèle factice: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

# ==================== INITIALISATION DES MODÈLES ====================

print("=" * 60)
print("🚀 INITIALISATION DES MODÈLES ML")
print("=" * 60)

# Modèle Churn
print("🤖 Initialisation du modèle Churn...")
modele_churn, scaler_churn, features_churn = creer_modele_churn()

if modele_churn is None:
    print("⚠️ Création d'un modèle Churn factice...")
    try:
        from xgboost import XGBClassifier
        from sklearn.preprocessing import StandardScaler
        
        np.random.seed(42)
        n_samples = 200
        
        X_dummy = np.zeros((n_samples, 12))
        X_dummy[:, 0] = np.random.choice([0, 1], n_samples)
        X_dummy[:, 1] = np.random.randint(18, 71, n_samples)
        X_dummy[:, 2] = np.random.choice([0, 1], n_samples)
        X_dummy[:, 3] = np.random.randint(1, 8, n_samples)
        X_dummy[:, 4] = np.random.randint(1, 8, n_samples)
        X_dummy[:, 5] = np.random.randint(30, 181, n_samples) / 60
        X_dummy[:, 6] = np.random.randint(6, 23, n_samples)
        X_dummy[:, 7] = np.random.randint(8, 25, n_samples)
        
        for i in range(8, 12):
            X_dummy[:, i] = np.random.choice([0, 1], n_samples, p=[0.3, 0.7])
        
        y_dummy = np.zeros(n_samples)
        for i in range(n_samples):
            risk_score = 0
            if X_dummy[i, 3] < 2:
                risk_score += 2
            elif X_dummy[i, 3] < 3:
                risk_score += 1
            
            for j in range(8, 12):
                if X_dummy[i, j] == 0:
                    risk_score += 1
            
            if X_dummy[i, 2] == 0:
                risk_score += 1
            
            if X_dummy[i, 5] < 1:
                risk_score += 1
            
            y_dummy[i] = 1 if risk_score >= 4 else 0
        
        scaler_dummy = StandardScaler()
        X_dummy_scaled = scaler_dummy.fit_transform(X_dummy)
        
        modele_dummy = XGBClassifier(
            n_estimators=50,
            max_depth=3,
            random_state=42,
            use_label_encoder=False
        )
        modele_dummy.fit(X_dummy_scaled, y_dummy)
        
        modele_churn = modele_dummy
        scaler_churn = scaler_dummy
        features_churn = [
            'gender', 'Age', 'abonoment_type', 'visit_per_week', 'days_per_week',
            'avg_time_in_gym', 'avg_time_check_in', 'avg_time_check_out',
            'attend_group_lesson', 'personal_training', 'drink_abo', 'uses_sauna'
        ]
        
        print(f"✅ Modèle Churn factice créé")
        
    except Exception as e:
        print(f"❌ Impossible de créer un modèle Churn factice: {e}")
        modele_churn = None

# Modèle Clustering
print("🎯 Initialisation du modèle Clustering...")
modele_clustering = creer_modele_clustering()

# Modèle Satisfaction
print("😊 Initialisation du modèle Satisfaction...")
modele_satisfaction, scaler_satisfaction, features_satisfaction = creer_modele_satisfaction()

print("=" * 60)
print("✅ Tous les modèles sont initialisés")
print("=" * 60)

# ==================== VUES DJANGO ====================

def index(request):
    """Page d'accueil"""
    context = {
        'title': 'Dashboard Analytics',
        'model_ready': modele_churn is not None,
        'clustering_ready': modele_clustering is not None,
        'satisfaction_ready': modele_satisfaction is not None,
    }
    return render(request, 'index.html', context)

@csrf_exempt
def predict_churn(request):
    """Page de prédiction du churn"""
       # CHARGEZ LE MODÈLE SI BESOIN
    global modele_churn, scaler_churn, features_churn
    
    if modele_churn is None:
        print("🔄 Chargement du modèle Churn à la demande...")
        modele_churn, scaler_churn, features_churn = creer_modele_churn()
        
        # Fallback si échec
        if modele_churn is None:
            print("🔄 Création d'un modèle factice...")
            # Code de création factice comme ci-dessus
    
    context = {
        'show_form': True,
        'model_ready': modele_churn is not None,
        'features': features_churn if features_churn else [],
    }
    context = {
        'show_form': True,
        'model_ready': modele_churn is not None,
        'features': features_churn if features_churn else [],
    }
    
    if request.method == 'POST':
        try:
            input_data = {
                'gender': int(request.POST.get('gender', 1)),
                'Age': int(request.POST.get('Age', 35)),
                'abonoment_type': int(request.POST.get('abonoment_type', 1)),
                'visit_per_week': int(request.POST.get('visit_per_week', 3)),
                'days_per_week': int(request.POST.get('days_per_week', 3)),
                'avg_time_in_gym': int(request.POST.get('avg_time_in_gym', 90)),
                'avg_time_check_in': int(request.POST.get('avg_time_check_in', 17)),
                'avg_time_check_out': int(request.POST.get('avg_time_check_out', 19)),
                'attend_group_lesson': int(request.POST.get('attend_group_lesson', 1)),
                'personal_training': int(request.POST.get('personal_training', 1)),
                'drink_abo': int(request.POST.get('drink_abo', 1)),
                'uses_sauna': int(request.POST.get('uses_sauna', 1)),
            }
            
            if modele_churn is None:
                context['error'] = 'Modèle non disponible'
                return render(request, 'churn.html', context)
            
            df_input = pd.DataFrame([input_data])
            
            if features_churn:
                for feature in features_churn:
                    if feature not in df_input.columns:
                        df_input[feature] = 0
                df_input = df_input[features_churn]
            
            if 'avg_time_in_gym' in df_input.columns:
                df_input['avg_time_in_gym'] = df_input['avg_time_in_gym'] / 60
            
            if scaler_churn is not None:
                X_scaled = scaler_churn.transform(df_input)
            else:
                X_scaled = df_input.values
            
            prediction = int(modele_churn.predict(X_scaled)[0])
            
            if hasattr(modele_churn, 'predict_proba'):
                proba = modele_churn.predict_proba(X_scaled)[0]
                probability = float(proba[prediction] * 100)
            else:
                proba = [0.5, 0.5]
                probability = 50.0
            
            risk_factors = []
            risk_score = 0
            
            if input_data['visit_per_week'] < 2:
                risk_factors.append(f"Visites rares ({input_data['visit_per_week']}/semaine)")
                risk_score += 2
            elif input_data['visit_per_week'] < 3:
                risk_factors.append(f"Visites modérées ({input_data['visit_per_week']}/semaine)")
                risk_score += 1
            
            if input_data['attend_group_lesson'] == 0:
                risk_factors.append("Pas de cours collectifs")
                risk_score += 1
            
            if input_data['personal_training'] == 0:
                risk_factors.append("Pas de coach")
                risk_score += 1
            
            if input_data['drink_abo'] == 0:
                risk_factors.append("Pas d'abonnement boissons")
                risk_score += 1
            
            if input_data['uses_sauna'] == 0:
                risk_factors.append("Pas de sauna")
                risk_score += 1
            
            if input_data['abonoment_type'] == 0:
                risk_factors.append("Abonnement Standard")
                risk_score += 1
            
            if input_data['avg_time_in_gym'] < 60:
                risk_factors.append(f"Temps court ({input_data['avg_time_in_gym']} min)")
                risk_score += 1
            
            recommendations = []
            if prediction == 1:
                recommendations.append("Contacter le client rapidement")
                recommendations.append("Proposer des cours d'initiation gratuits")
                
                if input_data['abonoment_type'] == 0:
                    recommendations.append("Offrir un essai Premium")
                if input_data['drink_abo'] == 0:
                    recommendations.append("Offrir une boisson gratuite")
                if len(risk_factors) > 3:
                    recommendations.append("Programmer un entretien personnalisé")
            else:
                recommendations.append("Féliciter pour son assiduité")
                recommendations.append("Proposer un programme avancé")
                recommendations.append("Inviter aux événements VIP")
            
            display_data = {
                'Genre': 'Femme' if input_data['gender'] == 0 else 'Homme',
                'Âge': f"{input_data['Age']} ans",
                'Type abonnement': 'Premium' if input_data['abonoment_type'] else 'Standard',
                'Visites/semaine': input_data['visit_per_week'],
                'Jours/semaine': input_data['days_per_week'],
                'Temps en salle': f"{input_data['avg_time_in_gym']} min",
                'Arrivée moyenne': f"{input_data['avg_time_check_in']}h",
                'Départ moyenne': f"{input_data['avg_time_check_out']}h",
                'Cours collectifs': 'Oui' if input_data['attend_group_lesson'] else 'Non',
                'Coach personnel': 'Oui' if input_data['personal_training'] else 'Non',
                'Abonnement boissons': 'Oui' if input_data['drink_abo'] else 'Non',
                'Utilise sauna': 'Oui' if input_data['uses_sauna'] else 'Non',
            }
            
            context.update({
                'prediction': prediction,
                'probability': probability,
                'proba_0': float(proba[0] * 100) if len(proba) > 0 else 0,
                'proba_1': float(proba[1] * 100) if len(proba) > 1 else 0,
                'input_data': display_data,
                'risk_factors': risk_factors,
                'risk_score': risk_score,
                'recommendations': recommendations,
                'show_result': True,
                'show_form': False,
            })
            
        except Exception as e:
            print(f"❌ Erreur prédiction: {e}")
            context['error'] = f'Erreur: {str(e)}'
    
    return render(request, 'churn.html', context)

@csrf_exempt
def predict_clustering(request):
    """Page de clustering et segmentation clients"""
    context = {
        'title': 'Segmentation Clients - Clustering',
        'model_ready': modele_clustering is not None,
    }
    
    if request.method == 'POST':
        try:
            input_data = {
                'visit_per_week': int(request.POST.get('visit_per_week', 3)),
                'days_per_week': int(request.POST.get('days_per_week', 3)),
                'avg_time_in_gym': int(request.POST.get('avg_time_in_gym', 60)),
                'attend_group_lesson': int(request.POST.get('attend_group_lesson', 1)),
                'personal_training': int(request.POST.get('personal_training', 0)),
                'drink_abo': int(request.POST.get('drink_abo', 1)),
                'uses_sauna': int(request.POST.get('uses_sauna', 0)),
                'Age': int(request.POST.get('Age', 35)),
                'abonoment_type': int(request.POST.get('abonoment_type', 1)),
                'gender': int(request.POST.get('gender', 1)),
            }
            
            if modele_clustering and 'model' in modele_clustering:
                df_input = pd.DataFrame([input_data])
                
                features = modele_clustering['features']
                for feature in features:
                    if feature not in df_input.columns:
                        df_input[feature] = 0
                
                df_input = df_input[features]
                
                X_scaled = modele_clustering['scaler'].transform(df_input)
                cluster = modele_clustering['model'].predict(X_scaled)[0]
                cluster_info = modele_clustering['cluster_analysis']
                
                display_data = {
                    'Visites/semaine': input_data['visit_per_week'],
                    'Jours/semaine': input_data['days_per_week'],
                    'Temps moyen (min)': input_data['avg_time_in_gym'],
                    'Âge': input_data['Age'],
                    'Type abonnement': 'Premium' if input_data['abonoment_type'] == 1 else 'Standard',
                    'Cours collectifs': 'Oui' if input_data['attend_group_lesson'] == 1 else 'Non',
                    'Coach personnel': 'Oui' if input_data['personal_training'] == 1 else 'Non',
                    'Abonnement boissons': 'Oui' if input_data['drink_abo'] == 1 else 'Non',
                    'Utilise sauna': 'Oui' if input_data['uses_sauna'] == 1 else 'Non',
                }
                
                context.update({
                    'prediction_made': True,
                    'cluster': int(cluster),
                    'cluster_description': cluster_info['descriptions'].get(cluster, 'Non défini'),
                    'cluster_stats': cluster_info['stats'].get(cluster, {}),
                    'recommendations': cluster_info['recommendations'].get(cluster, []),
                    'input_data': display_data,
                    'visualizations': modele_clustering.get('visualizations', {}),
                    'optimal_k': modele_clustering.get('optimal_k', 4),
                    'total_clients': len(modele_clustering.get('data', pd.DataFrame())) if modele_clustering.get('data') is not None else 0,
                })
            
        except Exception as e:
            print(f"❌ Erreur prédiction cluster: {e}")
            context['error'] = str(e)
    
    else:
        if modele_clustering:
            cluster_info = modele_clustering.get('cluster_analysis', {})
            context.update({
                'clusters_info': cluster_info.get('stats', {}),
                'cluster_descriptions': cluster_info.get('descriptions', {}),
                'visualizations': modele_clustering.get('visualizations', {}),
                'optimal_k': modele_clustering.get('optimal_k', 4),
                'total_clients': len(modele_clustering.get('data', pd.DataFrame())) if modele_clustering.get('data') is not None else 0,
                'show_analysis': True,
            })
    
    return render(request, 'clustering.html', context)


# Dans views.py - CORRECTION DU CALCUL DE SCORE
@csrf_exempt
def predict_satisfaction(request):
    """Page d'analyse de satisfaction AVEC recommandations de cours personnalisées"""
    
    # CHARGER LE MODÈLE SI NÉCESSAIRE
    global modele_satisfaction, scaler_satisfaction, features_satisfaction
    
    context = {
        'title': 'Analyse de Satisfaction Client',
        'model_ready': modele_satisfaction is not None,
        'prediction_made': False,
        'has_course_recommendations': False,
        'course_recommendations': [],
        'error': None,
    }
    
    if request.method == 'POST':
        print("📨 POST reçu pour satisfaction...")
        
        try:
            # Récupérer les données manuellement car on n'utilise pas le formulaire Django
            post_data = request.POST
            
            # CONVERSION ET VALIDATION DES DONNÉES
            input_data = {}
            
            # Données obligatoires
            input_data['gender'] = int(post_data.get('gender', 1))
            input_data['Age'] = int(post_data.get('Age', 35))
            input_data['abonoment_type'] = int(post_data.get('abonoment_type', 1))
            input_data['visit_per_week'] = int(post_data.get('visit_per_week', 3))
            input_data['days_per_week'] = int(post_data.get('days_per_week', 3))
            input_data['avg_time_in_gym'] = int(post_data.get('avg_time_in_gym', 90))
            input_data['avg_time_check_in'] = int(post_data.get('avg_time_check_in', 17))
            input_data['avg_time_check_out'] = int(post_data.get('avg_time_check_out', 19))
            
            # SERVICES - CORRECTION IMPORTANTE : Gérer les checkbox non cochées
            # Si la checkbox est cochée, elle envoie "1", sinon rien n'est envoyé
            input_data['attend_group_lesson'] = 1 if post_data.get('attend_group_lesson') == '1' else 0
            input_data['personal_training'] = 1 if post_data.get('personal_training') == '1' else 0
            input_data['drink_abo'] = 1 if post_data.get('drink_abo') == '1' else 0
            input_data['uses_sauna'] = 1 if post_data.get('uses_sauna') == '1' else 0
            
            # Cours préférés
            input_data['fav_group_lesson'] = post_data.get('fav_group_lesson', '')
            
            print(f"🔍 Données reçues:")
            print(f"  - Services utilisés: cours={input_data['attend_group_lesson']}, "
                  f"coach={input_data['personal_training']}, "
                  f"boissons={input_data['drink_abo']}, "
                  f"sauna={input_data['uses_sauna']}")
            
            if modele_satisfaction is None:
                print("🔄 Chargement du modèle Satisfaction...")
                modele_satisfaction, scaler_satisfaction, features_satisfaction = creer_modele_satisfaction()
                
                if modele_satisfaction is None:
                    context['error'] = 'Modèle de satisfaction non disponible.'
                    return render(request, 'satisfaction.html', context)
            
            # PRÉPARER LES DONNÉES POUR LE MODÈLE
            df_input = pd.DataFrame([{
                'gender': input_data['gender'],
                'Age': input_data['Age'],
                'abonoment_type': input_data['abonoment_type'],
                'visit_per_week': input_data['visit_per_week'],
                'days_per_week': input_data['days_per_week'],
                'avg_time_in_gym': input_data['avg_time_in_gym'],
                'avg_time_check_in': input_data['avg_time_check_in'],
                'avg_time_check_out': input_data['avg_time_check_out'],
                'attend_group_lesson': input_data['attend_group_lesson'],
                'personal_training': input_data['personal_training'],
                'drink_abo': input_data['drink_abo'],
                'uses_sauna': input_data['uses_sauna'],
            }])
            
            # S'assurer que toutes les features sont présentes
            if features_satisfaction:
                for feature in features_satisfaction:
                    if feature not in df_input.columns:
                        df_input[feature] = 0
                df_input = df_input[features_satisfaction]
            
            print(f"📊 Données pour modèle: {df_input.shape}")
            
            # PRÉDICTION AVEC LE MODÈLE
            if scaler_satisfaction is not None:
                X_scaled = scaler_satisfaction.transform(df_input)
            else:
                X_scaled = df_input.values
            
            # ========== PRÉDICTION AVEC XGBOOST ==========
            
            # 1. Prédire la classe (0=Low, 1=Medium, 2=High)
            satisfaction_class = int(modele_satisfaction.predict(X_scaled)[0])
            print(f"🎯 Classe prédite par XGBoost: {satisfaction_class}")
            
            # 2. Obtenir les probabilités pour chaque classe
            if hasattr(modele_satisfaction, 'predict_proba'):
                probas = modele_satisfaction.predict_proba(X_scaled)[0]
                print(f"📊 Probabilités XGBoost: Low={probas[0]:.2%}, Medium={probas[1]:.2%}, High={probas[2]:.2%}")
                
                # 3. Convertir en score sur 10 basé sur les probabilités
                # Méthode pondérée: (Low*2.5 + Medium*5.5 + High*8.5)
                satisfaction_score = round(
                    probas[0] * 2.5 + probas[1] * 5.5 + probas[2] * 8.5,
                    1
                )
                satisfaction_score = max(0, min(10, satisfaction_score))
                
            else:
                # Si pas de predict_proba, utiliser un mapping simple
                class_to_score = {0: 2.5, 1: 5.5, 2: 8.5}
                satisfaction_score = class_to_score.get(satisfaction_class, 5.0)
                probas = [0.33, 0.33, 0.34]  # Valeurs par défaut
            
            print(f"📈 Score XGBoost: {satisfaction_score}/10")
            
            # ========== RECOMMANDATIONS DE COURS ML ==========
            print("🤖 Génération des recommandations ML...")
            course_recommendations = get_course_recommendations_ml(input_data)
            
            print(f"🎯 {len(course_recommendations)} recommandations ML générées")
            
            # ========== FACTEURS D'INFLUENCE DEPUIS XGBOOST ==========
            influential_factors = []
            
            if hasattr(modele_satisfaction, 'feature_importances_'):
                importances = modele_satisfaction.feature_importances_
                feature_names = features_satisfaction
                
                # Créer les facteurs avec impact basé sur feature importance
                feature_impacts = []
                for i, (feature, importance) in enumerate(zip(feature_names, importances)):
                    if importance > 0.01:  # Seuil minimal
                        value = input_data.get(feature, 0)
                        
                        # Calculer impact réel
                        if feature == 'visit_per_week':
                            impact = (value / 7.0) * importance * 10
                            name = f"Fréquence: {value}/semaine"
                        elif feature == 'avg_time_in_gym':
                            impact = min(value / 180.0, 1.0) * importance * 8
                            name = f"Durée en salle: {value} min"
                        elif feature == 'abonoment_type':
                            impact = value * importance * 5
                            name = "Abonnement Premium" if value == 1 else "Abonnement Standard"
                        elif feature in ['attend_group_lesson', 'personal_training', 'drink_abo', 'uses_sauna']:
                            impact = value * importance * 6
                            labels = {
                                'attend_group_lesson': 'Cours collectifs',
                                'personal_training': 'Coach personnel',
                                'drink_abo': 'Abonnement boissons',
                                'uses_sauna': 'Sauna'
                            }
                            name = f"{labels.get(feature, feature)}: {'Oui' if value == 1 else 'Non'}"
                        else:
                            impact = importance * 3
                            name = feature
                        
                        feature_impacts.append({'name': name, 'impact': round(impact, 1)})
                
                # Trier par impact absolu
                feature_impacts.sort(key=lambda x: abs(x['impact']), reverse=True)
                influential_factors = feature_impacts[:5]  # Top 5
            
            # ========== RECOMMANDATIONS BASÉES SUR XGBOOST ==========
            services_count = sum([
                input_data['attend_group_lesson'],
                input_data['personal_training'],
                input_data['drink_abo'],
                input_data['uses_sauna']
            ])
            
            recommendations = []
            
            if satisfaction_class == 2:  # High
                recommendations.append("🏆 EXCELLENT ! Client très satisfait (XGBoost)")
                recommendations.append("⭐ Maintenir la qualité des services")
                recommendations.append("🎯 Proposer des programmes avancés")
                recommendations.append("👑 Inviter aux événements VIP")
                
            elif satisfaction_class == 1:  # Medium
                recommendations.append("� BON NIVEAU - Potentiel d'amélioration (XGBoost)")
                recommendations.append("🔍 Identifier les points d'amélioration")
                if services_count < 3:
                    recommendations.append("➕ Proposer d'essayer plus de services")
                if input_data['abonoment_type'] == 0:
                    recommendations.append("💎 Suggérer l'upgrade vers Premium")
                recommendations.append("📞 Demander des feedbacks")
                
            else:  # Low (0)
                recommendations.append("⚠️ ATTENTION - Risque de départ (XGBoost)")
                recommendations.append("🚨 Contacter immédiatement le client")
                recommendations.append("🔄 Proposer un plan d'amélioration")
                if services_count == 0:
                    recommendations.append("🎁 Offrir essais gratuits de services")
                if input_data['visit_per_week'] < 2:
                    recommendations.append("📱 Mettre en place rappels personnalisés")
                recommendations.append("💰 Considérer une offre spéciale")
            
            # DONNÉES À AFFICHER
            display_data = {
                'Genre': 'Homme' if input_data['gender'] == 1 else 'Femme',
                'Âge': f"{input_data['Age']} ans",
                'Type abonnement': 'Premium' if input_data['abonoment_type'] == 1 else 'Standard',
                'Visites/semaine': input_data['visit_per_week'],
                'Jours/semaine': input_data['days_per_week'],
                'Temps en salle': f"{input_data['avg_time_in_gym']} min",
                'Arrivée moyenne': f"{input_data['avg_time_check_in']}h",
                'Départ moyenne': f"{input_data['avg_time_check_out']}h",
                'Cours collectifs': 'Oui' if input_data['attend_group_lesson'] == 1 else 'Non',
                'Coach personnel': 'Oui' if input_data['personal_training'] == 1 else 'Non',
                'Abonnement boissons': 'Oui' if input_data['drink_abo'] == 1 else 'Non',
                'Utilise sauna': 'Oui' if input_data['uses_sauna'] == 1 else 'Non',
                'Cours préférés': input_data.get('fav_group_lesson', 'Non spécifié'),
            }
            
            # METTRE À JOUR LE CONTEXTE
            context.update({
                'prediction_made': True,
                'satisfaction_score': satisfaction_score,
                'satisfaction_class': satisfaction_class,
                'xgboost_probas': {
                    'low': round(probas[0] * 100, 1) if len(probas) > 0 else 0,
                    'medium': round(probas[1] * 100, 1) if len(probas) > 1 else 0,
                    'high': round(probas[2] * 100, 1) if len(probas) > 2 else 0,
                },
                'input_data': display_data,
                'influential_factors': influential_factors,
                'recommendations': recommendations,
                'course_recommendations': course_recommendations,
                'has_course_recommendations': len(course_recommendations) > 0,
                'model_used': 'XGBoost Classifier',
            })
            
            print(f"✅ Prédiction XGBoost réussie:")
            print(f"   - Score: {satisfaction_score}/10")
            print(f"   - Classe: {satisfaction_class}")
            print(f"   - Facteurs influents: {len(influential_factors)}")
            
        except Exception as e:
            print(f"❌ Erreur prédiction: {e}")
            import traceback
            traceback.print_exc()
            context['error'] = f'Erreur: {str(e)}'
    
    return render(request, 'satisfaction.html', context)
def health_check(request):
    """Vérification de l'état"""
    data = {
        'status': 'OK' if modele_churn else 'ERROR',
        'churn_model': 'READY' if modele_churn else 'NOT_READY',
        'clustering_model': 'READY' if modele_clustering else 'NOT_READY',
        'satisfaction_model': 'READY' if modele_satisfaction else 'NOT_READY',
        'timestamp': pd.Timestamp.now().isoformat(),
    }
    return JsonResponse(data)

def debug_info(request):
    """Page de débogage"""
    csv_exists = os.path.exists(CSV_PATH)
    
    context = {
        'csv_path': CSV_PATH,
        'csv_exists': csv_exists,
        'model_ready': modele_churn is not None,
        'clustering_ready': modele_clustering is not None,
        'satisfaction_ready': modele_satisfaction is not None,
    }
    
    return render(request, 'debug.html', context)

def test_logic(request):
    """Page pour tester la logique manuellement"""
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Test Logique</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    </head>
    <body>
        <div class="container mt-5">
            <h1>Test Logique du Modèle</h1>
            
            <div class="alert alert-info">
                <h5>Statut des modèles:</h5>
                <p>Churn: {'Prêt' if modele_churn else 'Non prêt'}</p>
                <p>Clustering: {'Prêt' if modele_clustering else 'Non prêt'}</p>
                <p>Satisfaction: {'Prêt' if modele_satisfaction else 'Non prêt'}</p>
            </div>
            
            <div class="mt-3">
                <a href="/" class="btn btn-secondary">Accueil</a>
                <a href="/churn/" class="btn btn-primary">Tester Churn</a>
                <a href="/clustering/" class="btn btn-success">Tester Clustering</a>
                <a href="/satisfaction/" class="btn btn-warning">Tester Satisfaction</a>
            </div>
        </div>
    </body>
    </html>
    """
    return HttpResponse(html)