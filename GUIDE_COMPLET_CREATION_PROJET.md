# 📘 GUIDE COMPLET - CRÉATION D'UN PROJET DJANGO ML

## 🎯 VUE D'ENSEMBLE DU PROJET

**Nom du projet**: ProjetMLDjango  
**Type**: Application Web Django avec Machine Learning  
**Objectif**: Prédiction du churn, clustering clients, et satisfaction pour une salle de sport

---

## 📂 STRUCTURE COMPLÈTE DU PROJET

```
ProjetMLDjango/
│
├── manage.py                    # Script principal Django
├── db.sqlite3                   # Base de données SQLite
├── requirements.txt             # Dépendances Python
├── create_initial_models.py     # Script pour entraîner les modèles ML
│
├── data/                        # 📊 DONNÉES
│   └── gym_membership.csv       # Dataset principal
│
├── config/                      # ⚙️ CONFIGURATION DJANGO
│   ├── __init__.py
│   ├── settings.py              # Paramètres du projet
│   ├── urls.py                  # URLs principales
│   ├── wsgi.py                  # Configuration serveur
│   └── asgi.py                  # Configuration async
│
├── mlapp/                       # 🤖 APPLICATION PRINCIPALE
│   ├── __init__.py
│   ├── admin.py                 # Interface admin
│   ├── apps.py                  # Configuration app
│   ├── models.py                # Modèles de données (vide pour ce projet)
│   ├── views.py                 # Logique métier (1848 lignes!)
│   ├── urls.py                  # URLs de l'application
│   ├── forms.py                 # Formulaires Django (218 lignes)
│   ├── tests.py                 # Tests unitaires
│   │
│   ├── migrations/              # Migrations de base de données
│   │   └── __init__.py
│   │
│   └── ml_models/               # 🧠 MODÈLES MACHINE LEARNING
│       ├── __init__.py
│       ├── train_models.py      # Entraînement des modèles (222 lignes)
│       ├── churn_predictor.py   # Classe pour prédiction churn
│       ├── clustering_predictor.py  # Classe pour clustering
│       ├── satisfaction_predictor.py  # Classe pour satisfaction
│       ├── utils.py             # Fonctions utilitaires
│       │
│       └── saved_models/        # 💾 MODÈLES ENTRAÎNÉS (fichiers .pkl)
│           ├── churn_model.pkl
│           ├── churn_scaler.pkl
│           ├── clustering_model.pkl
│           ├── clustering_scaler.pkl
│           ├── satisfaction_model.pkl
│           └── satisfaction_scaler.pkl
│
├── templates/                   # 🎨 TEMPLATES HTML
│   ├── base.html                # Template de base
│   ├── index.html               # Page d'accueil
│   ├── churn.html               # Formulaire churn
│   ├── clustering.html          # Formulaire clustering
│   ├── satisfaction.html        # Formulaire satisfaction
│   └── result.html              # Page de résultats
│
└── static/                      # 📁 FICHIERS STATIQUES
    ├── css/
    │   └── style.css            # Styles personnalisés
    └── js/
        └── script.js            # Scripts JavaScript
```

---

## 🛠️ ÉTAPES DE CRÉATION DÉTAILLÉES

### **ÉTAPE 1: CONFIGURATION DE L'ENVIRONNEMENT**

#### 1.1 Créer un environnement virtuel
```bash
# Créer l'environnement
python -m venv venv

# Activer l'environnement (Windows)
venv\Scripts\activate

# Activer l'environnement (Linux/Mac)
source venv/bin/activate
```

#### 1.2 Installer les dépendances
Créez le fichier `requirements.txt`:
```txt
Django>=4.2
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
xgboost>=2.0
joblib>=1.3
matplotlib>=3.7
seaborn>=0.12
```

Installez:
```bash
pip install -r requirements.txt
```

---

### **ÉTAPE 2: CRÉATION DU PROJET DJANGO**

#### 2.1 Créer le projet
```bash
django-admin startproject config .
```

**⚠️ IMPORTANT**: Le point (`.`) à la fin crée le projet dans le dossier actuel

Cette commande crée:
- `manage.py`
- Dossier `config/` avec:
  - `settings.py`
  - `urls.py`
  - `wsgi.py`
  - `asgi.py`
  - `__init__.py`

#### 2.2 Créer l'application principale
```bash
python manage.py startapp mlapp
```

Cette commande crée le dossier `mlapp/` avec tous ses fichiers.

---

### **ÉTAPE 3: CONFIGURATION DE DJANGO**

#### 3.1 Modifier `config/settings.py`

**Ajoutez l'application dans INSTALLED_APPS:**
```python
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'mlapp',  # ← AJOUTER ICI
]
```

**Configurez les templates:**
```python
TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [BASE_DIR / 'templates'],  # ← AJOUTER ICI
        'APP_DIRS': True,
        ...
    },
]
```

**Configurez les fichiers statiques:**
```python
STATIC_URL = 'static/'
STATICFILES_DIRS = [
    BASE_DIR / 'static',  # ← AJOUTER ICI
]
```

#### 3.2 Modifier `config/urls.py`

```python
from django.contrib import admin
from django.urls import include, path

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('mlapp.urls')),  # ← AJOUTER ICI
]
```

---

### **ÉTAPE 4: CRÉER LA STRUCTURE DES DOSSIERS**

```bash
# Créer les dossiers manuellement
mkdir templates
mkdir static
mkdir static\css
mkdir static\js
mkdir data
mkdir mlapp\ml_models
mkdir mlapp\ml_models\saved_models
```

**OU** Django les créera automatiquement quand vous ajouterez des fichiers.

---

### **ÉTAPE 5: PRÉPARER LES DONNÉES**

#### 5.1 Placer le dataset
- Copiez votre fichier CSV dans `data/gym_membership.csv`
- **IMPORTANT**: Le chemin est `ProjetMLDjango/data/gym_membership.csv`

---

### **ÉTAPE 6: CRÉER LES MODÈLES MACHINE LEARNING**

#### 6.1 Créer le dossier `mlapp/ml_models/`

#### 6.2 Créer `mlapp/ml_models/__init__.py` (vide)

#### 6.3 Créer `mlapp/ml_models/train_models.py`

**Ce fichier contient:**
- Fonction `train_and_save_models()`
- Chargement du CSV depuis `data/gym_membership.csv`
- Prétraitement des données
- Entraînement de 3 modèles:
  1. **Churn**: XGBoost Classifier
  2. **Clustering**: K-Means
  3. **Satisfaction**: XGBoost Classifier
- Sauvegarde dans `mlapp/ml_models/saved_models/`

**Code principal:**
```python
import pandas as pd
import joblib
import os
from django.conf import settings

def train_and_save_models():
    # 1. Charger les données
    csv_path = os.path.join(settings.BASE_DIR, 'data', 'gym_membership.csv')
    df = pd.read_csv(csv_path)
    
    # 2. Prétraiter les données
    # ... (nettoyage, encodage, normalisation)
    
    # 3. Entraîner les modèles
    from xgboost import XGBClassifier
    from sklearn.cluster import KMeans
    
    xgb_churn = XGBClassifier(...)
    xgb_churn.fit(X_train, y_train)
    
    kmeans = KMeans(n_clusters=4)
    kmeans.fit(X_clustering)
    
    xgb_satisfaction = XGBClassifier(...)
    xgb_satisfaction.fit(X_satisfaction, y_satisfaction)
    
    # 4. Sauvegarder
    model_dir = os.path.join(settings.BASE_DIR, 'mlapp', 'ml_models', 'saved_models')
    os.makedirs(model_dir, exist_ok=True)
    
    joblib.dump(xgb_churn, os.path.join(model_dir, 'churn_model.pkl'))
    joblib.dump(scaler_churn, os.path.join(model_dir, 'churn_scaler.pkl'))
    # ... etc
```

#### 6.4 Créer les classes de prédiction

**`mlapp/ml_models/churn_predictor.py`:**
```python
import joblib
from pathlib import Path

class ChurnPredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.load_model()
    
    def load_model(self):
        base_dir = Path(__file__).parent
        self.model = joblib.load(base_dir / 'saved_models' / 'churn_model.pkl')
        self.scaler = joblib.load(base_dir / 'saved_models' / 'churn_scaler.pkl')
    
    def predict(self, input_data):
        # Transformer les données
        # Faire la prédiction
        return result
```

**Même chose pour:**
- `clustering_predictor.py`
- `satisfaction_predictor.py`

---

### **ÉTAPE 7: CRÉER LES FORMULAIRES**

Créez `mlapp/forms.py` avec:

```python
from django import forms

class ChurnForm(forms.Form):
    gender = forms.ChoiceField(
        label="Genre",
        choices=[(0, 'Femme'), (1, 'Homme')],
        widget=forms.Select(attrs={'class': 'form-control'})
    )
    
    Age = forms.IntegerField(
        label="Âge",
        widget=forms.NumberInput(attrs={'class': 'form-control'})
    )
    
    # ... tous les autres champs
```

**3 formulaires créés:**
1. `ChurnForm` - 11 champs
2. `ClusteringForm` - 11 champs
3. `SatisfactionForm` - 11 champs

---

### **ÉTAPE 8: CRÉER LES VUES**

Créez `mlapp/views.py` (1848 lignes!) avec:

**Structure:**
```python
from django.shortcuts import render
from .forms import ChurnForm, ClusteringForm, SatisfactionForm
from .ml_models.churn_predictor import ChurnPredictor
# ... imports

def index(request):
    """Page d'accueil"""
    return render(request, 'index.html')

def predict_churn(request):
    """Prédiction du churn"""
    if request.method == 'POST':
        form = ChurnForm(request.POST)
        if form.is_valid():
            # Récupérer les données
            data = form.cleaned_data
            
            # Faire la prédiction
            predictor = ChurnPredictor()
            result = predictor.predict(data)
            
            # Retourner le résultat
            return render(request, 'result.html', {'result': result})
    else:
        form = ChurnForm()
    
    return render(request, 'churn.html', {'form': form})

def predict_clustering(request):
    # ... même logique

def predict_satisfaction(request):
    # ... même logique
```

---

### **ÉTAPE 9: CONFIGURER LES URLs**

#### 9.1 Créer `mlapp/urls.py`

```python
from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('churn/', views.predict_churn, name='predict_churn'),
    path('clustering/', views.predict_clustering, name='predict_clustering'),
    path('satisfaction/', views.predict_satisfaction, name='predict_satisfaction'),
]
```

---

### **ÉTAPE 10: CRÉER LES TEMPLATES HTML**

#### 10.1 Créer `templates/base.html`

**Template de base avec Bootstrap:**
```html
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <title>{% block title %}GYM ML Analytics{% endblock %}</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    {% load static %}
    <link rel="stylesheet" href="{% static 'css/style.css' %}">
</head>
<body>
    <nav class="navbar navbar-dark bg-dark">
        <!-- Navigation -->
    </nav>
    
    <main>
        {% block content %}{% endblock %}
    </main>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
```

#### 10.2 Créer les autres templates

**`templates/index.html`:**
- Étend `base.html`
- 3 cartes Bootstrap pour chaque fonctionnalité
- Liens vers les formulaires

**`templates/churn.html`:**
- Formulaire de prédiction du churn
- Utilise `{{ form.as_p }}` ou rendu personnalisé

**`templates/result.html`:**
- Affiche les résultats de prédiction
- Graphiques (si générés)

---

### **ÉTAPE 11: ENTRAÎNER LES MODÈLES**

#### 11.1 Créer un script d'entraînement

Créez `create_initial_models.py` à la racine:

```python
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')

import django
django.setup()

from mlapp.ml_models.train_models import train_and_save_models

if __name__ == '__main__':
    train_and_save_models()
```

#### 11.2 Exécuter l'entraînement

```bash
python create_initial_models.py
```

**Résultat:**
Les fichiers `.pkl` sont créés dans `mlapp/ml_models/saved_models/`:
- `churn_model.pkl` (~500 KB)
- `churn_scaler.pkl` (~5 KB)
- `clustering_model.pkl` (~20 KB)
- `clustering_scaler.pkl` (~5 KB)
- `satisfaction_model.pkl` (~500 KB)
- `satisfaction_scaler.pkl` (~5 KB)

---

### **ÉTAPE 12: INITIALISER LA BASE DE DONNÉES**

```bash
# Créer les migrations
python manage.py makemigrations

# Appliquer les migrations
python manage.py migrate

# Créer un superutilisateur (optionnel)
python manage.py createsuperuser
```

---

### **ÉTAPE 13: TESTER L'APPLICATION**

```bash
# Lancer le serveur
python manage.py runserver

# Ouvrir dans le navigateur
# http://127.0.0.1:8000/
```

---

## 📍 OÙ SONT PLACÉS LES MODÈLES ML ?

### **Emplacement des modèles entraînés:**
```
mlapp/ml_models/saved_models/
├── churn_model.pkl         ← Modèle XGBoost pour churn
├── churn_scaler.pkl        ← Scaler StandardScaler
├── clustering_model.pkl    ← Modèle K-Means
├── clustering_scaler.pkl   ← Scaler MinMaxScaler
├── satisfaction_model.pkl  ← Modèle XGBoost pour satisfaction
└── satisfaction_scaler.pkl ← Scaler MinMaxScaler
```

### **Code d'entraînement:**
- `mlapp/ml_models/train_models.py` (222 lignes)

### **Classes de prédiction:**
- `mlapp/ml_models/churn_predictor.py`
- `mlapp/ml_models/clustering_predictor.py`
- `mlapp/ml_models/satisfaction_predictor.py`

### **Utilisation dans les vues:**
```python
# Dans views.py
from .ml_models.churn_predictor import ChurnPredictor

predictor = ChurnPredictor()
result = predictor.predict(data)
```

---

## 🚀 DÉPLOIEMENT

### **Option 1: Déploiement local (développement)**
```bash
python manage.py runserver 0.0.0.0:8000
```

### **Option 2: Déploiement sur serveur (production)**

#### Préparer pour la production:

**1. Modifier `settings.py`:**
```python
DEBUG = False
ALLOWED_HOSTS = ['votre-domaine.com', 'www.votre-domaine.com']

# Utiliser une base de données production (PostgreSQL)
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': 'gym_ml_db',
        'USER': 'postgres',
        'PASSWORD': 'votre-password',
        'HOST': 'localhost',
        'PORT': '5432',
    }
}
```

**2. Collecter les fichiers statiques:**
```bash
python manage.py collectstatic
```

**3. Utiliser Gunicorn:**
```bash
pip install gunicorn
gunicorn config.wsgi:application --bind 0.0.0.0:8000
```

**4. Configurer Nginx (reverse proxy):**
```nginx
server {
    listen 80;
    server_name votre-domaine.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    location /static/ {
        alias /chemin/vers/ProjetMLDjango/staticfiles/;
    }
}
```

### **Option 3: Déploiement Cloud**

#### **Heroku:**
```bash
# Créer Procfile
echo "web: gunicorn config.wsgi" > Procfile

# Ajouter runtime.txt
echo "python-3.11.0" > runtime.txt

# Déployer
git init
heroku create nom-app
git add .
git commit -m "Initial commit"
git push heroku main
```

#### **PythonAnywhere:**
1. Uploader les fichiers
2. Créer un environnement virtuel
3. Installer les dépendances
4. Configurer WSGI
5. Recharger l'application

#### **AWS / Google Cloud / Azure:**
- Utiliser Docker
- Configurer un serveur virtuel
- Installer les dépendances
- Configurer Nginx + Gunicorn

---

## 📋 CHECKLIST COMPLÈTE

### ✅ Avant de commencer:
- [ ] Python 3.8+ installé
- [ ] pip installé
- [ ] Environnement virtuel créé
- [ ] Dataset CSV disponible

### ✅ Configuration projet:
- [ ] Django installé
- [ ] Projet créé avec `startproject`
- [ ] Application créée avec `startapp`
- [ ] `settings.py` configuré
- [ ] URLs configurées

### ✅ Structure des dossiers:
- [ ] `templates/` créé
- [ ] `static/` créé
- [ ] `data/` créé
- [ ] `mlapp/ml_models/` créé
- [ ] `mlapp/ml_models/saved_models/` créé

### ✅ Code ML:
- [ ] `train_models.py` créé
- [ ] Classes de prédiction créées
- [ ] Modèles entraînés et sauvegardés (fichiers .pkl)

### ✅ Django:
- [ ] `forms.py` créé avec les 3 formulaires
- [ ] `views.py` créé avec les fonctions
- [ ] `urls.py` configuré
- [ ] Templates HTML créés

### ✅ Base de données:
- [ ] Migrations effectuées
- [ ] Base de données créée

### ✅ Tests:
- [ ] Serveur démarre sans erreur
- [ ] Page d'accueil accessible
- [ ] Formulaires fonctionnent
- [ ] Prédictions retournent des résultats

---

## 🔧 DÉPANNAGE COMMUN

### Erreur: "No module named 'mlapp'"
**Solution:** Vérifiez que `'mlapp'` est dans `INSTALLED_APPS`

### Erreur: "Template does not exist"
**Solution:** Vérifiez que `DIRS: [BASE_DIR / 'templates']` dans `settings.py`

### Erreur: "Static files not found"
**Solution:** Ajoutez `STATICFILES_DIRS = [BASE_DIR / 'static']`

### Erreur: "Model file not found"
**Solution:** Exécutez `python create_initial_models.py` pour créer les modèles

### Erreur: "CSV not found"
**Solution:** Vérifiez le chemin dans `train_models.py`: `data/gym_membership.csv`

---

## 📚 TECHNOLOGIES UTILISÉES

- **Framework Web:** Django 4.2+
- **ML - Classification:** XGBoost
- **ML - Clustering:** K-Means (scikit-learn)
- **Prétraitement:** pandas, numpy
- **Normalisation:** StandardScaler, MinMaxScaler
- **Sérialisation:** joblib
- **Frontend:** Bootstrap 5, Font Awesome
- **Database:** SQLite (dev), PostgreSQL (prod)

---

## 🎓 CONCEPTS CLÉS

### **1. Séparation des responsabilités:**
- `models.py`: Structure des données (non utilisé ici car pas de base de données pour les prédictions)
- `views.py`: Logique métier
- `forms.py`: Validation des entrées
- `templates/`: Présentation
- `ml_models/`: Intelligence artificielle

### **2. Architecture ML:**
- **Entraînement offline**: Les modèles sont pré-entraînés
- **Prédiction online**: Les modèles chargés font des prédictions en temps réel
- **Scalers sauvegardés**: Garantit la même transformation que lors de l'entraînement

### **3. Workflow:**
1. Utilisateur remplit formulaire
2. Django valide les données
3. Données envoyées au modèle ML
4. Modèle fait la prédiction
5. Résultat affiché à l'utilisateur

---

## 🎯 POUR CRÉER UN NOUVEAU PROJET

### **Suivez exactement ces étapes:**

1. **Créez l'environnement:**
```bash
mkdir MonNouveauProjet
cd MonNouveauProjet
python -m venv venv
venv\Scripts\activate
pip install django pandas scikit-learn xgboost joblib
```

2. **Créez le projet Django:**
```bash
django-admin startproject config .
python manage.py startapp monapp
```

3. **Configurez `settings.py`:**
- Ajoutez `'monapp'` dans `INSTALLED_APPS`
- Configurez `TEMPLATES` et `STATICFILES_DIRS`

4. **Créez la structure:**
```bash
mkdir templates static data monapp\ml_models monapp\ml_models\saved_models
```

5. **Ajoutez vos données:**
- Placez votre CSV dans `data/`

6. **Créez vos modèles ML:**
- Créez `monapp/ml_models/train_models.py`
- Entraînez et sauvegardez dans `saved_models/`

7. **Créez l'interface:**
- `forms.py` pour les formulaires
- `views.py` pour la logique
- Templates HTML pour l'affichage

8. **Lancez:**
```bash
python manage.py migrate
python manage.py runserver
```

---

## 📞 AIDE & SUPPORT

**Si vous rencontrez un problème:**
1. Vérifiez les logs dans le terminal
2. Consultez la documentation Django
3. Vérifiez que tous les fichiers sont aux bons emplacements
4. Assurez-vous que les modèles .pkl sont bien créés

---

**Créé le:** Décembre 2024  
**Version:** 1.0  
**Auteur:** Guide de déploiement Django ML

---

✅ **CE GUIDE VOUS PERMET DE REPRODUIRE EXACTEMENT LE MÊME PROJET !**
