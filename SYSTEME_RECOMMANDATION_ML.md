# 🤖 Système de Recommandation ML de Cours

## 📋 Vue d'ensemble

Le système de recommandation a été **remplacé** d'un système statique vers un **système de Machine Learning** basé sur le **Collaborative Filtering** avec **similarité cosinus**.

---

## 🔄 Avant vs Après

### ❌ AVANT (Système Statique)
```python
def get_course_recommendations(fav_group_lesson, client_data):
    # Recommandations fixes basées sur des règles IF/ELSE
    if frequency >= 4:
        return ["HIIT", "CrossFit", "Spinning"]
    else:
        return ["Yoga", "Pilates", "Zumba"]
```

**Problèmes** :
- ❌ Recommandations identiques pour tous les clients du même profil
- ❌ Pas d'apprentissage des préférences réelles
- ❌ Ignore les cours préférés des autres clients
- ❌ Pas d'adaptation au fil du temps

---

### ✅ APRÈS (Système ML)
```python
def get_course_recommendations_ml(client_data):
    # 1. Charger tous les clients de la base
    # 2. Créer vecteur de caractéristiques du client
    # 3. Calculer similarité cosinus avec TOUS les clients
    # 4. Trouver les 10 clients les plus similaires
    # 5. Analyser leurs cours préférés
    # 6. Pondérer par similarité
    # 7. Retourner top 5 cours recommandés
```

**Avantages** :
- ✅ Recommandations **personnalisées** basées sur clients similaires
- ✅ Utilise les **données réelles** de la base
- ✅ **Pondération** par degré de similarité
- ✅ S'améliore avec plus de données

---

## 🧮 Algorithme de Recommandation

### Étape 1 : Vectorisation du Profil Client
```python
client_features = [
    Age,                    # 18-80 ans
    gender,                 # 0=Femme, 1=Homme
    abonoment_type,         # 0=Standard, 1=Premium
    visit_per_week,         # 1-7 visites
    avg_time_in_gym,        # 30-180 minutes
    attend_group_lesson,    # 0 ou 1
    personal_training,      # 0 ou 1
    drink_abo,              # 0 ou 1
    uses_sauna              # 0 ou 1
]
```

### Étape 2 : Normalisation
```python
scaler = StandardScaler()
X_clients_scaled = scaler.fit_transform(X_clients)
client_vector_scaled = scaler.transform(client_vector)
```

### Étape 3 : Similarité Cosinus
```python
from sklearn.metrics.pairwise import cosine_similarity

similarities = cosine_similarity(client_vector_scaled, X_clients_scaled)[0]
# similarities = [0.95, 0.87, 0.82, 0.75, ...]
```

**Formule** :
```
similarité(A, B) = (A · B) / (||A|| × ||B||)
```

- **1.0** = Profil identique
- **0.8-0.99** = Très similaire
- **0.5-0.79** = Moyennement similaire
- **< 0.5** = Peu similaire (rejeté)

### Étape 4 : Extraction des Cours
```python
for idx, similarity in zip(top_10_indices, top_10_similarities):
    if similarity >= 0.5:  # Seuil minimum
        client_courses = get_courses_from_db(idx)
        for course in client_courses:
            course_weights[course] += similarity
```

### Étape 5 : Classement Pondéré
```python
sorted_courses = sorted(course_weights.items(), key=lambda x: x[1], reverse=True)

# Exemple :
# [("Kickboxing", 3.42), ("Yoga", 2.87), ("BodyPump", 2.35), ...]
#    ↑ cours        ↑ poids cumulé
```

---

## 📊 Exemple Concret

### Profil Client
```
Age: 28 ans
Genre: Homme
Abonnement: Premium
Fréquence: 5 visites/semaine
Durée: 120 minutes
Services: Cours collectifs + Coach + Sauna
```

### Clients Similaires Trouvés
```
Client #42 : Similarité 94% → Préfère [Kickboxing, CrossFit, TRX]
Client #87 : Similarité 89% → Préfère [HIIT, Kickboxing, Spinning]
Client #156: Similarité 82% → Préfère [BodyPump, Kickboxing, Running]
Client #203: Similarité 78% → Préfère [CrossFit, HIIT, TRX]
```

### Pondération
```
Kickboxing : 0.94 + 0.89 + 0.82 = 2.65
CrossFit   : 0.94 + 0.78 = 1.72
HIIT       : 0.89 + 0.78 = 1.67
TRX        : 0.94 + 0.78 = 1.72
BodyPump   : 0.82 = 0.82
```

### Recommandations Finales
```
1. Kickboxing - 94% de similarité avec votre profil
2. CrossFit - 87% de similarité avec votre profil
3. TRX - 86% de similarité avec votre profil
4. HIIT - 83% de similarité avec votre profil
5. BodyPump - 82% de similarité avec votre profil
```

---

## 🔧 Code Implémenté

### Fonction Principale
```python
def get_course_recommendations_ml(client_data):
    """
    Collaborative Filtering avec Cosine Similarity
    """
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.preprocessing import StandardScaler
    
    # 1. Charger données
    data = preparer_donnees()
    df_clients = data['cleaned']
    df_lessons = data['fav_lessons']
    
    # 2. Créer vecteur client
    client_vector = np.array([...]).reshape(1, -1)
    
    # 3. Normaliser
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_clients[features])
    client_scaled = scaler.transform(client_vector)
    
    # 4. Calculer similarités
    similarities = cosine_similarity(client_scaled, X_scaled)[0]
    
    # 5. Top 10 similaires
    top_indices = np.argsort(similarities)[::-1][:10]
    
    # 6. Extraire cours + pondérer
    course_weights = {}
    for idx, sim in zip(top_indices, similarities[top_indices]):
        if sim >= 0.5:
            courses = extract_courses(df_lessons, idx)
            for course in courses:
                course_weights[course] += sim
    
    # 7. Retourner top 5
    return sorted(course_weights.items(), key=lambda x: x[1], reverse=True)[:5]
```

---

## 📈 Performance et Amélioration

### Seuil de Similarité
```python
if similarity < 0.5:  # Minimum 50% de similarité
    continue
```

- **Trop bas** (< 0.3) : Recommandations peu pertinentes
- **Optimal** (0.5-0.6) : Bon équilibre précision/diversité
- **Trop haut** (> 0.8) : Peu de recommandations

### Fallback System
Si ML échoue (pas assez de données, erreur) :
```python
return get_default_recommendations(client_data)
```

Utilise des règles basiques pour toujours retourner des cours.

---

## 🎯 Affichage dans l'Interface

### Template HTML Modifié
```html
<div class="card border-primary">
    <div class="card-header bg-gradient">
        <i class="fas fa-robot"></i> Recommandations ML
        <small>Basées sur clients similaires via Machine Learning</small>
    </div>
    <div class="card-body">
        {% for course in course_recommendations %}
        <div class="card">
            <i class="fas fa-dumbbell"></i> {{ course }}
        </div>
        {% endfor %}
    </div>
</div>
```

---

## 🧪 Test du Système

### Scénario 1 : Client Actif
```python
Input:
    Age: 25, Premium, 6 visites/sem, 150 min
    Services: Tous activés

Expected Output:
    - HIIT - 92% similarité
    - CrossFit - 88% similarité
    - Kickboxing - 85% similarité
```

### Scénario 2 : Client Débutant
```python
Input:
    Age: 50, Standard, 2 visites/sem, 60 min
    Services: Aucun

Expected Output:
    - Yoga Débutant - 87% similarité
    - Stretching - 79% similarité
    - Aquagym - 72% similarité
```

---

## 📚 Dépendances Requises

```python
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from collections import Counter
import numpy as np
import pandas as pd
```

---

## ✅ Validation

### Checklist
- ✅ Système ML actif dans `views.py`
- ✅ Fonction `get_course_recommendations_ml()` créée
- ✅ Similarité cosinus implémentée
- ✅ Pondération par similarité
- ✅ Fallback système en place
- ✅ Template HTML mis à jour
- ✅ Badge "Recommandations ML" visible

---

## 🚀 Prochaines Améliorations

1. **Matrix Factorization** : SVD pour cours latents
2. **Deep Learning** : Neural Collaborative Filtering
3. **Hybrid System** : Content-based + Collaborative
4. **A/B Testing** : Comparer ML vs règles statiques
5. **Feedback Loop** : Réentraîner avec nouveaux choix clients

---

**Auteur** : Système de recommandation ML - Version 1.0  
**Date** : Décembre 2025  
**Algorithme** : Collaborative Filtering avec Cosine Similarity
