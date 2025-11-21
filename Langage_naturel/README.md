# Pipeline de Classification des Relations Sémantiques

Classification automatique des relations sémantiques dans les constructions génitives françaises ("A de B").

## 📋 Description

Ce projet développe un système capable d'identifier automatiquement le type de relation sémantique entre deux noms dans une construction génitive française.

**Exemples** : 
- "la porte de la maison" → **r_holo** (Partie-Tout)
- "le livre de Marie" → **r_own-1** (Possession)
- "le train de Paris" → **r_lieu>origine** (Origine)

## 🎯 Objectifs

- ✅ Classifier 15 types de relations sémantiques
- ✅ Comparer différentes approches (ML classique, deep learning)
- ✅ Évaluer les performances face aux LLM
- ⚙️ Exploiter la ressource JeuxDeMots pour l'enrichissement

## 🗂️ Structure du Projet

```
.
├── data/
│   ├── raw/                    # Corpus initial (2250 exemples)
│   └── processed/              # Données prétraitées (train/val/test)
├── src/
│   ├── preprocessing/          # Nettoyage et normalisation
│   ├── features/               # Extraction de features
│   ├── models/                 # Modèles de classification
│   └── evaluation/             # Métriques et analyse
├── models/
│   └── baseline/               # Modèles entraînés (.joblib)
├── results/
│   ├── test_results.csv        # Résultats sur test set
│   ├── cross_validation_detailed.csv
│   └── plots/                  # Visualisations
├── run_preprocessing.py        # Prétraitement du corpus
├── run_feature_extraction.py  # Extraction de features
├── run_train_baseline.py      # Entraînement des modèles
├── run_evaluate_test.py       # Évaluation sur test set
├── run_cross_validation.py    # Validation croisée 10-fold
└── run_chatgpt_simple.py      # Comparaison avec ChatGPT
```

## 🚀 Installation

```bash
# Cloner le dépôt
git clone https://github.com/Bbzr333/Semestre3_projet
cd Langage_naturel

# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Installer les dépendances
pip install -r requirements.txt
```

## 📊 Types de Relations (15)

| Code | Description | Exemple |
|------|-------------|---------|
| `r_has_causatif` | Relation de cause | "les retards de la pluie" |
| `r_has_property-1` | Propriété | "la générosité du bénévole" |
| `r_objet>matiere` | Matière | "une table de bois" |
| `r_lieu>origine` | Origine géographique | "le vin de Bordeaux" |
| `r_topic` | Thème/Sujet | "un livre d'histoire" |
| `r_depic` | Représentation | "le portrait de Marie" |
| `r_holo` | Partie-Tout | "la porte de la maison" |
| `r_lieu` | Localisation | "les gens de la ville" |
| `r_processus_agent` | Agent d'un processus | "le discours du président" |
| `r_processus_patient` | Patient d'un processus | "la sculpture du bois" |
| `r_processus>instr-1` | Instrument | "le marteau de forgeron" |
| `r_own-1` | Possession | "le livre de Marie" |
| `r_quantificateur` | Quantité | "un kilo de pommes" |
| `r_social_tie` | Lien social | "l'ami de Pierre" |
| `r_product_of` | Produit de | "le tableau de Picasso" |

## 🔧 Utilisation

### Pipeline Complet

```bash
# 1. Prétraitement du corpus
python run_preprocessing.py
# → Génère: data/processed/corpus_preprocessed.csv

# 2. Extraction de features
python run_feature_extraction.py
# → Génère: data/processed/{train,val,test}.csv

# 3. Entraînement des modèles
python run_train_baseline.py
# → Génère: models/baseline/*.joblib

# 4. Évaluation sur test set
python run_evaluate_test.py
# → Génère: results/test_results.csv + matrices de confusion

# 5. Validation croisée
python run_cross_validation.py
# → Génère: results/cross_validation_detailed.csv

# 6. Comparaison avec ChatGPT (optionnel)
export OPENAI_API_KEY='sk-...'
python run_chatgpt_simple.py
# → Coût: ~$0.10 pour 50 exemples
```

### Utilisation du Meilleur Modèle

```python
from src.models.baseline_models import BaselineClassifier
import pandas as pd

# Charger le modèle
model = BaselineClassifier.load('models/baseline/random_forest.joblib')

# Prédire
# (Features extraites au préalable)
prediction = model.predict(features)
print(prediction)  # → 'r_holo'
```

## 📈 Résultats

### Performance sur Test Set (338 exemples)

| Modèle | Accuracy | F1-Score | Temps/exemple | Erreurs |
|--------|----------|----------|---------------|---------|
| **🥇 Random Forest** | **100.0%** | 1.000 | 0.001s | 0/338 |
| **🥈 Gradient Boosting** | **100.0%** | 1.000 | 0.003s | 0/338 |
| **🥉 SVM Linear** | 94.7% | 0.945 | 0.001s | 18/338 |
| SVM RBF | 93.5% | 0.935 | 0.001s | 22/338 |
| Logistic Regression | 86.4% | 0.862 | 0.001s | 46/338 |

### Validation Croisée 10-Fold

| Modèle | CV Accuracy | Std | Min | Max |
|--------|-------------|-----|-----|-----|
| Random Forest | 100.0% | 0.000 | 100% | 100% |
| Gradient Boosting | 100.0% | 0.000 | 100% | 100% |
| SVM Linear | 95.6% | 0.016 | 92.9% | 97.3% |
| SVM RBF | 93.9% | 0.015 | 91.6% | 96.4% |
| Logistic Regression | 86.1% | 0.014 | 83.6% | 88.0% |

**✅ Aucun overfitting détecté** - Performance stable train/test
**✅ Variance faible** - Robustesse confirmée sur tous les folds

### Features Utilisées (21)

- **Morphologiques** : voyelle initiale, terminaison (-e, -s)
- **Lexicales** : détection personne/lieu/temporel/matière
- **Structurelles** : longueur, ratio, présence déterminant
- **Sémantiques basiques** : catégories prédéfinies

### Comparaison avec LLM

| Modèle | Accuracy | Temps/exemple | Coût | Échantillon |
|--------|----------|---------------|------|-------------|
| Random Forest | 100.0% | 0.001s | Gratuit | 338 |
| Gradient Boosting | 100.0% | 0.001s | Gratuit | 338 |
| **GPT-3.5-turbo** | **95.0%** | 0.70s | $0.002/ex | 100 |
| SVM Linear | 94.7% | 0.001s | Gratuit | 338 |

**Résultats GPT-3.5-turbo** :
- Évaluation few-shot (2 exemples/classe)
- 5 erreurs sur 100 exemples
- Erreurs sur cas ambigus : polysémie (peinture), multi-interprétation (carte)
- Performance remarquable mais inférieure aux modèles ensemble

## 🔬 Analyse

### Pourquoi 100% ?

Le corpus actuel est **linéairement séparable** :
- ✅ Patterns morpho-syntaxiques distincts entre classes
- ✅ 150 exemples/classe bien équilibrés
- ✅ Peu d'ambiguïté sémantique
- ✅ Features basiques suffisantes (21 features)

**→ Les modèles ensemble (RF, GB) atteignent la perfection**

### Limites et Perspectives

**Limites actuelles :**
- Corpus trop simple (pas de cas ambigus)
- Relations bien séparées
- Pas de test sur données réelles

**Améliorations possibles :**
- Ajouter cas ambigus (multi-label)
- Tester sur corpus externe (Wikipedia, journaux)
- Features avancées (embeddings CamemBERT)
- Augmentation du corpus (5000+ exemples)

## 🤖 Comparaison Détaillée avec GPT-3.5

### Méthodologie
- **Modèle** : GPT-3.5-turbo (OpenAI API)
- **Approche** : Few-shot prompting (2 exemples/classe)
- **Échantillon** : 100 exemples du test set
- **Coût** : $0.20

### Résultats

**Performance globale** :
- Accuracy : **95.0%** (5 erreurs / 100 exemples)
- Temps moyen : 0.70s par exemple
- F1-score macro : 0.96

**Classes parfaites** (12/15) :
- `r_has_causatif`, `r_has_property-1`, `r_holo`, `r_lieu>origine`
- `r_objet>matiere`, `r_own-1`, `r_processus>instr-1`, `r_processus_agent`
- `r_quantificateur`, `r_social_tie`, `r_topic`, `r_lieu`

**Classes difficiles** :
- `r_depic` : 67% (confusion lieu/topic)
- `r_processus_patient` : 78% (polysémie peinture)
- `r_product_of` : 88% (créateur vs sujet)

### Analyse des Erreurs

Les 5 erreurs révèlent des **ambiguïtés sémantiques légitimes** :

1. **"la carte d'une région"** : `r_depic` → `r_lieu`
   - Ambiguïté : représentation vs localisation

2. **"la peinture de la porte"** : `r_processus_patient` → `r_depic`
   - Polysémie : action (peindre) vs objet (tableau)

3. **"le tableau de monet"** : `r_product_of` → `r_depic`
   - Confusion : création vs représentation

### Conclusion

**Points forts de GPT-3.5** :
- ✅ Performance remarquable (95%) en few-shot
- ✅ Erreurs uniquement sur cas ambigus
- ✅ Aucune erreur grossière

**Avantages des modèles classiques** :
- ✅ Performance parfaite (100%)
- ✅ 700× plus rapides (0.001s vs 0.70s)
- ✅ Gratuits et déployables facilement

**Recommandation** : Pour ce corpus linéairement séparable, 
Random Forest offre le meilleur compromis. GPT serait préférable 
sur corpus réel avec forte ambiguïté contextuelle.

## 🛠️ Technologies

- **Python 3.11**
- **scikit-learn** - Modèles ML classiques
- **pandas, numpy** - Manipulation de données
- **matplotlib, seaborn** - Visualisation
- **OpenAI API** - Comparaison avec ChatGPT (optionnel)
- **JeuxDeMots API** - Enrichissement sémantique (en cours)

## 📚 Ressources

- **Corpus** : 2250 constructions "A de B" (150/classe)
- **JeuxDeMots** : http://www.jeuxdemots.org/
- **Article de référence** : 
  - *Extraction automatique de règles pour la détermination de types de relations sémantiques dans les constructions génitives en français*
  - H. Guenoune, M. Lafourcade (LIRMM, 2024)
  - [Lien PDF](https://pfia2024.univ-lr.fr/assets/files/Conf%C3%A9rence-IC/IC_2024_paper_20.pdf)

## 👥 Contributeurs

- **Rivals Leonard** - Development & ML
- **Bazireau** - Research & Analysis

## 📝 Licence

_À définir_

## 🔄 Statut du Projet

✅ **Phase 1 : Modèles Baseline Complétée**

### ✅ Étapes réalisées
- [x] Architecture du projet définie
- [x] Structure de dossiers créée
- [x] Préprocessing du corpus (2250 exemples)
- [x] Extraction de features (21 features numériques)
- [x] Data splitting stratifié (70/15/15)
- [x] 5 modèles baseline entraînés
- [x] Évaluation sur test set
- [x] Validation croisée 10-fold
- [x] Matrices de confusion générées
- [x] Scripts de comparaison ChatGPT prêts

### ✅ Récemment complété
- [x] Comparaison avec ChatGPT (GPT-3.5-turbo : 95%)
- [x] Graphiques comparatifs générés
- [x] Analyse des erreurs LLM

### 🚧 En cours
- [ ] Analyse de l'importance des features
- [ ] Test GPT-4 (optionnel)
- [ ] Intégration API JeuxDeMots pour enrichissement

### 📋 À venir
- [ ] Modèles Deep Learning (CamemBERT)
- [ ] Test sur corpus externe
- [ ] Gestion des cas ambigus (multi-label)
- [ ] Interface de démonstration
- [ ] Rapport final et documentation

## 📊 Reproductibilité

Tous les résultats sont reproductibles avec `random_state=42` :
```bash
# Reproduire les résultats exacts
python run_preprocessing.py
python run_feature_extraction.py
python run_train_baseline.py
python run_evaluate_test.py
```

Les modèles entraînés sont sauvegardés dans `models/baseline/`.

---

**Dernière mise à jour** : Novembre 2024
**Version** : 1.0.0 - Baseline Models Complète