# Documentation Technique Détaillée

Ce document fournit une analyse approfondie du projet, des méthodologies utilisées, des résultats détaillés et des instructions de reproductibilité complètes.

---

# Pipeline de Classification des Relations Sémantiques

Classification automatique des relations sémantiques dans les constructions génitives françaises ("A de B").

## Description

Ce projet développe un système capable d'identifier automatiquement le type de relation sémantique entre deux noms dans une construction génitive française.

**Exemples** : 
- "la porte de la maison" → **r_holo** (Partie-Tout)
- "le livre de Marie" → **r_own-1** (Possession)
- "le train de Paris" → **r_lieu>origine** (Origine)

## Objectifs

- Classifier 15 types de relations semantiques
- Comparer differentes approches (ML classique, deep learning)
- Evaluer les performances face aux LLM
- Exploiter la ressource JeuxDeMots pour l'enrichissement semantique

## Structure du Projet

```
.
├── data/
│   ├── raw/                    # Corpus initial (2250 exemples)
│   ├── processed/              # Donnees pretraitees (train/val/test)
│   └── generated/              # Donnees generees via JDM
├── src/
│   ├── preprocessing/          # Nettoyage et normalisation
│   ├── features/               # Extraction de features (basique + JDM)
│   ├── models/                 # Modeles de classification
│   ├── data/                   # Generateur de donnees JDM
│   └── utils/                  # Client API JeuxDeMots
├── models/
│   ├── baseline/               # Modeles entraines (.joblib)
│   └── camembert/              # Modele CamemBERT (PyTorch)
├── results/
│   ├── test_results.csv        # Resultats sur test set
│   ├── cross_validation_detailed.csv
│   └── plots/                  # Visualisations
├── run_preprocessing.py        # Pretraitement du corpus
├── run_feature_extraction.py   # Extraction de features
├── run_generate_jdm_data.py    # Generation de donnees via JDM
├── run_train_baseline.py       # Entrainement des modeles
├── run_evaluate_test.py        # Evaluation sur test set
├── run_cross_validation.py     # Validation croisee 10-fold
├── run_chatgpt_simple.py       # Comparaison avec ChatGPT
├── run_train_camembert.py      # Entrainement CamemBERT
├── run_evaluate_camembert.py   # Evaluation CamemBERT
├── demo.py                     # Interface Gradio interactive
├── api.py                      # API FastAPI (REST)
└── EVOLUTION_PROJET.md         # Historique et comparaisons
```

## Installation

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

## Types de Relations (15)

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

## Utilisation

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

## Resultats

### Performance sur Test Set (338 exemples)

| Modèle | Accuracy | F1-Score | Temps/exemple | Erreurs |
|--------|----------|----------|---------------|---------|
| **Random Forest** | **100.0%** | 1.000 | 0.001s | 0/338 |
| **CamemBERT** | **100.0%** | 1.000 | 0.05s | 0/338 |
| **Gradient Boosting** | **100.0%** | 1.000 | 0.003s | 0/338 |
| SVM Linear | 94.7% | 0.945 | 0.001s | 18/338 |
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

**Aucun overfitting detecte** - Performance stable train/test
**Variance faible** - Robustesse confirmee sur tous les folds

### Features Utilisees (102+ avec JDM)

**Features basiques (21):**
- **Morphologiques** : voyelle initiale, terminaison (-e, -s)
- **Lexicales** : detection personne/lieu/temporel/matiere
- **Structurelles** : longueur, ratio, presence determinant

**Features JDM semantiques (81+):**
- **Existence** : `nom1_exists_jdm`, `nom2_exists_jdm`, `both_exist_jdm`
- **Hyperonymes** : `nom1_hypernym_count`, `shared_hypernym_count`, `hypernym_overlap_ratio`
- **Types semantiques** : `nom1_is_person_jdm`, `nom2_is_location_jdm`, etc. (9 categories)
- **Relations** : `nom1_has_r_holo`, `nom2_r_lieu_count`, etc. (10 types de relations)
- **Compatibilite** : `nom2_is_hypernym_of_nom1`, `has_hierarchical_relation`

**Relations JDM extraites:**
```python
RELEVANT_RELATIONS = [
    'r_isa',       # Hyperonymie
    'r_holo',      # Holonymie (partie-tout)
    'r_has_part',  # Meronymie (tout-partie)
    'r_lieu',      # Lieu
    'r_agent',     # Agent
    'r_patient',   # Patient
    'r_carac',     # Caracteristique
    'r_domain',    # Domaine
    'r_hypo',      # Hyponymes (NOUVEAU)
    'r_syn',       # Synonymes (NOUVEAU)
]
```

### Comparaison avec LLM

| Modèle | Accuracy | Temps/exemple | Coût | Échantillon |
|--------|----------|---------------|------|-------------|
| Random Forest | 100.0% | 0.001s | Gratuit | 338 |
| **CamemBERT** | **100.0%** | 0.05s | Gratuit | 338 |
| Gradient Boosting | 100.0% | 0.001s | Gratuit | 338 |
| GPT-3.5-turbo | 95.0% | 0.70s | $0.002/ex | 100 |
| SVM Linear | 94.7% | 0.001s | Gratuit | 338 |

**Résultats GPT-3.5-turbo** :
- Évaluation few-shot (2 exemples/classe)
- 5 erreurs sur 100 exemples
- Erreurs sur cas ambigus : polysémie (peinture), multi-interprétation (carte)
- Performance remarquable mais inférieure aux modèles ensemble

## Analyse

### Pourquoi 100% ?

Le corpus actuel est **linéairement séparable** :

- Patterns morpho-syntaxiques distincts entre classes
- 150 exemples/classe bien equilibres
- Peu d'ambiguite semantique
- Features basiques suffisantes (21 features)

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

## Comparaison Detaillee avec GPT-3.5

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

- Performance remarquable (95%) en few-shot
- Erreurs uniquement sur cas ambigus
- Aucune erreur grossiere

**Avantages des modeles classiques** :

- Performance parfaite (100%)
- 700x plus rapides (0.001s vs 0.70s)
- Gratuits et deployables facilement

**Recommandation** : Pour ce corpus linéairement séparable, 
Random Forest offre le meilleur compromis. GPT serait préférable 
sur corpus réel avec forte ambiguïté contextuelle.

## Integration JeuxDeMots

L'API JeuxDeMots (https://jdm-api.demo.lirmm.fr) est utilisee pour:

### Enrichissement des features
```bash
# Extraction avec features JDM (102 features)
python run_feature_extraction.py

# Extraction sans JDM (21 features basiques)
python run_feature_extraction.py --no-jdm
```

### Generation de donnees d'entrainement
```bash
# Generer 100 exemples/classe depuis JDM
python run_generate_jdm_data.py

# Generer 200 exemples/classe
python run_generate_jdm_data.py --n-per-class 200

# Extraire features du corpus augmente
python run_feature_extraction_augmented.py
```

### Utilisation directe de l'API
```python
from src.utils.jdm_api import get_jdm_api

api = get_jdm_api()
api.term_exists("maison")           # True
api.get_hypernyms("chien")          # ['animal', 'mammifere', ...]
api.get_semantic_types("voiture")   # {'vehicule', 'transport', ...}
api.get_signature("livre")          # Dict complet
```

## Demo Interactive (Gradio)

Une interface web interactive permet de tester les modeles en temps reel.

### Lancement

```bash
python demo.py
# Ouvre http://localhost:7860
```

### Fonctionnalites

**Onglet 1 - Comparaison Modeles:**
- Entrez une phrase genitive ("A de B")
- Comparaison cote-a-cote CamemBERT vs Random Forest
- Affichage des probabilites pour chaque classe
- Extraction JDM en temps reel pour les nouvelles phrases

**Onglet 2 - Knowledge Graph:**
- Entrez plusieurs phrases (une par ligne)
- Visualisation interactive du graphe de relations
- Navigation zoom/pan, nœuds cliquables
- Legende des types de relations

### Extraction JDM Temps Reel

La demo extrait maintenant les features JDM a la volee pour les phrases non presentes dans le corpus :

```
Phrase entree → Preprocessing → Features JDM → Prediction RF
```

Cela permet de tester Random Forest sur **n'importe quelle phrase**, pas seulement celles du test set.

---

## Deep Learning avec CamemBERT

### Entrainement

```bash
# Entrainement avec parametres par defaut (5 epochs)
python run_train_camembert.py

# Personnaliser les hyperparametres
python run_train_camembert.py --epochs 10 --batch_size 32 --lr 3e-5
```

### Evaluation

```bash
python run_evaluate_camembert.py
```

### Utilisation du modele

```python
from src.models.deep_learning.camembert_classifier import CamemBERTClassifier

# Charger le modele entraine
model = CamemBERTClassifier.load('models/camembert/best_model')

# Predire sur du texte brut (pas besoin de features!)
predictions = model.predict([
    "la porte de la maison",
    "le livre de Marie",
    "le vin de Bordeaux"
])
print(predictions)  # ['r_holo', 'r_own-1', 'r_lieu>origine']
```

### Resultats CamemBERT

| Epoch | Train Loss | Train Acc | Val Acc |
|-------|------------|-----------|---------|
| 1 | 2.61 | 40.4% | 99.4% |
| 2 | 1.91 | 98.0% | 100% |
| 3 | 1.42 | 98.8% | 100% |
| 4 | 1.17 | 99.0% | 100% |
| 5 | 1.05 | 99.5% | 100% |

**Test final: 100% accuracy (0 erreurs sur 338 exemples)**

## Technologies

- **Python 3.11**
- **scikit-learn** - Modeles ML classiques
- **PyTorch + Transformers** - Deep Learning (CamemBERT)
- **pandas, numpy** - Manipulation de donnees
- **matplotlib, seaborn** - Visualisation
- **OpenAI API** - Comparaison avec ChatGPT (optionnel)
- **JeuxDeMots API** - Enrichissement semantique (integre)

## Ressources

- **Corpus initial** : 2250 constructions "A de B" (150/classe)
- **Corpus augmente** : 3000+ exemples (avec generation JDM)
- **JeuxDeMots API** : https://jdm-api.demo.lirmm.fr
- **JeuxDeMots** : http://www.jeuxdemots.org/
- **Article de reference** :
  - *Extraction automatique de regles pour la determination de types de relations semantiques dans les constructions genitives en francais*
  - H. Guenoune, M. Lafourcade (LIRMM, 2024)
  - Lien PDF

## Documentation

- **EVOLUTION_PROJET.md** : Historique detaille du projet, comparaison avant/apres JDM
- **API JDM** : Client Python pour JeuxDeMots
- **Feature Extractor** : Extraction des 102+ features

## Contributeurs

- **Rivals Leonard** - Development & ML
- **Bazireau** - Research & Analysis

## Licence

_À définir_

## Statut du Projet

**Phase 1 : Modeles Baseline - Complete**
**Phase 2 : Integration JeuxDeMots - Complete**
**Phase 3 : Deep Learning (CamemBERT) - Complete**
**Phase 4 : Demo Interactive - Complete**
**Phase 5 : Ameliorations Pipeline - Complete**

### Etapes realisees
- [x] Architecture du projet definie
- [x] Structure de dossiers creee
- [x] Preprocessing du corpus (2250 exemples)
- [x] Extraction de features (21 features basiques)
- [x] Data splitting stratifie (70/15/15)
- [x] 5 modeles baseline entraines (RF: 100%, GB: 100%)
- [x] Evaluation sur test set
- [x] Validation croisee 10-fold
- [x] Matrices de confusion generees
- [x] Comparaison avec ChatGPT (GPT-3.5-turbo : 95%)
- [x] Integration API JeuxDeMots REST
- [x] 102+ features (21 basiques + 81+ JDM semantiques)
- [x] Generateur de donnees depuis JDM (1900+ exemples)
- [x] Corpus augmente avec deduplication
- [x] Relations JDM etendues (r_hypo, r_syn)
- [x] Demo Gradio interactive
- [x] Extraction features JDM temps reel
- [x] Knowledge Graph interactif
- [x] Amelioration diversite augmentation

### Recemment complete
- [x] **CamemBERT** : classifieur deep learning (100% accuracy)
- [x] **Pipeline complet** : entrainement + evaluation CamemBERT
- [x] **Inference sur texte brut** : pas besoin de features manuelles
- [x] **Demo interactive Gradio** : comparaison CamemBERT vs Random Forest
- [x] **Extraction JDM temps reel** : prediction sur n'importe quelle phrase
- [x] **Knowledge Graph interactif** : visualisation des relations
- [x] **Amelioration augmentation** : +100 termes, enrichissement JDM dynamique

### A venir
- [ ] Test sur corpus externe (Wikipedia, journaux)
- [ ] Gestion des cas ambigus (multi-label)
- [ ] API FastAPI de production
- [ ] Rapport final et documentation

## Reproductibilite

Tous les resultats sont reproductibles avec `random_state=42` :

```bash
# Pipeline standard (corpus original)
python run_preprocessing.py
python run_feature_extraction.py
python run_train_baseline.py
python run_evaluate_test.py

# Pipeline CamemBERT (deep learning)
python run_train_camembert.py --epochs 5
python run_evaluate_camembert.py

# Pipeline avec augmentation JDM
python run_generate_jdm_data.py --n-per-class 100
python run_feature_extraction_augmented.py
python run_train_baseline.py --data-dir data/processed/augmented
python run_evaluate_test.py --data-dir data/processed/augmented
```

Les modeles entraines sont sauvegardes dans `models/baseline/` et `models/camembert/`.

---