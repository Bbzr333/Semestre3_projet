# 📋 RÉCAPITULATIF DE LA STRUCTURE DU PROJET

## ✅ Fichiers Créés

```
projet_clip_hai923/
│
├── 📄 README.md                      # Description générale du projet + checklist
├── 📄 QUICKSTART.md                  # Guide de démarrage rapide
├── 📄 requirements.txt               # Dépendances Python
├── 📄 .gitignore                     # Fichiers à ignorer (Git)
├── 📄 create_structure.sh            # Script de création de structure
│
├── 📁 notebooks/
│   └── 📓 TEMPLATE_projet_clip.ipynb # Template notebook principal avec structure complète
│
├── 📁 data/                          # [À REMPLIR] Données Flickr (images + textes)
│
├── 📁 models/                        # [À REMPLIR] Modèles sauvegardés (.pth)
│   ├── cnn_classifier.pth            # (à créer) Modèle CNN étape 1
│   ├── smallbert_classifier.pth      # (à créer) Modèle SmallBERT étape 2
│   └── clip_model.pth                # (à créer) Modèle CLIP final étape 3
│
├── 📁 rapport_latex/
│   └── 📄 main.tex                   # Template rapport LaTeX (8 pages max)
│
├── 📁 results/
│   ├── 📁 images/                    # [À REMPLIR] Visualisations
│   └── 📁 metrics/                   # [À REMPLIR] Métriques et courbes
│
└── 📁 utils/
    ├── 📄 __init__.py                # Package Python
    ├── 📄 config.py                  # Configuration (hyperparamètres, chemins)
    └── 📄 utils.py                   # Fonctions utilitaires (loss contrastive, etc.)
```

---

## 📌 Utilité de Chaque Fichier

### Fichiers Racine

**README.md**
- Description du projet
- Structure détaillée
- Checklist des 3 étapes
- Points critiques à vérifier

**QUICKSTART.md** ⭐
- Guide de démarrage rapide
- Organisation du travail par semaines
- Checklist points critiques
- Problèmes fréquents et solutions
- Conseils rapport LaTeX
- Préparation du rendu

**requirements.txt**
- Liste des dépendances Python
- Versions spécifiées
- Installation: `pip install -r requirements.txt`

**.gitignore**
- Fichiers à ne pas versionner
- Données, modèles, __pycache__, etc.

**create_structure.sh**
- Script bash pour recréer la structure
- Utile pour démarrer sur une nouvelle machine

---

### Dossier notebooks/

**TEMPLATE_projet_clip.ipynb** ⭐⭐⭐
- **FICHIER PRINCIPAL POUR LE CODE**
- Structure complète des 3 étapes
- Sections bien organisées
- TODO clairs pour chaque partie
- Checklist finale intégrée
- **À RENOMMER en `[GROUPE]_projet_clip.ipynb` avant le rendu**

---

### Dossier utils/

**config.py**
- Toutes les configurations centralisées
- Hyperparamètres pour CNN, SmallBERT, CLIP
- Chemins de sauvegarde
- Device (CPU/GPU)
- Classes du dataset

**utils.py** ⭐
- **Fonctions critiques déjà implémentées:**
  - `ContrastiveLoss`: Loss pour CLIP (fournie)
  - `normalize_embeddings()`: Normalisation (CRITIQUE)
  - `compute_similarity_matrix()`: Calcul similarités
  - `save_model()` / `load_model()`: Sauvegarde/chargement
  - `plot_training_history()`: Visualisation courbes
  - `display_top_k_results()`: Affichage résultats inférence

**__init__.py**
- Initialisation du package utils
- Import automatique des configs et utils

---

### Dossier rapport_latex/

**main.tex** ⭐
- Template complet pour le rapport
- Structure pré-remplie avec TODO
- Sections principales:
  1. Introduction
  2. Architecture du Modèle
  3. Implémentation
  4. Résultats Expérimentaux
  5. Conclusion
  6. Annexes (max 2 pages)
- **IMPORTANT:** Utiliser le template officiel LIRMM par-dessus

---

## 🎯 Workflow Recommandé

### 1️⃣ Avant de Commencer
```bash
# Vérifier la structure
ls -R projet_clip_hai923/

# S'inscrire sur le Google Sheets (OBLIGATOIRE sous 15 jours)
# Noter votre numéro de groupe
```

### 2️⃣ Préparation
```bash
# Installer dépendances
pip install -r requirements.txt

# Remplir vos informations dans:
# - README.md
# - TEMPLATE_projet_clip.ipynb
# - main.tex
```

### 3️⃣ Développement (suivre QUICKSTART.md)
```python
# Semaine 1: Étapes 1 & 2
# - notebooks/TEMPLATE_projet_clip.ipynb → Sections Étape 1 et 2

# Semaine 2: Étape 3 (CŒUR DU PROJET)
# - notebooks/TEMPLATE_projet_clip.ipynb → Section Étape 3
# - Utiliser utils/utils.py (ContrastiveLoss déjà fournie!)

# Semaine 3: Finalisation
# - Rapport LaTeX (rapport_latex/main.tex)
# - Préparer les livrables
```

### 4️⃣ Préparation du Rendu
```bash
# 1. Renommer le notebook
mv notebooks/TEMPLATE_projet_clip.ipynb notebooks/[GROUPE]_projet_clip.ipynb

# 2. Exporter notebook en PDF
jupyter nbconvert --to pdf notebooks/[GROUPE]_projet_clip.ipynb

# 3. Compiler le rapport LaTeX
cd rapport_latex/
pdflatex main.tex
mv main.pdf ../[GROUPE].pdf

# 4. Créer l'archive
cd ..
zip -r [GROUPE].zip [GROUPE].pdf [GROUPE]_projet_clip.ipynb [GROUPE]_projet_clip.pdf

# 5. Vérifier le contenu
unzip -l [GROUPE].zip
```

---

## ⚠️ POINTS CRITIQUES À NE PAS OUBLIER

### ✅ Dans le Code
- [ ] Dimensions embeddings **IDENTIQUES** (image et texte)
- [ ] Normalisation des embeddings **ACTIVÉE**
- [ ] Loss contrastive **INTÉGRÉE** (déjà dans utils.py)
- [ ] Sauvegarde/rechargement **TESTÉ**
- [ ] Inférence top-5 avec **SCORES AFFICHÉS**

### ✅ Dans le Rapport
- [ ] **8 pages MAX** (+ 2 pages annexes max)
- [ ] Template officiel LIRMM utilisé
- [ ] NE PAS décrire objectif/données (focus sur le travail)
- [ ] Nom/prénom/n° carte de TOUS les membres

### ✅ Pour le Rendu
- [ ] Nomenclature: `[GROUPE].zip`, `[GROUPE].pdf`, `[GROUPE]_*.ipynb`
- [ ] 3 fichiers: rapport.pdf + notebook.ipynb + notebook.pdf
- [ ] Déposant = personne 1ère colonne fichier inscription

**⚠️ PÉNALITÉ: -4 points si non-respect!**

---

## 🆘 Aide et Support

### Problèmes Courants
Voir section "Problèmes Fréquents" dans **QUICKSTART.md**

### Ressources
- **ProjetClip.ipynb** (fourni): Codes + astuces + accès données
- **Guides LIRMM**: https://gite.lirmm.fr/poncelet/deeplearning/
  - Guide vision (pour CNN)
  - Guide texte (pour SmallBERT)
- **Template LaTeX**: https://www.lirmm.fr/~poncelet/Ressources/template_projet.zip

### Rappel IA
- ✅ Autorisé: Correction syntaxe rapport
- ❌ Interdit: Rédaction sections / code par IA
- ⚠️  Les encadrants détectent l'usage abusif → oral obligatoire

---

## 📊 Estimation du Temps

| Tâche | Temps estimé | Priorité |
|-------|--------------|----------|
| Étape 1: CNN | 3-5h | Medium |
| Étape 2: SmallBERT | 3-5h | Medium |
| Étape 3: CLIP | 10-15h | **HIGH** |
| Rapport LaTeX | 5-8h | **HIGH** |
| Tests & Debug | 2-4h | Medium |
| **TOTAL** | **23-37h** | |

**⚠️ Conseil:** Ne pas perdre de temps à optimiser les étapes 1 et 2 !

---

## 🎓 Bon Courage !

Suivez le **QUICKSTART.md** pour un guide pas-à-pas détaillé.

Structure créée le: $(date)
