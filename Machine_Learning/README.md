# Projet HAI923 - Modèle CLIP Image-Texte

**Nom:** [À COMPLÉTER]  
**Prénom:** [À COMPLÉTER]  
**N° Carte Étudiant:** [À COMPLÉTER]  
**Numéro de Groupe:** [À COMPLÉTER]

## Description du Projet

Réalisation d'un modèle CLIP (Contrastive Language-Image Pre-training) pour associer des images et des textes.

**Dataset:** Flickr - 4 classes ("bike", "ball", "water", "dog") - 150 paires image-texte par classe (600 total)

## Structure du Projet

```
projet_clip_hai923/
├── notebooks/          # Notebooks Jupyter (.ipynb)
├── data/              # Données Flickr (à télécharger)
├── models/            # Modèles sauvegardés (.pth, .h5)
├── rapport_latex/     # Rapport LaTeX (8 pages max + 2 pages annexes)
├── results/           # Résultats des expériences
│   ├── images/        # Visualisations
│   └── metrics/       # Métriques et courbes
├── utils/             # Fonctions utilitaires
└── README.md          # Ce fichier
```

## Étapes du Projet

### ✅ Étape 1: Classifieur CNN (Images - 4 classes)
- [ ] Architecture CNN de base
- [ ] Entraînement
- [ ] Évaluation

### ✅ Étape 2: Classifieur SmallBERT (Textes - 4 classes)
- [ ] Configuration SmallBERT (ATTENTION: pas de token `<CLS>`)
- [ ] Entraînement
- [ ] Évaluation

### ✅ Étape 3: Modèle CLIP (CŒUR DU PROJET)
- [ ] Encodeur Image (CNN sans classification)
- [ ] Encodeur Texte (SmallBERT sans classification)
- [ ] Intégration loss contrastive
- [ ] Normalisation des embeddings
- [ ] Entraînement CLIP
- [ ] Inférence (Texte→Images top-5, Image→Textes top-5)
- [ ] Sauvegarde/rechargement du modèle

### 🔧 Travail Facultatif
- [ ] Remplacement SmallBERT par DistilBERT
- [ ] Enrichissement des textes courts via LLM

## Points Critiques à Vérifier

- ✅ Dimensions embeddings identiques (image et texte)
- ✅ Normalisation des sorties des encodeurs
- ✅ Loss contrastive intégrée
- ✅ Sauvegarde/rechargement fonctionnel
- ✅ Top-5 avec scores affichés

## Ressources

- Guides LIRMM: https://gite.lirmm.fr/poncelet/deeplearning/
- Template LaTeX: https://www.lirmm.fr/~poncelet/Ressources/template_projet.zip
- Inscription groupe: [lien Google Sheets]

## Rendu

**Format:** `[NUMÉRO_GROUPE].zip`
- `[NUMÉRO_GROUPE].pdf` (rapport LaTeX)
- `[NUMÉRO_GROUPE]_*.ipynb` (notebook)
- `[NUMÉRO_GROUPE]_*.pdf` (notebook en PDF)

**⚠️ ATTENTION:** Pénalité de -4 points si non-respect des consignes de nommage !
