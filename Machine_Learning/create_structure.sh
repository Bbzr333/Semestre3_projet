#!/bin/bash

# Script de création de la structure du projet CLIP HAI923
# Auteur: Claude
# Usage: bash create_structure.sh

echo "=========================================="
echo "Création de la structure du projet CLIP"
echo "=========================================="

# Répertoire de base
BASE_DIR="projet_clip_hai923"

# Créer la structure de répertoires
echo "📁 Création des répertoires..."
mkdir -p "$BASE_DIR"/{notebooks,data,models,rapport_latex,results/{images,metrics},utils}

# Vérifier la structure
echo ""
echo "✅ Structure créée:"
tree "$BASE_DIR" 2>/dev/null || find "$BASE_DIR" -print | sed -e 's;[^/]*/;|____;g;s;____|; |;g'

echo ""
echo "=========================================="
echo "Fichiers à créer manuellement:"
echo "=========================================="
echo "1. README.md - Description du projet"
echo "2. requirements.txt - Dépendances Python"
echo "3. .gitignore - Fichiers à ignorer"
echo "4. QUICKSTART.md - Guide de démarrage"
echo "5. utils/config.py - Configuration"
echo "6. utils/utils.py - Fonctions utilitaires"
echo "7. utils/__init__.py - Package Python"
echo "8. notebooks/TEMPLATE_projet_clip.ipynb - Notebook principal"
echo "9. rapport_latex/main.tex - Rapport LaTeX"

echo ""
echo "=========================================="
echo "Prochaines étapes:"
echo "=========================================="
echo "1. Compléter vos informations (nom, prénom, n° carte)"
echo "2. S'inscrire sur le Google Sheets (15 jours max!)"
echo "3. Télécharger les données depuis ProjetClip.ipynb"
echo "4. Commencer par les étapes 1 et 2 (classifieurs)"
echo "5. Focus sur l'étape 3 (CLIP - cœur du projet)"
echo ""
echo "⚠️  N'oubliez pas: -4 points si non-respect des consignes de nommage!"
echo ""
