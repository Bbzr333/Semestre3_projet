╔════════════════════════════════════════════════════════════════════════════╗
║                                                                            ║
║                  📋 INDEX COMPLET DU PROJET CLIP HAI923                    ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝

🎯 OBJECTIF: Réalisation d'un modèle CLIP Image-Texte
📚 COURS: HAI923 - Université de Montpellier
👤 POUR: Leonard (et son équipe de 4)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                            📁 FICHIERS CRÉÉS (12)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

┌─ DOCUMENTATION (6 fichiers)
│
├─ 📘 README.md                    │ Description + checklist du projet
├─ 🚀 QUICKSTART.md               │ ⭐⭐⭐ GUIDE PRINCIPAL (À LIRE EN 1ER)
├─ 📋 STRUCTURE.md                │ Récapitulatif de la structure
├─ 💻 COMMANDES_BASH.md           │ ⭐ Toutes les commandes utiles
├─ 📄 INDEX.md                    │ Ce fichier (index général)
└─ 🚫 .gitignore                  │ Fichiers à ignorer (Git)

┌─ CONFIGURATION (3 fichiers)
│
├─ 📦 requirements.txt            │ Dépendances Python
├─ ⚙️  create_structure.sh         │ Script bash de création
└─ 🐍 utils/__init__.py           │ Package Python

┌─ CODE PYTHON (2 fichiers)
│
├─ ⚙️  utils/config.py             │ ⭐ Configuration complète (hyperparamètres)
└─ 🛠️  utils/utils.py              │ ⭐⭐ Fonctions utilitaires (Loss contrastive!)

┌─ NOTEBOOK PRINCIPAL (1 fichier)
│
└─ 📓 TEMPLATE_projet_clip.ipynb  │ ⭐⭐⭐ NOTEBOOK PRINCIPAL (structure 3 étapes)

┌─ RAPPORT LATEX (1 fichier)
│
└─ 📄 main.tex                    │ ⭐ Template rapport (8 pages max)


TOTAL: 13 fichiers créés + 4 dossiers vides (data, models, results, etc.)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                        🎯 ORDRE DE LECTURE RECOMMANDÉ
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1️⃣  INDEX.md (ce fichier)         │ Vue d'ensemble rapide
2️⃣  QUICKSTART.md                 │ Guide complet pas-à-pas ⭐⭐⭐
3️⃣  README.md                     │ Description + checklist
4️⃣  STRUCTURE.md                  │ Détails de la structure
5️⃣  COMMANDES_BASH.md             │ Référence commandes
6️⃣  utils/config.py               │ Comprendre la config
7️⃣  utils/utils.py                │ Comprendre les fonctions
8️⃣  TEMPLATE_projet_clip.ipynb   │ Commencer à coder!

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                          📖 DESCRIPTIONS DÉTAILLÉES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ 🚀 QUICKSTART.md - LE GUIDE PRINCIPAL                                ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

📄 Contenu:
  • Installation (local + Colab)
  • Organisation du travail (3 semaines)
  • Checklist points critiques
  • Conseils rapport LaTeX
  • Problèmes fréquents et solutions
  • Préparation du rendu

🎯 Utilité: C'est LE fichier à suivre de A à Z pour réussir le projet

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ 💻 COMMANDES_BASH.md - RÉFÉRENCE COMPLÈTE                            ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

📄 Contenu:
  • Installation & setup
  • Gestion des données
  • Développement (Jupyter, Python)
  • Gestion des modèles
  • Compilation LaTeX
  • Conversion notebook → PDF
  • Préparation du rendu
  • Debug & tests
  • Monitoring
  • Git (optionnel)
  • Commandes de secours

🎯 Utilité: Référence pour toutes les commandes bash nécessaires

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ 📓 TEMPLATE_projet_clip.ipynb - NOTEBOOK PRINCIPAL                   ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

📄 Contenu:
  • Structure complète des 3 étapes
  • TODO clairs pour chaque section
  • Étape 1: CNN pour images
  • Étape 2: SmallBERT pour textes
  • Étape 3: Modèle CLIP complet
  • Inférence (texte→images, image→textes)
  • Travail facultatif
  • Checklist finale

🎯 Utilité: Template de base pour tout le développement

⚠️  À RENOMMER en [GROUPE]_projet_clip.ipynb avant le rendu!

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ 🛠️  utils/utils.py - FONCTIONS CRITIQUES DÉJÀ IMPLÉMENTÉES           ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

📄 Fonctions clés:
  • ContrastiveLoss ⭐⭐⭐ (Loss CLIP - déjà codée!)
  • normalize_embeddings() ⭐⭐ (Normalisation - CRITIQUE)
  • compute_similarity_matrix() (Calcul similarités)
  • save_model() / load_model() (Sauvegarde)
  • plot_training_history() (Visualisation)
  • display_top_k_results() (Affichage inférence)

🎯 Utilité: Fonctions essentielles pour CLIP, prêtes à l'emploi!
           Ne pas recoder ces fonctions, les utiliser directement.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ ⚙️  utils/config.py - CONFIGURATION CENTRALISÉE                      ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

📄 Configuration:
  • Paramètres généraux (device, seed)
  • Config CNN (étape 1)
  • Config SmallBERT (étape 2)
  • Config CLIP (étape 3) ⭐
  • Config inférence
  • Chemins de sauvegarde

🎯 Utilité: Tous les hyperparamètres au même endroit
           Facile à modifier sans chercher dans le code

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ 📄 main.tex - TEMPLATE RAPPORT LATEX                                 ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

📄 Structure:
  • Introduction
  • Architecture du modèle
  • Implémentation
  • Résultats expérimentaux
  • Conclusion
  • Annexes (max 2 pages)

🎯 Utilité: Structure de base pour le rapport
           À compléter avec vos résultats

⚠️  NE PAS dépasser 8 pages (+ 2 pages annexes max)
⚠️  Utiliser le template LIRMM officiel par-dessus

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                            🎯 POINTS CLÉS À RETENIR
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ PRIORITÉS:

  1. Lire QUICKSTART.md de bout en bout
  2. S'inscrire sur Google Sheets (15 jours max!)
  3. Remplir nom/prénom/n° carte dans tous les fichiers
  4. Focus sur ÉTAPE 3 (CLIP) - cœur du projet
  5. Utiliser les fonctions de utils/utils.py
  6. Respecter les consignes de rendu (-4 pts sinon!)

⚠️  RAPPELS CRITIQUES:

  • Dimensions embeddings IDENTIQUES (image et texte)
  • Normalisation des embeddings ACTIVÉE
  • Loss contrastive INTÉGRÉE (déjà dans utils.py)
  • Sauvegarde/rechargement TESTÉ
  • Top-5 avec SCORES affichés
  • Rapport ≤ 8 pages (+ ≤ 2 pages annexes)
  • Usage abusif IA détectable → oral

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                              📦 LIVRABLES FINAUX
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Format: [NUMÉRO_GROUPE].zip contenant:

  1. [NUMÉRO_GROUPE].pdf               (rapport LaTeX)
  2. [NUMÉRO_GROUPE]_projet_clip.ipynb (notebook)
  3. [NUMÉRO_GROUPE]_projet_clip.pdf   (notebook en PDF)

Exemple pour le groupe 5:
  5.zip
  ├── 5.pdf
  ├── 5_projet_clip.ipynb
  └── 5_projet_clip.pdf

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                            🔗 LIENS IMPORTANTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📝 Inscription (OBLIGATOIRE):
   https://docs.google.com/spreadsheets/d/1y7EP1ev29xr7UxKpD5HD4IFhTQkzEpL1R3RuSuP8tfA/edit

📚 Guides LIRMM:
   https://gite.lirmm.fr/poncelet/deeplearning/

📄 Template LaTeX LIRMM:
   https://www.lirmm.fr/~poncelet/Ressources/template_projet.zip

🌐 pmllatex (Overleaf CNRS):
   https://plmlatex.math.cnrs.fr/login

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                           📊 ESTIMATION TEMPS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Étape 1 (CNN)           : 3-5h      [Medium]
Étape 2 (SmallBERT)     : 3-5h      [Medium]
Étape 3 (CLIP)          : 10-15h    [HIGH ⭐⭐⭐]
Rapport LaTeX           : 5-8h      [HIGH ⭐⭐]
Tests & Debug           : 2-4h      [Medium]
────────────────────────────────────────────
TOTAL                   : 23-37h

⚠️  NE PAS perdre de temps à optimiser les étapes 1 et 2!

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                              🆘 EN CAS DE PROBLÈME
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Consulter QUICKSTART.md → Section "Problèmes Fréquents"
2. Consulter COMMANDES_BASH.md → Commandes de debug
3. Relire le ProjetClip.ipynb fourni (contient astuces)
4. Contacter les encadrants (voir Moodle)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

╔════════════════════════════════════════════════════════════════════════════╗
║                                                                            ║
║                  ✅ Structure créée avec succès!                           ║
║                                                                            ║
║                  🚀 Bon courage Leonard (et ton équipe)!                  ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝

Créé le: 16 décembre 2024
Par: Claude (Assistant IA Anthropic)
