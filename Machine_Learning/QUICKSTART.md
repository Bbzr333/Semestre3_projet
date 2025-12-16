# 🚀 Guide de Démarrage Rapide - Projet CLIP HAI923

## 📋 Avant de Commencer

### 1. Inscription (OBLIGATOIRE)
- [ ] S'inscrire sur le Google Sheets: https://docs.google.com/spreadsheets/d/1y7EP1ev29xr7UxKpD5HD4IFhTQkzEpL1R3RuSuP8tfA/edit
- [ ] **DEADLINE: 15 jours après démarrage** (sinon note = 0)
- [ ] Noter votre numéro de groupe

### 2. Constitution de l'Équipe
- [ ] 4 personnes obligatoire (sauf accord préalable)
- [ ] Personne en 1ère colonne = déposant du projet

### 3. Ressources Essentielles
- [ ] Télécharger le template LaTeX: https://www.lirmm.fr/~poncelet/Ressources/template_projet.zip
- [ ] Accéder au notebook ProjetClip.ipynb (contient codes + données)
- [ ] Consulter les guides: https://gite.lirmm.fr/poncelet/deeplearning/

---

## 🛠️ Installation

### Option 1: Environnement Local
```bash
# Cloner/créer le projet
cd /chemin/vers/projet_clip_hai923

# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou: venv\Scripts\activate  # Windows

# Installer les dépendances
pip install -r requirements.txt
```

### Option 2: Google Colab (Recommandé pour GPU)
```python
# Monter Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Installer les packages
!pip install transformers torch torchvision

# Uploader les fichiers depuis votre Drive
```

---

## 📂 Organisation du Travail

### Semaine 1: Étapes 1 & 2 (Classifieurs de base)
**Objectif:** Avoir des classifieurs fonctionnels (ne PAS optimiser)

1. **Étape 1: CNN Images**
   - Créer architecture CNN simple
   - Entraîner sur 4 classes (bike, ball, water, dog)
   - Sauvegarder le modèle
   - **Ne PAS perdre de temps à optimiser**

2. **Étape 2: SmallBERT Textes**
   - Charger SmallBERT
   - **ATTENTION:** Pas de token `<CLS>` → choisir méthode de résumé
   - Entraîner sur 4 classes
   - Sauvegarder le modèle

### Semaine 2: Étape 3 (CLIP - CŒUR DU PROJET)
**Objectif:** Créer le modèle CLIP fonctionnel

3. **Étape 3: Modèle CLIP**
   - Créer encodeur image (CNN sans classification)
   - Créer encodeur texte (SmallBERT sans classification)
   - **VÉRIFIER:** Dimensions embeddings identiques
   - **VÉRIFIER:** Normalisation des embeddings
   - Intégrer loss contrastive
   - Entraîner le modèle CLIP
   - **TESTER:** Sauvegarde/rechargement
   - Implémenter inférence (texte→images, image→textes)
   - Afficher top-5 avec scores

### Semaine 3: Finalisation
- Travail facultatif (si temps)
- Rédaction rapport LaTeX
- Préparation des livrables

---

## ✅ Checklist Points Critiques

### Architecture CLIP
- [ ] Dimensions embeddings **IDENTIQUES** pour image et texte
- [ ] Normalisation des sorties **ACTIVÉE**
- [ ] Loss contrastive **INTÉGRÉE**
- [ ] Pas de fonction d'activation sur la projection finale

### Encodeurs
- [ ] CNN: couches de classification **RETIRÉES**
- [ ] SmallBERT: méthode de résumé de phrase **DÉFINIE**
- [ ] Projection vers espace latent sans activation

### Entraînement
- [ ] Sauvegarde du modèle **TESTÉE**
- [ ] Rechargement du modèle **TESTÉ**

### Inférence
- [ ] Texte → Images: top-5 avec **SCORES AFFICHÉS**
- [ ] Image → Textes: top-5 avec **SCORES AFFICHÉS**

---

## 📝 Conseils pour le Rapport LaTeX

### Structure Recommandée
1. **Introduction** (0.5 page)
   - Contexte (modèles multimodaux)
   - Objectif
   - **NE PAS** paraphraser l'énoncé

2. **Architecture** (2 pages)
   - Vue d'ensemble CLIP
   - Encodeur image (modifications CNN)
   - Encodeur texte (gestion SmallBERT sans `<CLS>`)
   - Loss contrastive

3. **Implémentation** (2 pages)
   - Préparation données
   - Étapes 1, 2, 3 (focus sur étape 3)
   - Hyperparamètres

4. **Résultats** (2.5 pages)
   - Métriques
   - Courbes d'entraînement
   - Exemples requêtes (avec scores!)
   - Analyse qualitative

5. **Conclusion** (0.5 page)
   - Récapitulatif
   - Limitations
   - Perspectives

6. **Annexes** (max 2 pages)
   - Code important
   - Résultats complémentaires

### Ce qu'il NE FAUT PAS faire
- ❌ Décrire l'objectif du projet (tout le monde le connaît)
- ❌ Décrire les données en détail
- ❌ Copier-coller de l'énoncé
- ❌ Utiliser l'IA pour rédiger des sections entières

### Ce qu'il FAUT faire
- ✅ Focus sur vos choix techniques
- ✅ Justifier vos décisions
- ✅ Analyser vos résultats
- ✅ Valoriser votre travail

---

## 📦 Préparation du Rendu

### Nomenclature des Fichiers
```
[NUMÉRO_GROUPE].zip
├── [NUMÉRO_GROUPE].pdf (rapport LaTeX)
├── [NUMÉRO_GROUPE]_projet_clip.ipynb
└── [NUMÉRO_GROUPE]_projet_clip.pdf
```

**Exemple pour le groupe 5:**
```
5.zip
├── 5.pdf
├── 5_projet_clip.ipynb
└── 5_projet_clip.pdf
```

### Contenu Obligatoire dans TOUS les Fichiers
- Nom, prénom, numéro carte étudiant de **CHAQUE** membre
- Numéro de groupe

### Vérification Finale
- [ ] Tous les fichiers contiennent nom/prénom/n° carte
- [ ] Nomenclature correcte (numéro groupe)
- [ ] Rapport ≤ 8 pages (+ ≤ 2 pages annexes)
- [ ] Rapport LaTeX avec template officiel
- [ ] Notebook .ipynb ET .pdf
- [ ] Archive .zip créée
- [ ] Déposant = personne 1ère colonne fichier inscription

**⚠️ PÉNALITÉ: -4 points si non-respect des consignes!**

---

## 🆘 Problèmes Fréquents

### "SmallBERT n'a pas de token `<CLS>`"
**Solution:** Utiliser mean pooling sur tous les tokens
```python
# Exemple de mean pooling
outputs = bert_model(input_ids, attention_mask)
last_hidden = outputs.last_hidden_state  # (batch, seq_len, hidden)
mean_pooled = (last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(1, keepdim=True)
```

### "Dimensions embeddings incompatibles"
**Solution:** Vérifier que la projection finale a la même taille pour image et texte
```python
# Les deux doivent avoir embedding_dim identique
image_projection = nn.Linear(cnn_features, embedding_dim)  # ex: 512
text_projection = nn.Linear(bert_hidden, embedding_dim)    # ex: 512
```

### "Le modèle ne se sauvegarde/charge pas"
**Solution:** Tester immédiatement après le premier entraînement
```python
# Sauvegarder
torch.save(model.state_dict(), 'model.pth')

# Charger
model.load_state_dict(torch.load('model.pth'))
model.eval()
```

### "Pas de GPU disponible"
**Solution:** Utiliser Google Colab avec GPU gratuit
```python
# Vérifier GPU
import torch
print(torch.cuda.is_available())  # Doit être True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

---

## 🎯 Rappel des Priorités

### 🔥 PRIORITÉ MAXIMALE
1. Modèle CLIP fonctionnel (étape 3)
2. Inférence avec top-5 + scores
3. Sauvegarde/rechargement testé
4. Rapport LaTeX (8 pages max)

### ⚡ Important
5. Classifieurs étapes 1 & 2 (fonctionnels, pas optimisés)
6. Respect des consignes de rendu

### 🌟 Bonus (si temps)
7. DistilBERT à la place de SmallBERT
8. Enrichissement des textes via LLM

---

## 📞 Contact et Support

- **Encadrants:** Voir Moodle
- **Issues courantes:** Relire le ProjetClip.ipynb
- **Guides:** https://gite.lirmm.fr/poncelet/deeplearning/

**⚠️ ATTENTION À L'USAGE DE L'IA:**
- Les encadrants ont fait faire le projet par plusieurs IA
- Usage abusif détectable → oral obligatoire
- ✅ Autorisé: Correction syntaxe/formulation rapport
- ❌ Interdit: Rédaction sections entières / code complet par IA

---

Bon courage! 🚀
