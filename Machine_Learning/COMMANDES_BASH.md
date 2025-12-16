# 💻 COMMANDES BASH - AIDE-MÉMOIRE PROJET CLIP

## 📦 INSTALLATION & SETUP

### Extraire l'archive du projet
```bash
# Si tu as téléchargé l'archive .tar.gz
tar -xzf projet_clip_hai923.tar.gz
cd projet_clip_hai923

# Vérifier la structure
ls -R
```

### Créer un environnement virtuel Python
```bash
# Créer l'environnement
python3 -m venv venv

# Activer (Linux/Mac)
source venv/bin/activate

# Activer (Windows)
venv\Scripts\activate

# Installer les dépendances
pip install -r requirements.txt

# Vérifier l'installation
pip list
```

### Setup Google Colab (si pas de GPU local)
```python
# Dans un notebook Colab
from google.colab import drive
drive.mount('/content/drive')

# Naviguer vers ton dossier
%cd /content/drive/MyDrive/projet_clip_hai923

# Installer les packages
!pip install transformers torch torchvision

# Vérifier GPU
import torch
print(f"GPU disponible: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
```

---

## 📊 GESTION DES DONNÉES

### Télécharger et organiser les données Flickr
```bash
# Créer les sous-dossiers pour chaque classe
cd data/
mkdir -p images/{bike,ball,water,dog}
mkdir -p captions

# Vérifier la structure
tree . # ou: ls -R

# Compter les images par classe (devrait être 150 chacune)
ls images/bike/ | wc -l
ls images/ball/ | wc -l
ls images/water/ | wc -l
ls images/dog/ | wc -l
```

### Vérifier les données
```bash
# Nombre total d'images (devrait être 600)
find data/images -name "*.jpg" -o -name "*.png" | wc -l

# Taille totale des données
du -sh data/
```

---

## 🔧 DÉVELOPPEMENT

### Lancer Jupyter Notebook
```bash
# Depuis le répertoire racine du projet
jupyter notebook

# Ouvrir: notebooks/TEMPLATE_projet_clip.ipynb
```

### Exécuter un script Python
```bash
# Si tu veux tester un module séparément
python -c "from utils.config import *; print(f'Device: {DEVICE}')"
python -c "from utils.utils import ContrastiveLoss; print('Loss OK')"
```

### Vérifier l'import des modules
```bash
# Tester les imports depuis la racine
cd projet_clip_hai923/
python3 << EOF
import sys
sys.path.append('./utils')
from config import *
from utils import *
print("✅ Tous les imports fonctionnent!")
print(f"Device: {DEVICE}")
print(f"Classes: {CLASSES}")
EOF
```

---

## 💾 GESTION DES MODÈLES

### Sauvegarder un modèle (dans ton code)
```python
import torch

# Sauvegarder
torch.save(model.state_dict(), 'models/mon_modele.pth')

# Charger
model.load_state_dict(torch.load('models/mon_modele.pth'))
model.eval()
```

### Vérifier les modèles sauvegardés
```bash
# Lister les modèles
ls -lh models/

# Taille de chaque modèle
du -h models/*.pth
```

---

## 📄 RAPPORT LATEX

### Compiler le rapport LaTeX
```bash
cd rapport_latex/

# Compilation simple
pdflatex main.tex

# Compilation complète (avec bibliographie)
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex

# Vérifier le PDF
ls -lh main.pdf
```

### Utiliser pmllatex (Overleaf CNRS) - RECOMMANDÉ
```bash
# 1. Aller sur https://plmlatex.math.cnrs.fr/login
# 2. Se connecter avec adresse institutionnelle
# 3. Créer nouveau projet
# 4. Upload main.tex et images
# 5. Compiler en ligne
```

### Nettoyer les fichiers temporaires LaTeX
```bash
cd rapport_latex/
rm -f *.aux *.log *.out *.toc *.bbl *.blg *.synctex.gz *.fdb_latexmk *.fls
```

---

## 📓 NOTEBOOK

### Convertir notebook en PDF
```bash
# Méthode 1: Via jupyter
jupyter nbconvert --to pdf notebooks/TEMPLATE_projet_clip.ipynb

# Méthode 2: Via navigateur
# Ouvrir le notebook → File → Download as → PDF via LaTeX
```

### Nettoyer les outputs du notebook
```bash
# Installer nbconvert si pas déjà fait
pip install nbconvert

# Nettoyer tous les outputs
jupyter nbconvert --clear-output --inplace notebooks/TEMPLATE_projet_clip.ipynb

# Vérifier
jupyter nbconvert --to notebook --execute notebooks/TEMPLATE_projet_clip.ipynb
```

---

## 📦 PRÉPARATION DU RENDU

### Renommer les fichiers selon le numéro de groupe
```bash
# Exemple pour le groupe 5
GROUPE=5

# Renommer le notebook
mv notebooks/TEMPLATE_projet_clip.ipynb notebooks/${GROUPE}_projet_clip.ipynb

# Convertir en PDF
jupyter nbconvert --to pdf notebooks/${GROUPE}_projet_clip.ipynb

# Renommer le rapport
cd rapport_latex/
pdflatex main.tex
cd ..
mv rapport_latex/main.pdf ${GROUPE}.pdf

# Vérifier
ls -lh ${GROUPE}*
```

### Créer l'archive finale
```bash
GROUPE=5  # Remplacer par ton numéro de groupe

# Créer l'archive .zip avec les 3 fichiers requis
zip ${GROUPE}.zip \
    ${GROUPE}.pdf \
    notebooks/${GROUPE}_projet_clip.ipynb \
    notebooks/${GROUPE}_projet_clip.pdf

# Vérifier le contenu
unzip -l ${GROUPE}.zip

# Vérifier la taille
ls -lh ${GROUPE}.zip
```

### Checklist finale avant le rendu
```bash
# Vérifier que tous les fichiers nécessaires sont présents
echo "Vérification de l'archive..."
unzip -l ${GROUPE}.zip | grep -E "\.pdf|\.ipynb"

# Vérifier que les noms/prénoms sont présents
echo "Vérifier les informations dans:"
echo "1. ${GROUPE}.pdf (ouvrir et vérifier première page)"
echo "2. ${GROUPE}_projet_clip.ipynb (ouvrir et vérifier première cellule)"

# Taille de l'archive (ne devrait pas être énorme)
du -h ${GROUPE}.zip
```

---

## 🧪 DEBUG & TESTS

### Tester la loss contrastive
```bash
python3 << EOF
import torch
from utils.utils import ContrastiveLoss

# Créer des embeddings factices
batch_size = 4
embed_dim = 512
img_emb = torch.randn(batch_size, embed_dim)
txt_emb = torch.randn(batch_size, embed_dim)

# Tester la loss
loss_fn = ContrastiveLoss(temperature=0.07)
loss = loss_fn(img_emb, txt_emb)

print(f"✅ Loss contrastive: {loss.item():.4f}")
EOF
```

### Vérifier les dimensions
```bash
python3 << EOF
from utils.config import *

print(f"CNN config:")
print(f"  Image size: {CNN_CONFIG['img_size']}")
print(f"  Batch size: {CNN_CONFIG['batch_size']}")

print(f"\nCLIP config:")
print(f"  Embedding dim: {CLIP_CONFIG['embedding_dim']}")
print(f"  Temperature: {CLIP_CONFIG['temperature']}")

print(f"\n✅ Vérifier que embedding_dim est identique pour image et texte!")
EOF
```

### Tester l'installation PyTorch + GPU
```bash
python3 << EOF
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA disponible: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Mémoire GPU: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
EOF
```

---

## 🔍 MONITORING & LOGS

### Surveiller l'entraînement
```bash
# Si tu logs dans un fichier
tail -f training.log

# Ou utiliser tqdm dans le code Python
from tqdm import tqdm
for epoch in tqdm(range(num_epochs), desc="Training"):
    # ...
```

### Vérifier l'utilisation GPU (si disponible)
```bash
# En temps réel
watch -n 1 nvidia-smi

# Une fois
nvidia-smi
```

### Espace disque
```bash
# Vérifier l'espace disque restant
df -h

# Taille du projet
du -sh projet_clip_hai923/

# Taille par dossier
du -h projet_clip_hai923/* | sort -hr
```

---

## 🗂️ GIT (optionnel mais recommandé)

### Initialiser Git
```bash
cd projet_clip_hai923/
git init
git add .
git commit -m "Initial commit - Structure du projet"
```

### Ignorer les gros fichiers
```bash
# .gitignore est déjà créé, mais pour ajouter:
echo "data/" >> .gitignore
echo "models/*.pth" >> .gitignore
echo "*.pyc" >> .gitignore
```

### Sauvegardes régulières
```bash
# Commit régulier
git add .
git commit -m "Étape 1: CNN terminé"

# Voir l'historique
git log --oneline

# Revenir à un commit précédent (si besoin)
git checkout <commit-hash>
```

---

## 📊 STATISTIQUES UTILES

### Compter les lignes de code
```bash
# Python uniquement
find . -name "*.py" | xargs wc -l | tail -1

# Notebook (approximatif)
jupyter nbconvert --to script notebooks/*.ipynb
find . -name "*.txt" | xargs wc -l
rm -f notebooks/*.txt  # Nettoyer
```

### Temps d'exécution
```bash
# Dans le code Python
import time
start = time.time()
# ... ton code ...
print(f"Temps: {time.time() - start:.2f}s")
```

---

## ⚠️ COMMANDES DE SECOURS

### Si tout plante, recréer la structure
```bash
# Sauvegarder ton travail actuel
cp -r projet_clip_hai923 projet_clip_hai923_backup

# Recréer depuis le script
cd /chemin/vers/structure
bash create_structure.sh
```

### Réinstaller les packages
```bash
# Désinstaller tout
pip freeze > requirements_old.txt
pip uninstall -r requirements_old.txt -y

# Réinstaller proprement
pip install -r requirements.txt
```

### Problème avec Jupyter
```bash
# Réinstaller Jupyter
pip install --upgrade jupyter notebook

# Ou utiliser JupyterLab (plus moderne)
pip install jupyterlab
jupyter lab
```

---

## 📞 AIDE RAPIDE

Si quelque chose ne fonctionne pas:

1. **Vérifier les imports:** `python -c "from utils.config import *"`
2. **Vérifier PyTorch:** `python -c "import torch; print(torch.__version__)"`
3. **Vérifier les données:** `ls -R data/`
4. **Lire les messages d'erreur complets**
5. **Consulter QUICKSTART.md section "Problèmes Fréquents"**

---

Bon courage Leonard! 🚀

N'hésite pas à revenir vers moi si tu as besoin d'aide sur des commandes spécifiques.
