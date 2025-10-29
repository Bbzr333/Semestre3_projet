# Pipeline de Classification des Relations Sémantiques

Classification automatique des relations sémantiques dans les constructions génitives françaises ("A de B").

## 📋 Description

Ce projet vise à développer un système capable d'identifier automatiquement le type de relation sémantique entre deux noms dans une construction génitive française.

**Exemple** : 
- "la porte de la maison" → Relation : **Partie-Tout**
- "le livre de Marie" → Relation : **Possession**
- "le train de Paris" → Relation : **Origine**

## 🎯 Objectifs

- Classifier 16 types de relations sémantiques
- Comparer différentes approches (symboliques, ML classique, deep learning)
- Évaluer les performances face aux LLM
- Exploiter la ressource JeuxDeMots pour l'enrichissement

## 🗂️ Structure du Projet

```
.
├── data/           # Corpus et données
├── src/            # Code source
├── models/         # Modèles entraînés
├── notebooks/      # Expérimentations Jupyter
├── configs/        # Fichiers de configuration
├── scripts/        # Scripts utilitaires
├── tests/          # Tests unitaires
└── results/        # Résultats et métriques
```

## 🚀 Installation

```bash
# Cloner le dépôt
git clone https://github.com/USERNAME/REPO_NAME.git
cd REPO_NAME

# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Installer les dépendances
pip install -r requirements.txt
```

## 📊 Types de Relations

Le système classifie 16 relations sémantiques :

1. **Agent** - "le discours du président"
2. **Bénéficiaire** - "le cadeau de Marie"
3. **Caractérisation** - "l'homme au chapeau"
4. **Contenu** - "un verre d'eau"
5. **Destination** - "le train pour Paris"
6. **Lieu** - "les gens de la ville"
7. **Lien Social** - "l'ami de Pierre"
8. **Matière** - "une table de bois"
9. **Origine** - "le vin de Bordeaux"
10. **Partie-Tout** - "la porte de la maison"
11. **Possession** - "le livre de Marie"
12. **Producteur** - "le tableau de Picasso"
13. **Propriété** - "un homme de courage"
14. **Quantité** - "un kilo de pommes"
15. **Temps** - "les jours d'hiver"
16. **Thème** - "un livre d'histoire"

## 🔧 Utilisation

_Section à compléter_

```python
# Exemple basique (à venir)
from src.models import RelationClassifier

classifier = RelationClassifier()
result = classifier.predict("la porte de la maison")
print(result)  # → "Partie-Tout"
```

## 📈 Résultats

_Section à compléter au fur et à mesure des expérimentations_

| Modèle | Accuracy | F1-Score | Temps d'inférence |
|--------|----------|----------|-------------------|
| ...    | ...      | ...      | ...               |

## 🛠️ Technologies

- Python 3.8+
- scikit-learn
- PyTorch
- Transformers (CamemBERT)
- JeuxDeMots API
- NumPy, Pandas

## 📚 Ressources

- **Corpus** : _[à décrire]_
- **JeuxDeMots** : http://www.jeuxdemots.org/
- **Articles de référence** : _[à ajouter]_

## 👥 Contributeurs

_[Vos noms]_

## 📝 Licence

_[À définir]_

## 🔄 Statut du Projet

🚧 **En développement actif**

### Étapes réalisées
- [x] Architecture du projet définie
- [x] Structure de dossiers créée
- [ ] Préprocessing du corpus
- [ ] Extraction de features
- [ ] Modèles baseline
- [ ] Évaluation comparative

---

**Dernière mise à jour** : Octobre 2025
