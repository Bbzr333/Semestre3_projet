# 💍 Algorithme du Mariage Stable — Gale-Shapley

Ce projet implémente l’**algorithme du mariage stable** (Gale–Shapley) en Python.  
Il comprend trois modules : génération des préférences, appariement stable et mesure de satisfaction.

---

## 🧱 1. Génération des préférences

Script : `preferences.py`  
Génère aléatoirement les préférences d’un ensemble d’étudiants et d’établissements.

### Fichier généré (`prefs_x_y.json`)
- **n** : taille de l’ensemble (modifiable pour les tests)  
- **students** : préférences des étudiants  
- **schools** : préférences des écoles  
- **rank_students** / **rank_schools** : rangs inversés pour accès rapide

### Exécution
```bash
python3 preferences.py --n x --seed y --out /path/to/prefs_x_y.json
```
**Paramètres :**
- `--n` : nombre d’étudiants et d’écoles  
- `--seed` : graine pour reproduire les résultats  
- `--out` : chemin du fichier de sortie  

✅ Résultat : un fichier JSON contenant les préférences complètes.

---

## ⚙️ 2. Algorithme de mariage stable

Script : `gale_shapley.py`  
Implémente l’algorithme Gale–Shapley avec deux modes :
- **DA_E** : les étudiants proposent (étudiants prioritaires)
- **DA_S** : les écoles proposent (écoles prioritaires)

### Exécution
```bash
python3 gale_shapley.py --in prefs_5_123.json --mode DA_E --out match_5_123_E.json
```

### Principe
1. Tous les participants sont libres.  
2. Chaque proposant fait une offre à son meilleur choix encore disponible.  
3. Le receveur accepte temporairement la meilleure offre reçue.  
4. Le processus continue jusqu’à stabilisation : plus aucun proposant libre.

✅ Résultat : un fichier JSON contenant les appariements stables.

---

## 📊 3. Mesure de satisfaction + Évaluation globale

Script : `metrics.py`  
Mesure la satisfaction des étudiants et des établissements à partir d’un matching.

### Fonctions principales
- `ranks_students` / `ranks_schools` : rang du partenaire obtenu (0 = meilleur choix)  
- `stats_from_ranks` : statistiques globales  
  - Moyenne, médiane, écart-type  
  - Taux de top 1 / top 3  
  - Score normalisé [0–1]  
  - Indice de Gini (inégalité)  
  - Histogramme des rangs  
- `is_stable` : vérifie l’absence de paires bloquantes

Script : `eval_matching.py`  
Enchaîne automatiquement les étapes précédentes.

### Exécution
```bash
python3 eval_matching.py \
  --prefs prefs_10_1.json \
  --match match_10_1_E.json \
  --out metrics_10_1_E.json
```

### Étapes effectuées
1. Lecture des fichiers  
2. Extraction des appariements  
3. Calcul des rangs  
4. Statistiques de satisfaction  
5. Vérification de la stabilité  
6. Sauvegarde du rapport final

✅ Résultat : `metrics_10_1_E.json` contenant toutes les statistiques.

---

## 🧠 Auteurs
- Matis — Implémentation Python et documentation  
- Léonard — Tests et validation des résultats  
---

> 💡 Chaque script est exécutable indépendamment. Les jeux de données générés sont compatibles entre eux pour faciliter les tests reproductibles.

