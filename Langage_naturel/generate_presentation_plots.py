"""
Script de génération des graphiques pour la présentation du projet.
Génère :
1. Projection PCA (Analyse critique du corpus)
2. Feature Importance (Justification JDM)
3. Comparaison Performance/Temps (Benchmark LLM vs Local)
4. Matrice de confusion LogReg (Analyse des erreurs)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import os

# Configuration du style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
OUTPUT_DIR = "results/presentation_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Chargement des données
print("Chargement des données...")
try:
    # On utilise le train set pour avoir plus de points pour la PCA
    df = pd.read_csv('data/processed/train.csv')
except FileNotFoundError:
    print("Erreur: 'data/processed/train.csv' introuvable. Assurez-vous d'avoir exécuté le pipeline.")
    exit(1)

# Préparation des features (exclusion des métadonnées)
excluded_cols = ['phrase_originale', 'type_jdm', 'nom1', 'nom2', 'determinant', 
                 'nom1_lemme', 'nom2_lemme', 'definitude', 'est_valide', 'notes']
numeric_cols = df.select_dtypes(include=['int64', 'float64', 'int32', 'float32', 'bool']).columns.tolist()
feature_cols = [col for col in numeric_cols if col not in excluded_cols]

X = df[feature_cols].fillna(0)
y = df['type_jdm']

# --- PLOT 1 : PROJECTION PCA (La séparabilité du corpus) ---
print("Génération PCA...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(12, 8))
scatter = sns.scatterplot(
    x=X_pca[:, 0], y=X_pca[:, 1],
    hue=y, palette="tab20", s=60, alpha=0.8, edgecolor='w'
)
plt.title("Projection PCA du Corpus (2D)\nIllustration de la séparabilité des classes", fontsize=14, fontweight='bold')
plt.xlabel(f"Composante Principale 1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
plt.ylabel(f"Composante Principale 2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., title="Relations")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/1_corpus_separability_pca.png", dpi=300)
plt.close()


# --- PLOT 2 : FEATURE IMPORTANCE (Impact JDM) ---
print("Génération Feature Importance...")
# On ré-entraîne un RF rapide pour avoir les importances fraîches
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
top_n = 15

top_features = [feature_cols[i] for i in indices[:top_n]]
top_importances = importances[indices[:top_n]]

# Code couleur : Bleu pour JDM, Gris pour Basique
colors = ['#3498db' if 'jdm' in f or 'hypernym' in f or 'r_' in f else '#95a5a6' for f in top_features]

plt.figure(figsize=(12, 8))
sns.barplot(x=top_importances, y=top_features, palette=colors)
plt.title("Top 15 Features les plus importantes\n(Bleu = Features JDM / Gris = Features Basiques)", fontsize=14, fontweight='bold')
plt.xlabel("Importance (Gini)")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/2_feature_importance_jdm.png", dpi=300)
plt.close()


# --- PLOT 3 : PERFORMANCE vs TEMPS (LLM vs Local) ---
print("Génération Comparaison LLM...")
# Données basées sur votre rapport
data_perf = {
    'Modèle': ['Random Forest', 'CamemBERT', 'Gradient Boosting', 'SVM Linear', 'Logistic Reg.', 'GPT-3.5-turbo'],
    'Accuracy': [1.00, 1.00, 1.00, 0.947, 0.864, 0.950],
    'Temps (s/ex)': [0.001, 0.05, 0.003, 0.001, 0.001, 0.70],
    'Type': ['ML Classique', 'Deep Learning', 'ML Classique', 'ML Classique', 'ML Classique', 'LLM (API)']
}
df_perf = pd.DataFrame(data_perf)

plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=df_perf, 
    x='Temps (s/ex)', y='Accuracy', 
    hue='Type', style='Type', s=200, palette='viridis'
)

plt.xscale('log') # Échelle log car GPT est beaucoup plus lent
plt.title("Compromis Performance vs Vitesse d'inférence", fontsize=14, fontweight='bold')
plt.xlabel("Temps par exemple (secondes, échelle log)")
plt.ylabel("Accuracy")
plt.ylim(0.80, 1.02)
plt.grid(True, which="both", ls="-", alpha=0.2)

# Annotations
for i, row in df_perf.iterrows():
    plt.text(
        row['Temps (s/ex)'] * 1.1, 
        row['Accuracy'] + (0.005 if row['Modèle'] != 'SVM Linear' else -0.01), 
        row['Modèle'], 
        fontsize=10, fontweight='bold'
    )

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/3_llm_vs_local_benchmark.png", dpi=300)
plt.close()


# --- PLOT 4 : MATRICE DE CONFUSION (Logistic Regression) ---
print("Génération Matrice de Confusion (LogReg)...")
# On entraîne une LogReg pour montrer les erreurs (car RF est parfait)
lr = LogisticRegression(max_iter=1000)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

cm = confusion_matrix(y_test, y_pred, labels=lr.classes_)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(12, 10))
sns.heatmap(
    cm_norm, annot=False, fmt='.2f', cmap='Reds',
    xticklabels=lr.classes_, yticklabels=lr.classes_
)
plt.title("Matrice de Confusion - Régression Logistique\n(Montre les limites de la séparation linéaire)", fontsize=14, fontweight='bold')
plt.ylabel('Vraie classe')
plt.xlabel('Classe prédite')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/4_confusion_matrix_logreg.png", dpi=300)
plt.close()

print(f"Graphiques générés dans {OUTPUT_DIR}")