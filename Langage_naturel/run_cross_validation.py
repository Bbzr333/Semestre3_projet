"""
Validation Croisée 10-Fold
Vérification de la robustesse et détection de data leakage
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, cross_validate, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

sys.path.append('src')

def plot_cv_scores(cv_results, save_path):
    """Visualise les scores de validation croisée"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    models = list(cv_results.keys())
    
    # 1. Boxplot des scores par modèle
    ax = axes[0, 0]
    data = [cv_results[model]['test_accuracy'] for model in models]
    bp = ax.boxplot(data, labels=models, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Distribution des Scores (10 Folds)', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.set_xticklabels(models, rotation=45, ha='right')
    
    # 2. Scores par fold pour chaque modèle
    ax = axes[0, 1]
    for model in models:
        scores = cv_results[model]['test_accuracy']
        ax.plot(range(1, 11), scores, marker='o', label=model, linewidth=2)
    ax.set_xlabel('Fold', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Scores par Fold', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xticks(range(1, 11))
    
    # 3. Mean ± Std
    ax = axes[1, 0]
    means = [cv_results[model]['test_accuracy'].mean() for model in models]
    stds = [cv_results[model]['test_accuracy'].std() for model in models]
    colors = plt.cm.viridis(np.linspace(0, 1, len(models)))
    bars = ax.bar(models, means, yerr=stds, capsize=5, color=colors, alpha=0.7)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Mean Accuracy ± Std', fontsize=14, fontweight='bold')
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    
    # Ajouter les valeurs sur les barres
    for i, (mean, std) in enumerate(zip(means, stds)):
        ax.text(i, mean + std + 0.01, f'{mean:.3f}±{std:.3f}', 
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 4. Temps d'entraînement par fold
    ax = axes[1, 1]
    for model in models:
        times = cv_results[model]['fit_time']
        ax.plot(range(1, 11), times, marker='s', label=model, linewidth=2)
    ax.set_xlabel('Fold', fontsize=12)
    ax.set_ylabel('Temps (secondes)', fontsize=12)
    ax.set_title('Temps d\'Entraînement par Fold', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xticks(range(1, 11))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  📊 Graphique sauvegardé: {save_path}")

def analyze_fold_difficulty(cv_results):
    """Analyse la difficulté de chaque fold"""
    print("\n🔍 ANALYSE DE LA DIFFICULTÉ DES FOLDS")
    print("=" * 70)
    
    # Calculer le score moyen de tous les modèles pour chaque fold
    all_models = list(cv_results.keys())
    n_folds = len(cv_results[all_models[0]]['test_accuracy'])    
    fold_difficulties = []
    for fold_idx in range(n_folds):
        fold_scores = [cv_results[model]['test_accuracy'][fold_idx] for model in all_models]
        avg_score = np.mean(fold_scores)
        std_score = np.std(fold_scores)
        fold_difficulties.append({
            'fold': fold_idx + 1,
            'avg_score': avg_score,
            'std_score': std_score,
            'min_score': np.min(fold_scores),
            'max_score': np.max(fold_scores)
        })
    
    df_folds = pd.DataFrame(fold_difficulties)
    df_folds = df_folds.sort_values('avg_score')
    
    print("\n📋 Classement des Folds (du plus difficile au plus facile):")
    print(df_folds.to_string(index=False))
    
    # Identifier les folds problématiques
    mean_difficulty = df_folds['avg_score'].mean()
    std_difficulty = df_folds['avg_score'].std()
    
    difficult_folds = df_folds[df_folds['avg_score'] < mean_difficulty - std_difficulty]
    
    if len(difficult_folds) > 0:
        print(f"\n⚠️  Folds particulièrement difficiles:")
        for _, fold in difficult_folds.iterrows():
            print(f"  • Fold {fold['fold']}: {fold['avg_score']:.3f} (écart-type: {fold['std_score']:.3f})")
    else:
        print(f"\n✅ Tous les folds ont une difficulté similaire (variance faible)")
    
    return df_folds

def main():
    print("=" * 70)
    print("🔄 VALIDATION CROISÉE 10-FOLD")
    print("=" * 70)
    
    # Chargement de tout le dataset (train + val + test)
    print("\n📂 Chargement des données...")
    train = pd.read_csv('data/processed/train.csv')
    val = pd.read_csv('data/processed/val.csv')
    test = pd.read_csv('data/processed/test.csv')
    
    # Combiner tous les datasets
    full_data = pd.concat([train, val, test], ignore_index=True)
    print(f"✓ Dataset complet: {len(full_data)} exemples")
    
    # Préparation des features
    excluded_cols = ['phrase_originale', 'nom1_lemme', 'nom2_lemme', 'type_jdm', 'definitude',
                     'nom1', 'nom2', 'determinant']
    numeric_cols = full_data.select_dtypes(include=['int64', 'float64', 'int32', 'float32', 'bool']).columns.tolist()
    feature_cols = [col for col in numeric_cols if col not in excluded_cols]
    
    X = full_data[feature_cols]
    y = full_data['type_jdm']
    
    # Nettoyage
    constant_cols = X.columns[X.std() == 0].tolist()
    if constant_cols:
        X = X.drop(columns=constant_cols)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    print(f"✓ Features: {X.shape[1]}")
    print(f"✓ Classes: {y.nunique()}")
    print(f"✓ Distribution: {y.value_counts().min()} à {y.value_counts().max()} exemples/classe")
    
    # Encodage des labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Configuration de la validation croisée
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    
    # Modèles à tester
    models = {
        'Logistic Regression': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(max_iter=1000, random_state=42))
        ]),
        'Random Forest': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
        ]),
        'SVM Linear': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', SVC(kernel='linear', random_state=42))
        ]),
        'SVM RBF': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', SVC(kernel='rbf', random_state=42))
        ]),
        'Gradient Boosting': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', GradientBoostingClassifier(random_state=42))
        ])
    }
    
    # Validation croisée pour chaque modèle
    cv_results = {}
    
    print("\n" + "=" * 70)
    print("🔄 EXÉCUTION DE LA VALIDATION CROISÉE")
    print("=" * 70)
    
    for model_name, model in models.items():
        print(f"\n📊 {model_name}")
        print("-" * 70)
        
        # Cross-validation avec métriques détaillées
        scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
        scores = cross_validate(
            model, X, y_encoded, 
            cv=cv, 
            scoring=scoring,
            return_train_score=True,
            n_jobs=-1
        )
        
        cv_results[model_name] = scores
        
        # Statistiques
        test_acc = scores['test_accuracy']
        train_acc = scores['train_accuracy']
        
        print(f"  📈 Test Accuracy:")
        print(f"    • Mean:  {test_acc.mean():.3f}")
        print(f"    • Std:   {test_acc.std():.3f}")
        print(f"    • Min:   {test_acc.min():.3f} (Fold {test_acc.argmin() + 1})")
        print(f"    • Max:   {test_acc.max():.3f} (Fold {test_acc.argmax() + 1})")
        
        print(f"\n  📊 Train Accuracy:")
        print(f"    • Mean:  {train_acc.mean():.3f}")
        print(f"    • Std:   {train_acc.std():.3f}")
        
        # Détection d'overfitting
        overfit = train_acc.mean() - test_acc.mean()
        print(f"\n  🔍 Overfitting:")
        print(f"    • Train - Test: {overfit:+.3f}")
        if overfit > 0.1:
            print(f"    • ⚠️  Overfitting détecté!")
        elif overfit < -0.05:
            print(f"    • ⚠️  Underfitting possible")
        else:
            print(f"    • ✅ Bonne généralisation")
        
        # Autres métriques
        print(f"\n  📋 Autres Métriques (Test):")
        print(f"    • Precision: {scores['test_precision_macro'].mean():.3f} ± {scores['test_precision_macro'].std():.3f}")
        print(f"    • Recall:    {scores['test_recall_macro'].mean():.3f} ± {scores['test_recall_macro'].std():.3f}")
        print(f"    • F1-Score:  {scores['test_f1_macro'].mean():.3f} ± {scores['test_f1_macro'].std():.3f}")
        
        # Temps
        print(f"\n  ⏱️  Temps:")
        print(f"    • Fit:   {scores['fit_time'].mean():.2f}s ± {scores['fit_time'].std():.2f}s")
        print(f"    • Score: {scores['score_time'].mean():.2f}s ± {scores['score_time'].std():.2f}s")
    
    # Analyse de la difficulté des folds
    fold_analysis = analyze_fold_difficulty(cv_results)
    
    # Comparaison finale
    print("\n" + "=" * 70)
    print("🏆 COMPARAISON FINALE")
    print("=" * 70)
    
    comparison = []
    for model_name in cv_results.keys():
        scores = cv_results[model_name]
        comparison.append({
            'Modèle': model_name,
            'CV Accuracy': f"{scores['test_accuracy'].mean():.3f} ± {scores['test_accuracy'].std():.3f}",
            'Min': f"{scores['test_accuracy'].min():.3f}",
            'Max': f"{scores['test_accuracy'].max():.3f}",
            'Train Acc': f"{scores['train_accuracy'].mean():.3f}",
            'Temps (s)': f"{scores['fit_time'].mean():.2f}"
        })
    
    df_comparison = pd.DataFrame(comparison)
    df_comparison = df_comparison.sort_values('CV Accuracy', ascending=False)
    print("\n" + df_comparison.to_string(index=False))
    
    # Visualisation
    results_dir = Path('results')
    plots_dir = results_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    plot_path = plots_dir / 'cross_validation_10fold.png'
    plot_cv_scores(cv_results, plot_path)
    
    # Sauvegarder les résultats détaillés
    detailed_results = []
    for model_name, scores in cv_results.items():
        for fold_idx in range(10):
            detailed_results.append({
                'model': model_name,
                'fold': fold_idx + 1,
                'test_accuracy': scores['test_accuracy'][fold_idx],
                'train_accuracy': scores['train_accuracy'][fold_idx],
                'test_precision': scores['test_precision_macro'][fold_idx],
                'test_recall': scores['test_recall_macro'][fold_idx],
                'test_f1': scores['test_f1_macro'][fold_idx],
                'fit_time': scores['fit_time'][fold_idx]
            })
    
    df_detailed = pd.DataFrame(detailed_results)
    csv_path = results_dir / 'cross_validation_detailed.csv'
    df_detailed.to_csv(csv_path, index=False)
    print(f"\n💾 Résultats détaillés: {csv_path}")
    
    # Analyse finale
    print("\n" + "=" * 70)
    print("🔬 ANALYSE FINALE")
    print("=" * 70)
    
    # Vérifier la cohérence avec les résultats précédents
    test_results = pd.read_csv('results/test_results.csv', index_col=0)
    
    print("\n📊 Comparaison CV vs Test Set:")
    print(f"{'Modèle':<25} {'CV Mean':<12} {'Test Set':<12} {'Diff':<10} {'Cohérence'}")
    print("-" * 70)
    
    for model_name in cv_results.keys():
        cv_mean = cv_results[model_name]['test_accuracy'].mean()
        
        # Mapper les noms de modèles
        model_key = model_name.lower().replace(' ', '_')
        if model_key in test_results.index:
            test_acc = test_results.loc[model_key, 'accuracy']
            diff = cv_mean - test_acc
            
            if abs(diff) < 0.05:
                status = "✅ Cohérent"
            elif diff > 0:
                status = "⚠️ CV > Test"
            else:
                status = "⚠️ Test > CV"
            
            print(f"{model_name:<25} {cv_mean:<12.3f} {test_acc:<12.3f} {diff:+.3f}     {status}")
    
    # Conclusion
    print("\n" + "=" * 70)
    print("💡 CONCLUSIONS")
    print("=" * 70)
    
    best_model = max(cv_results.keys(), key=lambda m: cv_results[m]['test_accuracy'].mean())
    best_score = cv_results[best_model]['test_accuracy'].mean()
    best_std = cv_results[best_model]['test_accuracy'].std()
    
    print(f"\n🥇 Meilleur modèle: {best_model}")
    print(f"   Accuracy: {best_score:.3f} ± {best_std:.3f}")
    
    if best_score > 0.98 and best_std < 0.02:
        print(f"\n✅ CORPUS TRÈS FACILE")
        print(f"   • Score élevé ({best_score:.1%}) avec faible variance")
        print(f"   • Les features actuelles suffisent largement")
        print(f"   • Recommandation: Augmenter la complexité du corpus")
    elif best_std > 0.05:
        print(f"\n⚠️  VARIANCE ÉLEVÉE")
        print(f"   • Performance instable entre folds")
        print(f"   • Certains folds plus difficiles")
        print(f"   • Recommandation: Vérifier la distribution des classes")
    else:
        print(f"\n✅ PERFORMANCE ROBUSTE")
        print(f"   • Score stable entre folds")
        print(f"   • Bonne généralisation")
    
    print("\n" + "=" * 70)
    print("✅ VALIDATION CROISÉE TERMINÉE")
    print("=" * 70)

if __name__ == '__main__':
    main()