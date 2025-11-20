"""
Évaluation des modèles sur le test set
Analyse détaillée des performances et matrices de confusion
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

sys.path.append('src')

from models.baseline_models import BaselineClassifier
from evaluation.evaluator import ModelEvaluator

def plot_confusion_matrix(cm, labels, model_name, save_path):
    """Affiche et sauvegarde la matrice de confusion"""
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Nombre de prédictions'})
    plt.title(f'Matrice de Confusion - {model_name}', fontsize=16, fontweight='bold')
    plt.ylabel('Vraie Classe', fontsize=12)
    plt.xlabel('Classe Prédite', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  💾 Matrice sauvegardée: {save_path}")

def analyze_errors(y_true, y_pred, X_test, test_df, model_name):
    """Analyse les erreurs de prédiction"""
    errors = y_true != y_pred
    n_errors = errors.sum()
    
    if n_errors == 0:
        print(f"  🎯 Aucune erreur ! Performance parfaite sur le test set.")
        return
    
    print(f"  ❌ {n_errors} erreurs sur {len(y_true)} ({n_errors/len(y_true)*100:.1f}%)")
    
    # Analyser les confusions les plus fréquentes
    error_df = pd.DataFrame({
        'true': y_true[errors],
        'pred': y_pred[errors],
        'phrase': test_df.loc[errors, 'phrase_originale'].values if 'phrase_originale' in test_df.columns else [''] * n_errors
    })
    
    confusion_pairs = error_df.groupby(['true', 'pred']).size().sort_values(ascending=False).head(5)
    
    if len(confusion_pairs) > 0:
        print(f"\n  🔍 Top 5 confusions:")
        for (true_label, pred_label), count in confusion_pairs.items():
            print(f"    • {true_label} → {pred_label}: {count} fois")
            # Afficher un exemple
            example = error_df[(error_df['true'] == true_label) & (error_df['pred'] == pred_label)].iloc[0]
            if example['phrase']:
                print(f"      Exemple: \"{example['phrase']}\"")
    
    return error_df

def main():
    print("=" * 70)
    print("🧪 ÉVALUATION SUR LE TEST SET")
    print("=" * 70)
    
    # Chargement du test set
    print("\n📂 Chargement des données...")
    test = pd.read_csv('data/processed/test.csv')
    print(f"✓ Test: {len(test)} exemples")
    
    # Préparation des features (même traitement que training)
    excluded_cols = ['phrase_originale', 'nom1_lemme', 'nom2_lemme', 'type_jdm', 'definitude',
                     'nom1', 'nom2', 'determinant']
    numeric_cols = test.select_dtypes(include=['int64', 'float64', 'int32', 'float32', 'bool']).columns.tolist()
    feature_cols = [col for col in numeric_cols if col not in excluded_cols]
    
    X_test = test[feature_cols]
    y_test = test['type_jdm']
    
    # Nettoyage (même que training)
    constant_cols = X_test.columns[X_test.std() == 0].tolist()
    if constant_cols:
        X_test = X_test.drop(columns=constant_cols)
    X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    print(f"✓ Features: {X_test.shape[1]}")
    print(f"✓ Classes: {y_test.nunique()}")
    
    # Créer les dossiers de sortie
    results_dir = Path('results')
    plots_dir = results_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Liste des modèles à évaluer
    models_dir = Path('models/baseline')
    model_files = list(models_dir.glob('*.joblib'))
    
    if not model_files:
        print("\n❌ Aucun modèle trouvé dans models/baseline/")
        return
    
    print(f"\n✓ {len(model_files)} modèles trouvés")
    
    # Évaluation de chaque modèle
    all_results = {}
    
    for model_path in sorted(model_files):
        model_name = model_path.stem
        print(f"\n{'='*70}")
        print(f"📊 Évaluation: {model_name}")
        print(f"{'='*70}")
        
        try:
            # Charger le modèle
            classifier = BaselineClassifier.load(model_path)
            
            # Prédictions
            y_pred = classifier.predict(X_test)
            
            # Métriques globales
            evaluator = ModelEvaluator()
            metrics = evaluator.evaluate(classifier, X_test, y_test)
            
            print(f"\n📈 Métriques Globales:")
            print(f"  • Accuracy:  {metrics['accuracy']:.3f}")
            print(f"  • Precision: {metrics['precision_macro']:.3f}")
            print(f"  • Recall:    {metrics['recall_macro']:.3f}")
            print(f"  • F1-Score:  {metrics['f1_macro']:.3f}")
            
            # Matrice de confusion
            cm = metrics['confusion_matrix']
            labels = classifier.label_encoder.classes_
            
            plot_path = plots_dir / f'confusion_matrix_{model_name}.png'
            plot_confusion_matrix(cm, labels, model_name, plot_path)
            
            # Analyse des erreurs
            print(f"\n🔍 Analyse des Erreurs:")
            error_df = analyze_errors(y_test, y_pred, X_test, test, model_name)
            
            # Sauvegarder les erreurs
            if error_df is not None and len(error_df) > 0:
                error_path = results_dir / f'errors_{model_name}.csv'
                error_df.to_csv(error_path, index=False)
                print(f"  💾 Erreurs sauvegardées: {error_path}")
            
            # Stocker les résultats
            all_results[model_name] = {
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision_macro'],
                'recall': metrics['recall_macro'],
                'f1_score': metrics['f1_macro'],
                'n_errors': (y_test != y_pred).sum()
            }
            
            # Rapport détaillé par classe
            print(f"\n📋 Rapport par Classe:")
            print(metrics['classification_report'])
            
        except Exception as e:
            print(f"  ❌ Erreur: {e}")
            all_results[model_name] = {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'n_errors': len(y_test),
                'error': str(e)
            }
    
    # Comparaison finale
    print(f"\n{'='*70}")
    print("🏆 COMPARAISON FINALE SUR TEST SET")
    print(f"{'='*70}")
    
    df_results = pd.DataFrame(all_results).T
    df_results = df_results.sort_values('accuracy', ascending=False)
    
    print("\n📊 Classement des Modèles:")
    print(df_results[['accuracy', 'f1_score', 'precision', 'recall', 'n_errors']].to_string())
    
    # Sauvegarder les résultats
    results_path = results_dir / 'test_results.csv'
    df_results.to_csv(results_path)
    print(f"\n💾 Résultats sauvegardés: {results_path}")
    
    # Meilleur modèle
    best_model = df_results.index[0]
    best_acc = df_results.iloc[0]['accuracy']
    best_errors = int(df_results.iloc[0]['n_errors'])
    
    print(f"\n{'='*70}")
    print(f"🥇 MEILLEUR MODÈLE: {best_model}")
    print(f"{'='*70}")
    print(f"  • Accuracy: {best_acc:.3f}")
    print(f"  • Erreurs:  {best_errors}/{len(y_test)}")
    print(f"  • Taux d'erreur: {best_errors/len(y_test)*100:.1f}%")
    
    # Vérification de l'overfitting
    val_results = pd.read_csv('results/baseline_comparison.csv', index_col=0)
    
    print(f"\n🔍 Analyse Overfitting:")
    print(f"{'Modèle':<25} {'Val Acc':<10} {'Test Acc':<10} {'Diff':<10} {'Status'}")
    print("-" * 70)
    
    for model in df_results.index:
        if model in val_results.index:
            val_acc = val_results.loc[model, 'accuracy']
            test_acc = df_results.loc[model, 'accuracy']
            diff = val_acc - test_acc
            
            if diff > 0.05:
                status = "⚠️ Overfitting"
            elif diff < -0.05:
                status = "🎯 Bonne généralisation"
            else:
                status = "✅ Stable"
            
            print(f"{model:<25} {val_acc:<10.3f} {test_acc:<10.3f} {diff:+.3f}     {status}")
    
    print(f"\n{'='*70}")
    print("✅ ÉVALUATION TERMINÉE")
    print(f"{'='*70}")
    print(f"\n📁 Résultats disponibles dans:")
    print(f"  • {results_path}")
    print(f"  • {plots_dir}/")

if __name__ == '__main__':
    main()