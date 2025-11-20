"""
Entraînement des modèles baseline
Version corrigée avec filtrage robuste des features
"""

import pandas as pd
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
import numpy as np

sys.path.append('src')

from src.models.baseline_models import BaselineClassifier
from src.evaluation.evaluator import ModelEvaluator

def main():
    print("🚀 Entraînement des modèles baseline")
    print("=" * 60)
    
    # Chargement des données
    train = pd.read_csv('data/processed/train.csv')
    val = pd.read_csv('data/processed/val.csv')
    
    # ========== CORRECTION : Filtrage robuste des features ==========
    # On exclut explicitement les colonnes non-numériques
    excluded_cols = ['phrase_originale', 'nom1_lemme', 'nom2_lemme', 'type_jdm', 'definitude']
    
    # Sélection automatique des colonnes numériques uniquement
    numeric_cols = train.select_dtypes(include=['int64', 'float64', 'int32', 'float32', 'bool']).columns.tolist()
    feature_cols = [col for col in numeric_cols if col not in excluded_cols]
    
    # Diagnostic des colonnes
    print(f"\n🔍 Diagnostic des colonnes:")
    print(f"  Total colonnes: {len(train.columns)}")
    print(f"  Colonnes numériques: {len(numeric_cols)}")
    print(f"  Features sélectionnées: {len(feature_cols)}")
    
    # Afficher quelques exemples de features
    print(f"\n📊 Exemples de features:")
    for i, col in enumerate(feature_cols[:8]):
        print(f"    {i+1}. {col}")
    if len(feature_cols) > 8:
        print(f"    ... et {len(feature_cols) - 8} autres")
    
    X_train = train[feature_cols]
    y_train = train['type_jdm']
    X_val = val[feature_cols]
    y_val = val['type_jdm']

    # === NETTOYAGE DES DONNÉES ===
    constant_cols = X_train.columns[X_train.std() == 0].tolist()
    if constant_cols:
        print(f"  🗑️  {len(constant_cols)} colonnes constantes supprimées")
        X_train = X_train.drop(columns=constant_cols)
        X_val = X_val.drop(columns=constant_cols)
    X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0)
    X_val = X_val.replace([np.inf, -np.inf], np.nan).fillna(0)
    print(f"  ✓ Features finales: {X_train.shape[1]}")
    # ==============================
    
    # Vérification et traitement des valeurs manquantes
    if X_train.isnull().any().any():
        print(f"\n⚠️  Valeurs manquantes détectées!")
        missing_summary = X_train.isnull().sum()[X_train.isnull().sum() > 0]
        print(missing_summary)
        X_train = X_train.fillna(0)
        X_val = X_val.fillna(0)
        print(f"✓ Valeurs manquantes remplacées par 0")
    
    # Vérification des valeurs infinies
    if not pd.api.types.is_numeric_dtype(X_train.values.flatten()):
        print(f"⚠️  Attention: types non-numériques détectés!")
        print(X_train.dtypes)
    
    print(f"\n✓ Train: {len(X_train)} exemples, {len(feature_cols)} features")
    print(f"✓ Val: {len(X_val)} exemples")
    print(f"✓ Classes: {y_train.nunique()}")
    
    # ============================================================
    
    # Entraînement de tous les modèles
    models_dir = Path('models/baseline')
    models_dir.mkdir(parents=True, exist_ok=True)
    
    results_dir = Path('results')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    for model_name in BaselineClassifier.MODELS.keys():
        print(f"\n📊 Entraînement: {model_name}")
        print("-" * 60)
        
        try:
            # Train
            classifier = BaselineClassifier(model_name=model_name)
            classifier.train(X_train, y_train)
            
            print(f"  ✓ Temps d'entraînement: {classifier.training_history['training_time_seconds']:.2f}s")
            
            # Eval sur validation
            evaluator = ModelEvaluator()
            metrics = evaluator.evaluate(classifier, X_val, y_val)
            
            print(f"  ✓ Accuracy: {metrics['accuracy']:.3f}")
            print(f"  ✓ F1-score (macro): {metrics['f1_macro']:.3f}")
            print(f"  ✓ Precision (macro): {metrics['precision_macro']:.3f}")
            print(f"  ✓ Recall (macro): {metrics['recall_macro']:.3f}")
            
            # Sauvegarde du modèle
            model_path = models_dir / f'{model_name}.joblib'
            classifier.save(model_path)
            print(f"  💾 Modèle sauvegardé: {model_path}")
            
            results[model_name] = {
                'accuracy': metrics['accuracy'],
                'f1_macro': metrics['f1_macro'],
                'precision_macro': metrics['precision_macro'],
                'recall_macro': metrics['recall_macro'],
                'training_time': classifier.training_history['training_time_seconds']
            }
            
        except Exception as e:
            print(f"  ❌ Erreur: {e}")
            results[model_name] = {
                'accuracy': 0.0,
                'f1_macro': 0.0,
                'precision_macro': 0.0,
                'recall_macro': 0.0,
                'training_time': 0.0,
                'error': str(e)
            }
    
    # Comparaison des modèles
    print("\n" + "=" * 60)
    print("📊 COMPARAISON DES MODÈLES")
    print("=" * 60)
    
    df_results = pd.DataFrame(results).T
    df_results = df_results.sort_values('accuracy', ascending=False)
    
    print("\n🏆 Classement par Accuracy:")
    print(df_results[['accuracy', 'f1_macro', 'precision_macro', 'recall_macro', 'training_time']].to_string())
    
    # Sauvegarde des résultats
    results_path = results_dir / 'baseline_comparison.csv'
    df_results.to_csv(results_path)
    print(f"\n💾 Résultats sauvegardés: {results_path}")
    
    # Meilleur modèle
    best_model = df_results.index[0]
    best_acc = df_results.iloc[0]['accuracy']
    print(f"\n🥇 Meilleur modèle: {best_model} (Accuracy: {best_acc:.3f})")
    
    print("\n" + "=" * 60)
    print("✅ ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS!")
    print("=" * 60)

if __name__ == '__main__':
    main()