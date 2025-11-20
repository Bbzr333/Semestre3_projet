"""
Script de diagnostic du corpus
Identifie les problèmes de types de données et propose des solutions
"""

import pandas as pd
import numpy as np

def diagnose_corpus(filepath):
    """
    Analyse un fichier CSV et identifie les problèmes potentiels
    """
    print("=" * 60)
    print("🔬 DIAGNOSTIC DU CORPUS")
    print("=" * 60)
    
    # Chargement
    print(f"\n📂 Fichier: {filepath}")
    df = pd.read_csv(filepath)
    print(f"✓ {len(df)} lignes, {len(df.columns)} colonnes")
    
    # Analyse des types
    print("\n📊 ANALYSE DES TYPES DE DONNÉES")
    print("-" * 60)
    
    numeric_cols = df.select_dtypes(include=['int64', 'float64', 'int32', 'float32', 'bool']).columns.tolist()
    text_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    print(f"\n✓ Colonnes numériques ({len(numeric_cols)}):")
    for col in numeric_cols[:15]:
        sample_val = df[col].iloc[0] if len(df) > 0 else 'N/A'
        print(f"    • {col:30s} = {sample_val}")
    if len(numeric_cols) > 15:
        print(f"    ... et {len(numeric_cols) - 15} autres")
    
    print(f"\n⚠️  Colonnes textuelles ({len(text_cols)}):")
    for col in text_cols:
        sample_val = df[col].iloc[0] if len(df) > 0 else 'N/A'
        print(f"    • {col:30s} = {sample_val}")
    
    # Vérification des valeurs manquantes
    print("\n🔍 VALEURS MANQUANTES")
    print("-" * 60)
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    
    if len(missing) > 0:
        print(f"⚠️  {len(missing)} colonnes avec valeurs manquantes:")
        for col, count in missing.items():
            pct = count / len(df) * 100
            print(f"    • {col:30s}: {count} ({pct:.1f}%)")
    else:
        print("✓ Aucune valeur manquante")
    
    # Vérification des valeurs infinies (pour colonnes numériques)
    print("\n🔍 VALEURS INFINIES")
    print("-" * 60)
    inf_found = False
    for col in numeric_cols:
        if np.isinf(df[col]).any():
            inf_count = np.isinf(df[col]).sum()
            print(f"⚠️  {col}: {inf_count} valeurs infinies")
            inf_found = True
    
    if not inf_found:
        print("✓ Aucune valeur infinie détectée")
    
    # Recommandations
    print("\n" + "=" * 60)
    print("💡 RECOMMANDATIONS")
    print("=" * 60)
    
    if len(text_cols) > 4:  # Plus que phrase_originale, type_jdm, _nom1, _nom2
        print("\n⚠️  PROBLÈME DÉTECTÉ:")
        print(f"  Trop de colonnes textuelles ({len(text_cols)}) détectées.")
        print(f"  Colonnes problématiques: {[c for c in text_cols if c not in ['phrase_originale', 'type_jdm', '_nom1', '_nom2']]}")
        print("\n🔧 SOLUTION:")
        print("  1. Re-exécuter run_feature_extraction.py (version corrigée)")
        print("  2. Ou filtrer manuellement dans run_train_baseline.py")
    
    if len(missing) > 0:
        print("\n⚠️  VALEURS MANQUANTES DÉTECTÉES:")
        print("🔧 SOLUTION:")
        print("  Ajouter dans le script de training:")
        print("  X_train = X_train.fillna(0)")
        print("  X_val = X_val.fillna(0)")
    
    if inf_found:
        print("\n⚠️  VALEURS INFINIES DÉTECTÉES:")
        print("🔧 SOLUTION:")
        print("  Remplacer les inf par une valeur limite:")
        print("  X_train = X_train.replace([np.inf, -np.inf], [1e10, -1e10])")
    
    # Statistiques des features numériques
    print("\n📊 STATISTIQUES DES FEATURES NUMÉRIQUES")
    print("-" * 60)
    
    if len(numeric_cols) > 0:
        stats = df[numeric_cols].describe()
        print(stats.iloc[:3])  # mean, std, min
        
        # Détection de colonnes constantes
        constant_cols = [col for col in numeric_cols if df[col].nunique() == 1]
        if constant_cols:
            print(f"\n⚠️  Colonnes constantes (à supprimer): {constant_cols}")
    
    print("\n" + "=" * 60)
    print("✅ DIAGNOSTIC TERMINÉ")
    print("=" * 60)
    
    return {
        'n_rows': len(df),
        'n_cols': len(df.columns),
        'n_numeric': len(numeric_cols),
        'n_text': len(text_cols),
        'n_missing': len(missing),
        'has_inf': inf_found,
        'text_columns': text_cols,
        'numeric_columns': numeric_cols
    }

if __name__ == '__main__':
    import sys
    
    # Fichiers à diagnostiquer
    files_to_check = [
        'data/processed/corpus_preprocessed.csv',
        'data/processed/corpus_with_features.csv',
        'data/processed/train.csv',
    ]
    
    for filepath in files_to_check:
        try:
            print(f"\n{'='*60}")
            print(f"Diagnostic de: {filepath}")
            print('='*60)
            diagnose_corpus(filepath)
            print("\n\n")
        except FileNotFoundError:
            print(f"⚠️  Fichier non trouvé: {filepath}\n")
        except Exception as e:
            print(f"❌ Erreur: {e}\n")