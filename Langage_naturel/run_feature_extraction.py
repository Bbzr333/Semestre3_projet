"""
Script d'extraction de features et split des données
"""

import pandas as pd
import sys
from pathlib import Path

sys.path.append('src')

from src.features.feature_extractor import BasicFeatureExtractor
from src.data.data_splitter import DataSplitter

def main():
    print("🚀 Feature Extraction & Data Splitting")
    print("=" * 60)
    
    # Chargement du corpus preprocessé
    df = pd.read_csv('data/processed/corpus_preprocessed.csv')
    print(f"✓ {len(df)} exemples chargés")
    print(f"✓ {df['type_jdm'].nunique()} classes détectées")
    
    # Extraction des features
    print("\n📊 Extraction des features...")
    extractor = BasicFeatureExtractor()
    features_df = extractor.extract_features(df)
    print(f"✓ {len(features_df.columns)} features extraites")
    
    # Fusion avec le corpus original
    df_full = pd.concat([df.reset_index(drop=True), features_df], axis=1)
    df_full.to_csv('data/processed/corpus_with_features.csv', index=False)
    print(f"✓ Corpus enrichi sauvegardé")
    
    # Split des données
    print("\n🔀 Découpage des données...")
    splitter = DataSplitter(test_size=0.15, val_size=0.15)
    train, val, test, stats = splitter.split_data(df_full)
    splitter.save_splits(train, val, test, stats)
    
    # Affichage des statistiques
    print("\n📈 Statistiques de split:")
    print(f"  Train: {stats['train_samples']} ({stats['train_ratio']:.1%})")
    print(f"  Val:   {stats['val_samples']} ({stats['val_ratio']:.1%})")
    print(f"  Test:  {stats['test_samples']} ({stats['test_ratio']:.1%})")
    
    print("\n✅ Feature extraction terminée !")

if __name__ == '__main__':
    main()