"""
Script d'extraction de features et split des données.

Usage:
    python run_feature_extraction.py              # Avec features JDM
    python run_feature_extraction.py --no-jdm    # Sans features JDM (mode basique)
"""

import pandas as pd
import sys
from pathlib import Path
import numpy as np
import argparse

sys.path.append('src')

from src.features.feature_extractor import BasicFeatureExtractor, EnhancedFeatureExtractor
from src.data.data_splitter import DataSplitter


def main():
    # Parsing des arguments
    parser = argparse.ArgumentParser(description='Extraction de features')
    parser.add_argument('--no-jdm', action='store_true',
                        help='Désactive les features JDM (mode basique uniquement)')
    args = parser.parse_args()

    use_jdm = not args.no_jdm

    print("Feature Extraction & Data Splitting")
    print("=" * 60)
    print(f"Mode: {'Enhanced (basique + JDM)' if use_jdm else 'Basique uniquement'}")

    # Chargement du corpus preprocessé
    df = pd.read_csv('data/processed/corpus_preprocessed.csv')
    print(f"  {len(df)} exemples chargés")
    print(f"  {df['type_jdm'].nunique()} classes détectées")

    # Extraction des features
    print("\nExtraction des features...")
    if use_jdm:
        extractor = EnhancedFeatureExtractor(use_jdm=True)
    else:
        extractor = BasicFeatureExtractor()

    features_df = extractor.extract_features(df)
    print(f"  {len(features_df.columns)} features extraites")
    
    # Fusion avec le corpus original
    df_full = pd.concat([df.reset_index(drop=True), features_df], axis=1)
    df_full.to_csv('data/processed/corpus_with_features.csv', index=False)
    print(f"  Corpus enrichi sauvegardé")

    # Split des données
    print("\nDecoupage des donnees...")
    splitter = DataSplitter(test_size=0.15, val_size=0.15)
    train, val, test, stats = splitter.split_data(df_full)
    splitter.save_splits(train, val, test, stats)

    # Affichage des statistiques
    print("\nStatistiques de split:")
    print(f"  Train: {stats['train_samples']} ({stats['train_ratio']:.1%})")
    print(f"  Val:   {stats['val_samples']} ({stats['val_ratio']:.1%})")
    print(f"  Test:  {stats['test_samples']} ({stats['test_ratio']:.1%})")

    print("\nFeature extraction terminee !")

if __name__ == '__main__':
    main()