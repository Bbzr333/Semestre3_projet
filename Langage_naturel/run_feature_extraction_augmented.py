"""
Script d'extraction de features pour le corpus augmente.

Usage:
    python run_feature_extraction_augmented.py              # Avec features JDM
    python run_feature_extraction_augmented.py --no-jdm    # Sans features JDM
"""

import pandas as pd
import sys
from pathlib import Path
import argparse

sys.path.append('src')

from src.features.feature_extractor import BasicFeatureExtractor, EnhancedFeatureExtractor
from src.data.data_splitter import DataSplitter


def main():
    # Parsing des arguments
    parser = argparse.ArgumentParser(description='Extraction de features (corpus augmente)')
    parser.add_argument('--no-jdm', action='store_true',
                        help='Desactive les features JDM')
    parser.add_argument('--input', type=str, default='data/processed/corpus_augmented.csv',
                        help='Fichier d\'entree (corpus augmente)')
    args = parser.parse_args()

    use_jdm = not args.no_jdm

    print("Feature Extraction - Corpus Augmente")
    print("=" * 60)
    print(f"Mode: {'Enhanced (basique + JDM)' if use_jdm else 'Basique uniquement'}")
    print(f"Input: {args.input}")

    # Verification du fichier d'entree
    if not Path(args.input).exists():
        print(f"\nErreur: Fichier non trouve: {args.input}")
        print("Executez d'abord: python run_generate_jdm_data.py")
        return

    # Chargement du corpus augmente
    df = pd.read_csv(args.input)
    print(f"\n  {len(df)} exemples charges")
    print(f"  {df['type_jdm'].nunique()} classes detectees")

    # Distribution des classes
    print("\nDistribution des classes:")
    print(df['type_jdm'].value_counts().to_string())

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
    output_features = 'data/processed/corpus_augmented_with_features.csv'
    df_full.to_csv(output_features, index=False)
    print(f"  Corpus enrichi sauvegarde: {output_features}")

    # Split des donnees
    print("\nDecoupage des donnees...")
    splitter = DataSplitter(test_size=0.15, val_size=0.15)
    train, val, test, stats = splitter.split_data(df_full)

    # Sauvegarde dans un sous-dossier pour ne pas ecraser les originaux
    output_dir = Path('data/processed/augmented')
    output_dir.mkdir(parents=True, exist_ok=True)

    train.to_csv(output_dir / 'train.csv', index=False)
    val.to_csv(output_dir / 'val.csv', index=False)
    test.to_csv(output_dir / 'test.csv', index=False)

    # Sauvegarde des stats
    import json
    with open(output_dir / 'split_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)

    # Affichage des statistiques
    print("\nStatistiques de split:")
    print(f"  Train: {stats['train_samples']} ({stats['train_ratio']:.1%})")
    print(f"  Val:   {stats['val_samples']} ({stats['val_ratio']:.1%})")
    print(f"  Test:  {stats['test_samples']} ({stats['test_ratio']:.1%})")

    print(f"\nFichiers crees dans {output_dir}/:")
    print("  - train.csv")
    print("  - val.csv")
    print("  - test.csv")
    print("  - split_stats.json")

    print("\nFeature extraction terminee !")
    print("\nProchaine etape:")
    print("  python run_train_baseline.py --data-dir data/processed/augmented")


if __name__ == '__main__':
    main()
