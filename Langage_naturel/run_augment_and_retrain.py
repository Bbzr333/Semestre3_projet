"""
Pipeline complet d'augmentation et ré-entraînement pour améliorer la généralisation.

Ce script:
1. Génère des données via JDM avec diversité maximale
2. Ajoute des paraphrases syntaxiques
3. Ajoute du bruit pour la robustesse
4. Génère des cas ambigus
5. Fusionne avec le corpus original
6. Extrait les features
7. Ré-entraîne les modèles
8. Teste sur corpus externe

Usage:
    python run_augment_and_retrain.py
    python run_augment_and_retrain.py --n-per-class 150 --noise-ratio 0.2
"""

import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

sys.path.append('src')

from src.data.jdm_data_generator import JDMDataGenerator
from src.preprocessing.preprocessor import GenitivePreprocessor
from src.features.feature_extractor import EnhancedFeatureExtractor
from src.models.baseline_models import BaselineClassifier
from src.evaluation.evaluator import ModelEvaluator


def generate_augmented_corpus(
    n_per_class: int = 100,
    n_paraphrases: int = 2,
    noise_ratio: float = 0.15,
    n_ambiguous: int = 100,
    verbose: bool = True
) -> pd.DataFrame:
    """Génère un corpus augmenté avec diversité maximale."""

    print("=" * 60)
    print("PHASE 1: GÉNÉRATION DU CORPUS AUGMENTÉ")
    print("=" * 60)

    generator = JDMDataGenerator(min_weight=5, enrich_with_jdm=True)

    # 1. Génère les données de base
    if verbose:
        print(f"\n1. Génération de {n_per_class} exemples par classe...")
    base_data = generator.generate_all_classes(n_per_class=n_per_class, verbose=verbose)

    # 2. Ajoute des paraphrases
    if verbose:
        print(f"\n2. Génération de paraphrases ({n_paraphrases} par exemple)...")
    data_with_paraphrases = generator.generate_paraphrases(
        base_data, n_paraphrases=n_paraphrases, verbose=verbose
    )

    # 3. Ajoute du bruit
    if verbose:
        print(f"\n3. Ajout de bruit ({noise_ratio*100:.0f}%)...")
    data_with_noise = generator.add_noise_to_data(
        data_with_paraphrases, noise_ratio=noise_ratio, verbose=verbose
    )

    # 4. Génère des cas ambigus
    if verbose:
        print(f"\n4. Génération de {n_ambiguous} cas ambigus...")
    ambiguous_data = generator.generate_ambiguous_examples(
        n_samples=n_ambiguous, verbose=verbose
    )
    final_data = data_with_noise + ambiguous_data

    # Convertit en DataFrame
    df = pd.DataFrame(final_data)

    if verbose:
        print(f"\n✓ Total: {len(df)} exemples générés")
        print(f"\nDistribution des classes:")
        print(df['type_jdm'].value_counts().head(10))

    return df


def preprocess_generated_data(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Préprocesse les données générées."""

    print("\n" + "=" * 60)
    print("PHASE 2: PREPROCESSING")
    print("=" * 60)

    preprocessor = GenitivePreprocessor(use_jdm=False)
    preprocessed_data = []

    for idx, row in df.iterrows():
        try:
            construction = preprocessor.preprocess_construction(
                row['phrase'],
                row['type_jdm']
            )
            preprocessed_data.append({
                'phrase_originale': construction.phrase_originale,
                'nom1': construction.nom1,
                'nom2': construction.nom2,
                'determinant': construction.determinant,
                'nom1_lemme': construction.nom1_lemme,
                'nom2_lemme': construction.nom2_lemme,
                'definitude': construction.definitude,
                'type_jdm': construction.type_jdm,
                'est_valide': construction.est_valide,
            })
        except Exception:
            continue

    df_preprocessed = pd.DataFrame(preprocessed_data)
    df_valid = df_preprocessed[df_preprocessed['est_valide']].copy()

    if verbose:
        print(f"✓ Exemples valides: {len(df_valid)}/{len(df_preprocessed)}")

    return df_valid


def merge_with_original(df_augmented: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Fusionne avec le corpus original."""

    print("\n" + "=" * 60)
    print("PHASE 3: FUSION AVEC CORPUS ORIGINAL")
    print("=" * 60)

    original_path = Path('data/processed/corpus_preprocessed.csv')

    if original_path.exists():
        df_original = pd.read_csv(original_path)
        if verbose:
            print(f"Corpus original: {len(df_original)} exemples")

        # Fusionne
        df_merged = pd.concat([df_original, df_augmented], ignore_index=True)

        # Déduplique
        df_merged = df_merged.drop_duplicates(
            subset=['nom1_lemme', 'nom2_lemme', 'type_jdm'],
            keep='first'
        )

        if verbose:
            print(f"Corpus augmenté: {len(df_augmented)} exemples")
            print(f"✓ Corpus fusionné: {len(df_merged)} exemples (après déduplication)")

        return df_merged
    else:
        if verbose:
            print("⚠️  Corpus original non trouvé, utilisation des données augmentées seules")
        return df_augmented


def extract_features_and_split(
    df: pd.DataFrame,
    use_jdm: bool = True,
    verbose: bool = True
) -> tuple:
    """Extrait les features et split les données."""

    print("\n" + "=" * 60)
    print("PHASE 4: EXTRACTION DES FEATURES")
    print("=" * 60)

    extractor = EnhancedFeatureExtractor(use_jdm=use_jdm)
    features_df = extractor.extract_features(df, verbose=verbose)

    # Ajoute les labels
    features_df['type_jdm'] = df['type_jdm'].values

    if verbose:
        print(f"✓ {len(features_df.columns) - 1} features extraites")

    # Split stratifié
    train_df, temp_df = train_test_split(
        features_df, test_size=0.3, stratify=features_df['type_jdm'], random_state=42
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, stratify=temp_df['type_jdm'], random_state=42
    )

    if verbose:
        print(f"\n✓ Split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    return train_df, val_df, test_df


def train_models(train_df: pd.DataFrame, val_df: pd.DataFrame, verbose: bool = True) -> dict:
    """Entraîne tous les modèles."""

    print("\n" + "=" * 60)
    print("PHASE 5: ENTRAÎNEMENT DES MODÈLES")
    print("=" * 60)

    # Prépare les features
    excluded_cols = ['type_jdm', 'phrase_originale', 'nom1', 'nom2', 'determinant',
                     'nom1_lemme', 'nom2_lemme', 'definitude', 'est_valide', 'notes']
    feature_cols = [c for c in train_df.columns if c not in excluded_cols
                    and train_df[c].dtype in ['int64', 'float64', 'int32', 'float32', 'bool']]

    X_train = train_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    y_train = train_df['type_jdm']
    X_val = val_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    y_val = val_df['type_jdm']

    if verbose:
        print(f"Features: {len(feature_cols)}")
        print(f"Train: {len(X_train)}, Val: {len(X_val)}")

    models = {}
    results = {}

    for model_name in BaselineClassifier.MODELS.keys():
        if verbose:
            print(f"\n  Training {model_name}...")

        try:
            classifier = BaselineClassifier(model_name=model_name, use_regularization=True)
            classifier.train(X_train, y_train)

            # Évaluation
            y_pred = classifier.predict(X_val)
            accuracy = (y_pred == y_val).mean()

            models[model_name] = classifier
            results[model_name] = accuracy

            if verbose:
                print(f"    → Val accuracy: {accuracy*100:.1f}%")

            # Sauvegarde
            model_path = Path(f'models/baseline/{model_name}.joblib')
            model_path.parent.mkdir(parents=True, exist_ok=True)
            classifier.save(model_path)

        except Exception as e:
            print(f"    ✗ Erreur: {e}")

    return models, results


def test_on_external_corpus(models: dict, verbose: bool = True) -> pd.DataFrame:
    """Teste sur le corpus externe."""

    print("\n" + "=" * 60)
    print("PHASE 6: TEST SUR CORPUS EXTERNE")
    print("=" * 60)

    # Import du corpus externe
    from run_test_external_corpus import (
        EXTERNAL_TEST_CORPUS, extract_features_for_corpus, evaluate_on_corpus
    )

    df_external = extract_features_for_corpus(EXTERNAL_TEST_CORPUS, use_jdm=True, verbose=verbose)
    results = evaluate_on_corpus(models, df_external, verbose=verbose)

    return results


def main():
    parser = argparse.ArgumentParser(description='Pipeline augmentation et ré-entraînement')
    parser.add_argument('--n-per-class', type=int, default=100,
                        help='Exemples par classe (défaut: 100)')
    parser.add_argument('--n-paraphrases', type=int, default=2,
                        help='Paraphrases par exemple (défaut: 2)')
    parser.add_argument('--noise-ratio', type=float, default=0.15,
                        help='Ratio de bruit (défaut: 0.15)')
    parser.add_argument('--n-ambiguous', type=int, default=150,
                        help='Nombre de cas ambigus (défaut: 150)')
    parser.add_argument('--no-jdm-features', action='store_true',
                        help='Désactive les features JDM')
    parser.add_argument('--skip-generation', action='store_true',
                        help='Skip la génération (utilise corpus existant)')
    args = parser.parse_args()

    print("🚀 PIPELINE D'AUGMENTATION ET RÉ-ENTRAÎNEMENT")
    print("=" * 60)
    print(f"Paramètres:")
    print(f"  - Exemples/classe: {args.n_per_class}")
    print(f"  - Paraphrases: {args.n_paraphrases}")
    print(f"  - Ratio bruit: {args.noise_ratio}")
    print(f"  - Cas ambigus: {args.n_ambiguous}")

    # 1. Génération
    if not args.skip_generation:
        df_augmented = generate_augmented_corpus(
            n_per_class=args.n_per_class,
            n_paraphrases=args.n_paraphrases,
            noise_ratio=args.noise_ratio,
            n_ambiguous=args.n_ambiguous
        )

        # 2. Preprocessing
        df_preprocessed = preprocess_generated_data(df_augmented)

        # 3. Fusion
        df_final = merge_with_original(df_preprocessed)

        # Sauvegarde
        augmented_path = Path('data/processed/corpus_augmented_full.csv')
        df_final.to_csv(augmented_path, index=False)
        print(f"\n💾 Corpus sauvegardé: {augmented_path}")
    else:
        augmented_path = Path('data/processed/corpus_augmented_full.csv')
        if augmented_path.exists():
            df_final = pd.read_csv(augmented_path)
            print(f"✓ Corpus chargé: {len(df_final)} exemples")
        else:
            print("❌ Corpus augmenté non trouvé!")
            return

    # 4. Features et split
    train_df, val_df, test_df = extract_features_and_split(
        df_final, use_jdm=not args.no_jdm_features
    )

    # Sauvegarde les splits
    Path('data/processed/augmented').mkdir(parents=True, exist_ok=True)
    train_df.to_csv('data/processed/augmented/train.csv', index=False)
    val_df.to_csv('data/processed/augmented/val.csv', index=False)
    test_df.to_csv('data/processed/augmented/test.csv', index=False)

    # 5. Entraînement
    models, train_results = train_models(train_df, val_df)

    # 6. Test externe
    external_results = test_on_external_corpus(models)

    # Résumé final
    print("\n" + "=" * 60)
    print("📊 RÉSUMÉ FINAL")
    print("=" * 60)

    print("\nAccuracy sur validation (corpus augmenté):")
    for model, acc in sorted(train_results.items(), key=lambda x: -x[1]):
        print(f"  {model}: {acc*100:.1f}%")

    if not external_results.empty:
        print("\nAccuracy sur corpus externe:")
        for _, row in external_results.iterrows():
            print(f"  {row['model']}: {row['accuracy']*100:.1f}%")

        # Sauvegarde
        external_results.to_csv('results/augmented_external_test.csv', index=False)
        print("\n💾 Résultats sauvegardés: results/augmented_external_test.csv")

    print("\n✅ PIPELINE TERMINÉ!")


if __name__ == '__main__':
    main()
