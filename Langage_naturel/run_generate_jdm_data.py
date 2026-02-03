"""
Script de generation de donnees d'entrainement via JDM et pipeline complet.

Usage:
    python run_generate_jdm_data.py                    # Genere 100 exemples/classe
    python run_generate_jdm_data.py --n-per-class 200  # Genere 200 exemples/classe
    python run_generate_jdm_data.py --no-merge         # Ne fusionne pas avec le corpus existant
"""

import pandas as pd
import sys
import argparse
from pathlib import Path

sys.path.append('src')

from src.data.jdm_data_generator import JDMDataGenerator
from src.preprocessing.preprocessor import GenitivePreprocessor


def main():
    # Parsing des arguments
    parser = argparse.ArgumentParser(description='Generation de donnees via JDM')
    parser.add_argument('--n-per-class', type=int, default=100,
                        help='Nombre d\'exemples a generer par classe (defaut: 100)')
    parser.add_argument('--no-merge', action='store_true',
                        help='Ne pas fusionner avec le corpus existant')
    parser.add_argument('--min-weight', type=int, default=10,
                        help='Poids minimum des relations JDM (defaut: 10)')
    parser.add_argument('--output', type=str, default='data/generated/corpus_jdm.csv',
                        help='Fichier de sortie pour les donnees generees')
    args = parser.parse_args()

    print("=" * 60)
    print("GENERATION DE DONNEES D'ENTRAINEMENT VIA JDM")
    print("=" * 60)
    print(f"Exemples par classe: {args.n_per_class}")
    print(f"Poids minimum JDM: {args.min_weight}")
    print(f"Fusionner avec corpus existant: {not args.no_merge}")
    print()

    # Cree le dossier de sortie si necessaire
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # === PHASE 1: Generation des donnees ===
    print("PHASE 1: Generation des donnees depuis JDM")
    print("-" * 60)

    generator = JDMDataGenerator(min_weight=args.min_weight)
    generated_data = generator.generate_all_classes(
        n_per_class=args.n_per_class,
        verbose=True
    )

    # Convertit en DataFrame
    df_generated = pd.DataFrame(generated_data)

    # Sauvegarde les donnees generees
    df_generated.to_csv(args.output, index=False, encoding='utf-8')
    print(f"\nDonnees generees sauvegardees: {args.output}")

    # Affiche les statistiques
    print("\nStatistiques par classe:")
    print(df_generated['type_jdm'].value_counts().to_string())

    # === PHASE 2: Preprocessing ===
    print("\n" + "=" * 60)
    print("PHASE 2: Preprocessing des donnees generees")
    print("-" * 60)

    preprocessor = GenitivePreprocessor(use_jdm=False)  # Pas besoin de JDM ici

    # Prepare le DataFrame pour le preprocessing
    df_for_preprocessing = df_generated[['phrase', 'type_jdm']].copy()
    df_for_preprocessing.columns = ['phrase', 'type_jdm']

    # Preprocess chaque ligne
    preprocessed_data = []
    for idx, row in df_for_preprocessing.iterrows():
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
            'notes': construction.notes
        })

    df_preprocessed = pd.DataFrame(preprocessed_data)

    # Filtre les lignes valides
    df_valid = df_preprocessed[df_preprocessed['est_valide']].copy()
    print(f"Exemples valides apres preprocessing: {len(df_valid)}/{len(df_preprocessed)}")

    # Sauvegarde le corpus genere preprocesse
    generated_preprocessed_path = 'data/generated/corpus_jdm_preprocessed.csv'
    df_valid.to_csv(generated_preprocessed_path, index=False, encoding='utf-8')
    print(f"Corpus genere preprocesse: {generated_preprocessed_path}")

    # === PHASE 3: Fusion avec corpus existant (optionnel) ===
    if not args.no_merge:
        print("\n" + "=" * 60)
        print("PHASE 3: Fusion avec le corpus existant")
        print("-" * 60)

        # Charge le corpus existant
        existing_path = 'data/processed/corpus_preprocessed.csv'
        if Path(existing_path).exists():
            df_existing = pd.read_csv(existing_path)
            print(f"Corpus existant: {len(df_existing)} exemples")

            # Fusionne les deux corpus
            df_merged = pd.concat([df_existing, df_valid], ignore_index=True)

            # Supprime les doublons potentiels
            df_merged = df_merged.drop_duplicates(
                subset=['nom1_lemme', 'nom2_lemme', 'type_jdm'],
                keep='first'
            )

            print(f"Corpus fusionne: {len(df_merged)} exemples")

            # Statistiques par classe
            print("\nDistribution des classes (corpus fusionne):")
            print(df_merged['type_jdm'].value_counts().to_string())

            # Sauvegarde le corpus augmente
            augmented_path = 'data/processed/corpus_augmented.csv'
            df_merged.to_csv(augmented_path, index=False, encoding='utf-8')
            print(f"\nCorpus augmente sauvegarde: {augmented_path}")
        else:
            print(f"Corpus existant non trouve: {existing_path}")
            print("Sauvegarde uniquement des donnees generees.")
            augmented_path = 'data/processed/corpus_augmented.csv'
            df_valid.to_csv(augmented_path, index=False, encoding='utf-8')

    # === RESUME ===
    print("\n" + "=" * 60)
    print("RESUME")
    print("=" * 60)
    print(f"Exemples generes: {len(df_generated)}")
    print(f"Exemples valides: {len(df_valid)}")
    if not args.no_merge and Path('data/processed/corpus_augmented.csv').exists():
        df_final = pd.read_csv('data/processed/corpus_augmented.csv')
        print(f"Corpus final (augmente): {len(df_final)}")

    print("\nFichiers crees:")
    print(f"  - {args.output}")
    print(f"  - {generated_preprocessed_path}")
    if not args.no_merge:
        print(f"  - data/processed/corpus_augmented.csv")

    print("\n" + "=" * 60)
    print("PROCHAINES ETAPES")
    print("=" * 60)
    print("Pour utiliser le corpus augmente, executez:")
    print()
    print("  # 1. Extraction des features (avec le corpus augmente)")
    print("  python run_feature_extraction_augmented.py")
    print()
    print("  # 2. Entrainement des modeles")
    print("  python run_train_baseline.py")
    print()
    print("  # 3. Evaluation")
    print("  python run_evaluate_test.py")
    print()


if __name__ == '__main__':
    main()
