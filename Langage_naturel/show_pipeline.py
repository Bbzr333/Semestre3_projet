"""
Visualisation de la pipeline complete du projet.
Affiche chaque etape avec son statut (fichiers presents ou non).

Usage:
    python show_pipeline.py
"""

import os
from pathlib import Path


# Couleurs ANSI pour le terminal
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    RESET = '\033[0m'


def check_exists(path):
    """Verifie si un fichier ou dossier existe."""
    return Path(path).exists()


def status_icon(exists):
    return f"{Colors.GREEN}[OK]{Colors.RESET}" if exists else f"{Colors.RED}[--]{Colors.RESET}"


def print_step(number, title, description, inputs, outputs, script):
    """Affiche une etape de la pipeline."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.RESET}")
    print(f"{Colors.BOLD}  ETAPE {number}: {title}{Colors.RESET}")
    print(f"{Colors.DIM}  {description}{Colors.RESET}")
    print(f"{Colors.BLUE}{'='*60}{Colors.RESET}")

    print(f"  Script : {Colors.YELLOW}{script}{Colors.RESET}")

    if inputs:
        print(f"  Entrees :")
        for inp in inputs:
            icon = status_icon(check_exists(inp))
            print(f"    {icon} {inp}")

    if outputs:
        print(f"  Sorties :")
        for out in outputs:
            icon = status_icon(check_exists(out))
            print(f"    {icon} {out}")


def main():
    os.chdir(Path(__file__).parent)

    print(f"\n{Colors.BOLD}{'#'*60}")
    print(f"  PIPELINE COMPLETE - Classification Relations Genitives")
    print(f"{'#'*60}{Colors.RESET}")
    print(f"{Colors.DIM}  Projet: Classification semantique des constructions 'A de B'{Colors.RESET}")

    # Etape 1
    print_step(
        1, "PREPROCESSING",
        "Extraction des constructions genitives, lemmatisation, normalisation",
        inputs=["data/raw/corpus_initial/corpus_A_de_B_relations_150.csv"],
        outputs=["data/processed/corpus_preprocessed.csv"],
        script="python run_preprocessing.py"
    )

    # Etape 2
    print_step(
        2, "EXTRACTION DE FEATURES + SPLIT",
        "Features morpho-syntaxiques + JDM, puis decoupage train/val/test",
        inputs=["data/processed/corpus_preprocessed.csv"],
        outputs=[
            "data/processed/corpus_with_features.csv",
            "data/processed/train.csv",
            "data/processed/val.csv",
            "data/processed/test.csv",
        ],
        script="python run_feature_extraction.py [--no-jdm]"
    )

    # Etape 3
    print_step(
        3, "ENTRAINEMENT BASELINE",
        "Random Forest, SVM, Logistic Regression, Gradient Boosting",
        inputs=[
            "data/processed/train.csv",
            "data/processed/val.csv",
        ],
        outputs=[
            "models/baseline/random_forest.joblib",
            "models/baseline/svm_linear.joblib",
            "models/baseline/logistic_regression.joblib",
            "models/baseline/gradient_boosting.joblib",
            "results/baseline_comparison.csv",
        ],
        script="python run_train_baseline.py"
    )

    # Etape 4
    print_step(
        4, "EVALUATION SUR TEST SET",
        "Metriques detaillees, matrices de confusion, analyse d'erreurs",
        inputs=[
            "data/processed/test.csv",
            "models/baseline/",
        ],
        outputs=[
            "results/test_results.csv",
            "results/plots/",
        ],
        script="python run_evaluate_test.py"
    )

    # Etape 5
    print_step(
        5, "VALIDATION CROISEE 10-FOLD",
        "Verification de la robustesse et detection d'overfitting",
        inputs=[
            "data/processed/train.csv",
            "data/processed/val.csv",
            "data/processed/test.csv",
        ],
        outputs=[
            "results/cross_validation_detailed.csv",
            "results/plots/cross_validation_10fold.png",
        ],
        script="python run_cross_validation.py"
    )

    # Etape 6 (optionnelle)
    print_step(
        6, "TEST CORPUS EXTERNE (optionnel)",
        "Evaluation de la generalisation sur des phrases nouvelles",
        inputs=["models/baseline/"],
        outputs=["results/external_test_results.csv"],
        script="python run_test_external_corpus.py"
    )

    # Etape 7 (optionnelle)
    print_step(
        7, "AUGMENTATION + RE-ENTRAINEMENT (optionnel)",
        "Generation de donnees via JDM, paraphrases, bruit, puis re-entrainement",
        inputs=["data/processed/corpus_preprocessed.csv"],
        outputs=[
            "data/processed/corpus_augmented_full.csv",
            "data/processed/augmented/train.csv",
        ],
        script="python run_augment_and_retrain.py"
    )

    # Etape 8 (optionnelle)
    print_step(
        8, "COMPARAISON CHATGPT (optionnel)",
        "Evaluation de GPT-3.5/4 en few-shot vs modeles baseline",
        inputs=[
            "data/processed/train.csv",
            "data/processed/test.csv",
        ],
        outputs=["results/chatgpt_comparison.csv"],
        script="python run_chatgpt_simple.py"
    )

    # Etape 9
    print_step(
        9, "DEMO INTERACTIVE",
        "Interface Gradio avec CamemBERT + meilleur baseline + Knowledge Graph",
        inputs=[
            "models/camembert/best_model/",
            "models/baseline/",
        ],
        outputs=[],
        script="python demo.py  (puis ouvrir http://localhost:7860)"
    )

    # Resume
    print(f"\n{Colors.BOLD}{'='*60}")
    print(f"  RESUME")
    print(f"{'='*60}{Colors.RESET}")

    steps_order = [
        ("run_preprocessing.py", "Preprocessing"),
        ("run_feature_extraction.py", "Feature extraction + split"),
        ("run_train_baseline.py", "Entrainement baseline"),
        ("run_evaluate_test.py", "Evaluation test set"),
        ("run_cross_validation.py", "Validation croisee"),
    ]

    print(f"\n  Pour lancer la pipeline de base, executer dans l'ordre :\n")
    for i, (script, desc) in enumerate(steps_order, 1):
        print(f"    {i}. python {script:<35s} # {desc}")

    print(f"\n  Scripts optionnels :")
    print(f"    - python run_test_external_corpus.py      # Test generalisation")
    print(f"    - python run_augment_and_retrain.py        # Augmentation donnees")
    print(f"    - python run_chatgpt_simple.py             # Comparaison ChatGPT")
    print(f"    - python demo.py                           # Interface Gradio")

    print()


if __name__ == '__main__':
    main()
