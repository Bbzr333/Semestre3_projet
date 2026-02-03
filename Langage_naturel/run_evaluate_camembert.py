#!/usr/bin/env python3
"""
Script d'evaluation du classifieur CamemBERT sur le test set.

Usage:
    python run_evaluate_camembert.py
    python run_evaluate_camembert.py --model_path models/camembert/best_model
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append('src')

from src.models.deep_learning.camembert_classifier import CamemBERTClassifier


def plot_confusion_matrix(y_true, y_pred, labels, output_path):
    """Genere et sauvegarde la matrice de confusion."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    plt.figure(figsize=(14, 12))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels
    )
    plt.title('Matrice de Confusion - CamemBERT', fontsize=14)
    plt.xlabel('Prediction', fontsize=12)
    plt.ylabel('Verite', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Matrice de confusion sauvegardee: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluer CamemBERT')
    parser.add_argument('--model_path', type=str, default='models/camembert/best_model',
                        help='Chemin du modele (default: models/camembert/best_model)')
    parser.add_argument('--data_dir', type=str, default='data/processed',
                        help='Dossier des donnees (default: data/processed)')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Dossier de sortie (default: results)')
    args = parser.parse_args()

    print("=" * 60)
    print("EVALUATION CAMEMBERT")
    print("=" * 60)

    # Charger le modele
    print(f"\nChargement du modele: {args.model_path}")
    model_path = Path(args.model_path)

    if not model_path.exists():
        print(f"ERREUR: Modele non trouve: {model_path}")
        print("Lancez d'abord: python run_train_camembert.py")
        sys.exit(1)

    classifier = CamemBERTClassifier.load(model_path)

    # Charger les donnees de test
    print("\nChargement des donnees de test...")
    data_dir = Path(args.data_dir)
    test_df = pd.read_csv(data_dir / 'test.csv')

    X_test = test_df['phrase_originale'].tolist()
    y_test = test_df['type_jdm'].tolist()

    print(f"  - Test: {len(X_test)} exemples")
    print(f"  - Classes: {len(set(y_test))}")

    # Predictions
    print("\nPrediction en cours...")
    y_pred = classifier.predict(X_test)

    # Calcul des metriques
    print("\n" + "=" * 60)
    print("RESULTATS")
    print("=" * 60)

    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average='macro', zero_division=0
    )

    print(f"\nMetriques globales:")
    print(f"  - Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  - Precision (macro): {precision:.4f}")
    print(f"  - Recall (macro): {recall:.4f}")
    print(f"  - F1-Score (macro): {f1:.4f}")

    # Nombre d'erreurs
    errors = sum(1 for yt, yp in zip(y_test, y_pred) if yt != yp)
    print(f"\n  - Erreurs: {errors}/{len(y_test)}")

    # Rapport de classification detaille
    print("\n" + "-" * 60)
    print("RAPPORT PAR CLASSE")
    print("-" * 60)
    labels = sorted(list(set(y_test)))
    report = classification_report(y_test, y_pred, labels=labels, zero_division=0)
    print(report)

    # Creer le dossier de sortie
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)

    # Matrice de confusion
    plot_confusion_matrix(
        y_test, y_pred, labels,
        plots_dir / 'confusion_matrix_camembert.png'
    )

    # Sauvegarder les resultats
    results = {
        'model': ['CamemBERT'],
        'accuracy': [accuracy],
        'precision_macro': [precision],
        'recall_macro': [recall],
        'f1_macro': [f1],
        'errors': [errors],
        'total': [len(y_test)]
    }
    results_df = pd.DataFrame(results)
    results_path = output_dir / 'camembert_results.csv'
    results_df.to_csv(results_path, index=False)
    print(f"\nResultats sauvegardes: {results_path}")

    # Sauvegarder les erreurs
    errors_data = []
    for i, (yt, yp) in enumerate(zip(y_test, y_pred)):
        if yt != yp:
            errors_data.append({
                'phrase': X_test[i],
                'true_label': yt,
                'predicted_label': yp
            })

    if errors_data:
        errors_df = pd.DataFrame(errors_data)
        errors_path = output_dir / 'errors_camembert.csv'
        errors_df.to_csv(errors_path, index=False)
        print(f"Erreurs sauvegardees: {errors_path}")

        # Afficher quelques erreurs
        print("\n" + "-" * 60)
        print("EXEMPLES D'ERREURS (5 premiers)")
        print("-" * 60)
        for i, row in errors_df.head(5).iterrows():
            print(f"\n  Phrase: {row['phrase']}")
            print(f"  Verite: {row['true_label']}")
            print(f"  Prediction: {row['predicted_label']}")

    # Comparaison avec baseline (si disponible)
    baseline_path = output_dir / 'test_results.csv'
    if baseline_path.exists():
        print("\n" + "=" * 60)
        print("COMPARAISON AVEC BASELINE")
        print("=" * 60)
        baseline_df = pd.read_csv(baseline_path)
        if 'accuracy' in baseline_df.columns:
            best_baseline = baseline_df.loc[baseline_df['accuracy'].idxmax()]
            print(f"\nMeilleur modele baseline: {best_baseline.get('model', 'N/A')}")
            print(f"  - Accuracy baseline: {best_baseline['accuracy']:.4f}")
            print(f"  - Accuracy CamemBERT: {accuracy:.4f}")

            diff = accuracy - best_baseline['accuracy']
            if diff > 0:
                print(f"\n  CamemBERT est meilleur de +{diff:.4f}")
            elif diff < 0:
                print(f"\n  Baseline est meilleur de +{-diff:.4f}")
            else:
                print(f"\n  Performance identique")

    print("\n" + "=" * 60)
    print("EVALUATION TERMINEE")
    print("=" * 60)


if __name__ == '__main__':
    main()
