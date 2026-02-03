#!/usr/bin/env python3
"""
Script d'entrainement du classifieur CamemBERT.

Usage:
    python run_train_camembert.py
    python run_train_camembert.py --epochs 10 --batch_size 32
    python run_train_camembert.py --lr 3e-5 --max_length 128
"""

import argparse
import pandas as pd
from pathlib import Path
import sys

sys.path.append('src')

from src.models.deep_learning.camembert_classifier import CamemBERTClassifier


def main():
    parser = argparse.ArgumentParser(description='Entrainer CamemBERT')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Nombre d\'epochs (default: 5)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Taille des batchs (default: 16)')
    parser.add_argument('--lr', type=float, default=2e-5,
                        help='Learning rate (default: 2e-5)')
    parser.add_argument('--max_length', type=int, default=64,
                        help='Longueur max des sequences (default: 64)')
    parser.add_argument('--data_dir', type=str, default='data/processed',
                        help='Dossier des donnees (default: data/processed)')
    parser.add_argument('--output_dir', type=str, default='models/camembert',
                        help='Dossier de sortie (default: models/camembert)')
    args = parser.parse_args()

    print("=" * 60)
    print("ENTRAINEMENT CAMEMBERT")
    print("=" * 60)
    print(f"\nParametres:")
    print(f"  - Epochs: {args.epochs}")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Learning rate: {args.lr}")
    print(f"  - Max length: {args.max_length}")
    print(f"  - Data dir: {args.data_dir}")
    print(f"  - Output dir: {args.output_dir}")

    # Charger les donnees
    print("\nChargement des donnees...")
    data_dir = Path(args.data_dir)

    train_df = pd.read_csv(data_dir / 'train.csv')
    val_df = pd.read_csv(data_dir / 'val.csv')

    # Extraire textes et labels
    X_train = train_df['phrase_originale'].tolist()
    y_train = train_df['type_jdm'].tolist()
    X_val = val_df['phrase_originale'].tolist()
    y_val = val_df['type_jdm'].tolist()

    print(f"  - Train: {len(X_train)} exemples")
    print(f"  - Val: {len(X_val)} exemples")
    print(f"  - Classes: {len(set(y_train))}")

    # Afficher quelques exemples
    print("\nExemples de donnees:")
    for i in range(min(3, len(X_train))):
        print(f"  [{y_train[i]}] {X_train[i]}")

    # Creer le dossier de sortie
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialiser et entrainer le modele
    print("\n" + "=" * 60)
    classifier = CamemBERTClassifier(
        num_labels=15,
        max_length=args.max_length,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        num_epochs=args.epochs
    )

    classifier.train(X_train, y_train, X_val, y_val)

    # Sauvegarder le modele
    model_path = output_dir / 'best_model'
    classifier.save(model_path)

    # Afficher le resume
    print("\n" + "=" * 60)
    print("RESUME DE L'ENTRAINEMENT")
    print("=" * 60)
    history = classifier.training_history
    print(f"\nTemps total: {history['training_time_seconds']:.1f}s")
    print(f"Meilleure accuracy validation: {history['best_val_acc']:.4f}")

    print("\nHistorique par epoch:")
    print("-" * 50)
    print(f"{'Epoch':<8} {'Train Loss':<12} {'Train Acc':<12} {'Val Acc':<12}")
    print("-" * 50)
    for i, epoch in enumerate(history['epochs']):
        print(f"{epoch:<8} {history['train_loss'][i]:<12.4f} "
              f"{history['train_acc'][i]:<12.4f} {history['val_acc'][i]:<12.4f}")

    print("\n" + "=" * 60)
    print(f"Modele sauvegarde: {model_path}")
    print("=" * 60)

    # Sauvegarder l'historique en CSV
    history_df = pd.DataFrame({
        'epoch': history['epochs'],
        'train_loss': history['train_loss'],
        'train_acc': history['train_acc'],
        'val_loss': history['val_loss'],
        'val_acc': history['val_acc']
    })
    history_path = output_dir / 'training_history.csv'
    history_df.to_csv(history_path, index=False)
    print(f"Historique sauvegarde: {history_path}")


if __name__ == '__main__':
    main()
