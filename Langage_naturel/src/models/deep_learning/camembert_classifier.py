"""
Classifieur CamemBERT pour la classification des relations semantiques francaises.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import (
    CamembertTokenizer,
    CamembertForSequenceClassification,
    get_linear_schedule_with_warmup
)
from sklearn.preprocessing import LabelEncoder
import numpy as np
from pathlib import Path
import joblib
from tqdm import tqdm
import time

import sys
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from src.data.dataset import FrenchRelationDataset


class CamemBERTClassifier:
    """
    Classifieur base sur CamemBERT pour les relations semantiques francaises.
    Interface compatible avec BaselineClassifier.
    """

    def __init__(
        self,
        model_name='camembert-base',
        num_labels=15,
        max_length=64,
        learning_rate=2e-5,
        batch_size=16,
        num_epochs=5,
        warmup_ratio=0.1,
        device=None
    ):
        """
        Initialise le classifieur CamemBERT.

        Args:
            model_name: Identifiant du modele HuggingFace
            num_labels: Nombre de classes (15 relations)
            max_length: Longueur maximale des sequences
            learning_rate: Taux d'apprentissage
            batch_size: Taille des batchs
            num_epochs: Nombre d'epochs
            warmup_ratio: Ratio de warmup pour le scheduler
            device: 'cuda', 'cpu', ou None pour auto-detection
        """
        self.model_name = model_name
        self.num_labels = num_labels
        self.max_length = max_length
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.warmup_ratio = warmup_ratio

        # Device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Composants du modele (initialises au chargement ou a l'entrainement)
        self.tokenizer = None
        self.model = None
        self.label_encoder = LabelEncoder()

        # Historique d'entrainement
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'epochs': [],
            'best_val_acc': 0.0,
            'training_time_seconds': 0.0
        }

    def _init_model(self):
        """Initialise le tokenizer et le modele."""
        print(f"Chargement de {self.model_name}...")
        self.tokenizer = CamembertTokenizer.from_pretrained(self.model_name)
        self.model = CamembertForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels
        )
        self.model.to(self.device)
        print(f"Modele charge sur {self.device}")

    def train(self, texts, labels, val_texts=None, val_labels=None):
        """
        Entraine le modele.

        Args:
            texts: Liste de phrases (phrase_originale)
            labels: Liste de labels (type_jdm)
            val_texts: Phrases de validation (optionnel)
            val_labels: Labels de validation (optionnel)

        Returns:
            self (pour le chainage)
        """
        start_time = time.time()

        # Initialiser le modele
        self._init_model()

        # Encoder les labels
        self.label_encoder.fit(labels)
        y_train = self.label_encoder.transform(labels)

        if val_labels is not None:
            y_val = self.label_encoder.transform(val_labels)
        else:
            y_val = None

        # Creer les datasets
        train_dataset = FrenchRelationDataset(
            texts, y_train, self.tokenizer, self.max_length
        )
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )

        if val_texts is not None and y_val is not None:
            val_dataset = FrenchRelationDataset(
                val_texts, y_val, self.tokenizer, self.max_length
            )
            val_loader = DataLoader(
                val_dataset, batch_size=self.batch_size, shuffle=False
            )
        else:
            val_loader = None

        # Optimizer et scheduler
        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)
        total_steps = len(train_loader) * self.num_epochs
        warmup_steps = int(total_steps * self.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        # Entrainement
        print(f"\nEntrainement sur {len(texts)} exemples...")
        print(f"Epochs: {self.num_epochs}, Batch size: {self.batch_size}")
        print(f"Learning rate: {self.learning_rate}, Warmup steps: {warmup_steps}")
        print("-" * 60)

        best_val_acc = 0.0

        for epoch in range(self.num_epochs):
            # Phase d'entrainement
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}")
            for batch in pbar:
                optimizer.zero_grad()

                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels_batch = batch['labels'].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels_batch
                )

                loss = outputs.loss
                loss.backward()
                optimizer.step()
                scheduler.step()

                train_loss += loss.item()
                predictions = torch.argmax(outputs.logits, dim=-1)
                train_correct += (predictions == labels_batch).sum().item()
                train_total += labels_batch.size(0)

                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

            train_loss /= len(train_loader)
            train_acc = train_correct / train_total

            # Phase de validation
            if val_loader is not None:
                val_loss, val_acc = self._evaluate(val_loader)
            else:
                val_loss, val_acc = 0.0, 0.0

            # Sauvegarder l'historique
            self.training_history['epochs'].append(epoch + 1)
            self.training_history['train_loss'].append(train_loss)
            self.training_history['train_acc'].append(train_acc)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['val_acc'].append(val_acc)

            # Affichage
            print(f"\nEpoch {epoch+1}/{self.num_epochs}:")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            if val_loader is not None:
                print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

            # Meilleur modele
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.training_history['best_val_acc'] = best_val_acc

        self.training_history['training_time_seconds'] = time.time() - start_time
        print(f"\nEntrainement termine en {self.training_history['training_time_seconds']:.1f}s")
        print(f"Meilleure accuracy validation: {best_val_acc:.4f}")

        return self

    def _evaluate(self, dataloader):
        """Evalue le modele sur un dataloader."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels_batch = batch['labels'].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels_batch
                )

                total_loss += outputs.loss.item()
                predictions = torch.argmax(outputs.logits, dim=-1)
                correct += (predictions == labels_batch).sum().item()
                total += labels_batch.size(0)

        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total
        return avg_loss, accuracy

    def predict(self, texts):
        """
        Predit les labels pour une liste de textes.

        Args:
            texts: Liste de phrases ou phrase unique

        Returns:
            Liste de labels predits (strings)
        """
        if isinstance(texts, str):
            texts = [texts]

        self.model.eval()
        predictions = []

        # Creer un dataset sans labels
        dataset = FrenchRelationDataset(
            texts, [0] * len(texts), self.tokenizer, self.max_length
        )
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                batch_preds = torch.argmax(outputs.logits, dim=-1)
                predictions.extend(batch_preds.cpu().numpy())

        # Decoder les labels
        return self.label_encoder.inverse_transform(predictions)

    def predict_proba(self, texts):
        """
        Retourne les probabilites pour chaque classe.

        Args:
            texts: Liste de phrases

        Returns:
            numpy array de shape (n_samples, n_classes)
        """
        if isinstance(texts, str):
            texts = [texts]

        self.model.eval()
        all_probs = []

        dataset = FrenchRelationDataset(
            texts, [0] * len(texts), self.tokenizer, self.max_length
        )
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                probs = torch.softmax(outputs.logits, dim=-1)
                all_probs.append(probs.cpu().numpy())

        return np.vstack(all_probs)

    def save(self, filepath):
        """
        Sauvegarde le modele complet.

        Args:
            filepath: Chemin du dossier de sauvegarde
        """
        filepath = Path(filepath)
        filepath.mkdir(parents=True, exist_ok=True)

        # Sauvegarder le modele et tokenizer (format HuggingFace)
        self.model.save_pretrained(filepath)
        self.tokenizer.save_pretrained(filepath)

        # Sauvegarder les metadonnees
        metadata = {
            'label_encoder': self.label_encoder,
            'training_history': self.training_history,
            'config': {
                'model_name': self.model_name,
                'num_labels': self.num_labels,
                'max_length': self.max_length,
                'learning_rate': self.learning_rate,
                'batch_size': self.batch_size,
                'num_epochs': self.num_epochs
            }
        }
        joblib.dump(metadata, filepath / 'metadata.joblib')

        print(f"Modele sauvegarde dans {filepath}")

    @classmethod
    def load(cls, filepath):
        """
        Charge un modele sauvegarde.

        Args:
            filepath: Chemin du dossier du modele

        Returns:
            Instance de CamemBERTClassifier
        """
        filepath = Path(filepath)

        # Charger les metadonnees
        metadata = joblib.load(filepath / 'metadata.joblib')
        config = metadata['config']

        # Creer l'instance
        instance = cls(
            model_name=config['model_name'],
            num_labels=config['num_labels'],
            max_length=config['max_length'],
            learning_rate=config['learning_rate'],
            batch_size=config['batch_size'],
            num_epochs=config['num_epochs']
        )

        # Charger le modele et tokenizer
        instance.tokenizer = CamembertTokenizer.from_pretrained(filepath)
        instance.model = CamembertForSequenceClassification.from_pretrained(filepath)
        instance.model.to(instance.device)

        # Restaurer les metadonnees
        instance.label_encoder = metadata['label_encoder']
        instance.training_history = metadata['training_history']

        print(f"Modele charge depuis {filepath}")
        print(f"Device: {instance.device}")

        return instance
