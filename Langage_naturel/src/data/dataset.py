"""
Dataset PyTorch pour la classification des relations semantiques francaises.
"""

import torch
from torch.utils.data import Dataset


class FrenchRelationDataset(Dataset):
    """
    Dataset PyTorch pour la classification des relations semantiques.

    Attributes:
        texts: Liste de phrases (phrase_originale)
        labels: Liste de labels encodes (entiers)
        tokenizer: Tokenizer HuggingFace (CamemBERT)
        max_length: Longueur maximale des sequences
    """

    def __init__(self, texts, labels, tokenizer, max_length=64):
        """
        Initialise le dataset.

        Args:
            texts: Liste de chaines de caracteres (phrases)
            labels: Liste de labels encodes (entiers) ou None pour inference
            tokenizer: Tokenizer HuggingFace
            max_length: Longueur maximale de tokenization
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])

        # Tokenization
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        item = {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
        }

        # Ajouter le label si disponible (pas en mode inference)
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)

        return item


class FrenchRelationDatasetInference(Dataset):
    """
    Dataset simplifie pour l'inference (sans labels).
    """

    def __init__(self, texts, tokenizer, max_length=64):
        """
        Args:
            texts: Liste de phrases
            tokenizer: Tokenizer HuggingFace
            max_length: Longueur maximale
        """
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
        }
