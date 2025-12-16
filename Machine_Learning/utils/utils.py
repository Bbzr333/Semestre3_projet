"""
Fonctions utilitaires pour le projet CLIP HAI923

Nom: [À COMPLÉTER]
Prénom: [À COMPLÉTER]
N° Carte Étudiant: [À COMPLÉTER]
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import random


def set_seed(seed: int = 42):
    """Fixe les graines aléatoires pour la reproductibilité"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model: nn.Module) -> int:
    """Compte le nombre de paramètres entraînables dans un modèle"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_model(model: nn.Module, path: str, optimizer=None, epoch: int = None):
    """Sauvegarde un modèle PyTorch"""
    checkpoint = {
        'model_state_dict': model.state_dict(),
    }
    if optimizer:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    if epoch:
        checkpoint['epoch'] = epoch
    
    torch.save(checkpoint, path)
    print(f"✅ Modèle sauvegardé: {path}")


def load_model(model: nn.Module, path: str, optimizer=None) -> Tuple[nn.Module, int]:
    """Charge un modèle PyTorch"""
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    print(f"✅ Modèle chargé: {path} (epoch {epoch})")
    
    return model, epoch


def plot_training_history(history: Dict, save_path: str = None):
    """Affiche les courbes d'entraînement"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss
    ax1.plot(history['train_loss'], label='Train')
    ax1.plot(history['val_loss'], label='Validation')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss pendant l\'entraînement')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy (si disponible)
    if 'train_acc' in history:
        ax2.plot(history['train_acc'], label='Train')
        ax2.plot(history['val_acc'], label='Validation')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Accuracy pendant l\'entraînement')
        ax2.legend()
        ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Graphique sauvegardé: {save_path}")
    
    plt.show()


def normalize_embeddings(embeddings: torch.Tensor) -> torch.Tensor:
    """
    Normalise les embeddings (CRITIQUE pour CLIP)
    
    Args:
        embeddings: Tensor de shape (batch_size, embedding_dim)
    
    Returns:
        Embeddings normalisés
    """
    return embeddings / embeddings.norm(dim=-1, keepdim=True)


def compute_similarity_matrix(image_embeddings: torch.Tensor, 
                              text_embeddings: torch.Tensor) -> torch.Tensor:
    """
    Calcule la matrice de similarité entre embeddings image et texte
    
    Args:
        image_embeddings: (batch_size, embedding_dim)
        text_embeddings: (batch_size, embedding_dim)
    
    Returns:
        Matrice de similarité (batch_size, batch_size)
    """
    # Normaliser les embeddings
    image_embeddings = normalize_embeddings(image_embeddings)
    text_embeddings = normalize_embeddings(text_embeddings)
    
    # Produit scalaire = similarité cosinus après normalisation
    similarity = torch.matmul(image_embeddings, text_embeddings.T)
    
    return similarity


class ContrastiveLoss(nn.Module):
    """
    Loss contrastive pour CLIP
    
    Cette loss rapproche les couples image-texte qui vont ensemble
    et éloigne ceux qui ne correspondent pas.
    """
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, image_embeddings: torch.Tensor, 
                text_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image_embeddings: (batch_size, embedding_dim)
            text_embeddings: (batch_size, embedding_dim)
        
        Returns:
            Loss contrastive
        """
        # Normaliser les embeddings
        image_embeddings = normalize_embeddings(image_embeddings)
        text_embeddings = normalize_embeddings(text_embeddings)
        
        # Calculer la matrice de similarité
        logits = torch.matmul(image_embeddings, text_embeddings.T) / self.temperature
        
        # Labels: la diagonale (couples corrects)
        batch_size = image_embeddings.shape[0]
        labels = torch.arange(batch_size).to(image_embeddings.device)
        
        # Loss dans les deux directions (image→texte et texte→image)
        loss_i2t = self.criterion(logits, labels)
        loss_t2i = self.criterion(logits.T, labels)
        
        loss = (loss_i2t + loss_t2i) / 2
        
        return loss


def display_top_k_results(query, results: List[Tuple], 
                          query_type: str = "text", k: int = 5):
    """
    Affiche les top-k résultats d'une requête CLIP
    
    Args:
        query: Texte ou image de la requête
        results: Liste de tuples (item, score)
        query_type: "text" ou "image"
        k: Nombre de résultats à afficher
    """
    print(f"\n{'='*60}")
    print(f"Requête ({query_type}): {query if isinstance(query, str) else '[Image]'}")
    print(f"{'='*60}\n")
    
    for i, (item, score) in enumerate(results[:k], 1):
        print(f"{i}. Score: {score:.4f}")
        if isinstance(item, str):
            print(f"   Texte: {item}")
        print()
