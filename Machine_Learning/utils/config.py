"""
Configuration du projet CLIP HAI923

Nom: [À COMPLÉTER]
Prénom: [À COMPLÉTER]
N° Carte Étudiant: [À COMPLÉTER]
"""

import torch

# ==================== PARAMÈTRES GÉNÉRAUX ====================
RANDOM_SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==================== DONNÉES ====================
DATA_DIR = "./data"
NUM_CLASSES = 4
CLASSES = ["bike", "ball", "water", "dog"]
IMAGES_PER_CLASS = 150
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1

# ==================== ÉTAPE 1: CNN IMAGES ====================
CNN_CONFIG = {
    "img_size": 224,
    "batch_size": 32,
    "num_epochs": 10,
    "learning_rate": 1e-3,
    "architecture": "simple_cnn"  # ou "resnet18", "vgg16", etc.
}

# ==================== ÉTAPE 2: SMALLBERT TEXTE ====================
SMALLBERT_CONFIG = {
    "model_name": "google/bert_uncased_L-4_H-512_A-8",  # SmallBERT
    "max_length": 128,
    "batch_size": 32,
    "num_epochs": 10,
    "learning_rate": 2e-5,
}

# ATTENTION: SmallBERT n'a PAS de token <CLS>
# Il faut trouver une autre méthode pour résumer la phrase

# ==================== ÉTAPE 3: MODÈLE CLIP ====================
CLIP_CONFIG = {
    "embedding_dim": 512,  # DOIT ÊTRE IDENTIQUE pour image et texte
    "temperature": 0.07,   # Température pour la loss contrastive
    "batch_size": 32,
    "num_epochs": 20,
    "learning_rate": 1e-4,
    "normalize_embeddings": True,  # CRITIQUE: normaliser les sorties
}

# ==================== INFÉRENCE ====================
INFERENCE_CONFIG = {
    "top_k": 5,  # Nombre de résultats à retourner
    "show_scores": True
}

# ==================== CHEMINS DE SAUVEGARDE ====================
PATHS = {
    "cnn_model": "./models/cnn_classifier.pth",
    "smallbert_model": "./models/smallbert_classifier.pth",
    "clip_model": "./models/clip_model.pth",
    "results": "./results"
}
