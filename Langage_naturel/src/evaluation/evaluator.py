"""
Système d'évaluation des modèles
"""

from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                            classification_report, confusion_matrix)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class ModelEvaluator:
    """
    Évalue les performances d'un modèle
    """
    
    def evaluate(self, model, X, y_true):
        """
        Évaluation complète d'un modèle
        """
        # Prédictions
        y_pred = model.predict(X)
        
        # Métriques globales
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro', zero_division=0
        )
        
        # Métriques par classe
        precision_per_class, recall_per_class, f1_per_class, support = \
            precision_recall_fscore_support(y_true, y_pred, zero_division=0)
        
        # Matrice de confusion
        cm = confusion_matrix(y_true, y_pred)
        
        return {
            'accuracy': accuracy,
            'precision_macro': precision,
            'recall_macro': recall,
            'f1_macro': f1,
            'precision_per_class': dict(zip(model.label_encoder.classes_, precision_per_class)),
            'recall_per_class': dict(zip(model.label_encoder.classes_, recall_per_class)),
            'f1_per_class': dict(zip(model.label_encoder.classes_, f1_per_class)),
            'confusion_matrix': cm,
            'classification_report': classification_report(y_true, y_pred)
        }
    
    def plot_confusion_matrix(self, cm, labels, save_path=None):
        """
        Visualise la matrice de confusion
        """
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=labels, yticklabels=labels)
        plt.title('Matrice de Confusion')
        plt.ylabel('Vraie classe')
        plt.xlabel('Classe prédite')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()