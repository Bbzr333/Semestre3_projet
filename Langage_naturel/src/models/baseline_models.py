"""
Modèles de classification baseline
"""

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
import joblib
import json
from datetime import datetime

class BaselineClassifier:
    """
    Gestionnaire de modèles baseline
    """
    
    MODELS = {
        'logistic_regression': LogisticRegression(max_iter=1000, random_state=42),
        'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'svm_linear': SVC(kernel='linear', random_state=42),
        'svm_rbf': SVC(kernel='rbf', random_state=42),
        'gradient_boosting': GradientBoostingClassifier(random_state=42),
    }
    
    def __init__(self, model_name='random_forest'):
        self.model_name = model_name
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        
        # Pipeline
        self.pipeline = Pipeline([
            ('scaler', self.scaler),
            ('classifier', self.MODELS[model_name])
        ])
        
        self.training_history = {}
    
    def train(self, X_train, y_train):
        """Entraîne le modèle"""
        start_time = datetime.now()
        
        # Encode labels
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        
        # Train
        self.pipeline.fit(X_train, y_train_encoded)
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        self.training_history = {
            'model_name': self.model_name,
            'training_time_seconds': training_time,
            'n_samples': len(X_train),
            'n_features': X_train.shape[1],
            'n_classes': len(self.label_encoder.classes_)
        }
        
        return self
    
    def predict(self, X):
        """Prédictions"""
        y_pred_encoded = self.pipeline.predict(X)
        return self.label_encoder.inverse_transform(y_pred_encoded)
    
    def predict_proba(self, X):
        """Probabilités de prédiction"""
        if hasattr(self.pipeline.named_steps['classifier'], 'predict_proba'):
            return self.pipeline.predict_proba(X)
        else:
            raise NotImplementedError(f"{self.model_name} ne supporte pas predict_proba")
    
    def save(self, filepath):
        """Sauvegarde le modèle"""
        model_data = {
            'pipeline': self.pipeline,
            'label_encoder': self.label_encoder,
            'training_history': self.training_history
        }
        joblib.dump(model_data, filepath)
    
    @classmethod
    def load(cls, filepath):
        """Charge un modèle sauvegardé"""
        model_data = joblib.load(filepath)
        instance = cls()
        instance.pipeline = model_data['pipeline']
        instance.label_encoder = model_data['label_encoder']
        instance.training_history = model_data.get('training_history', {})
        return instance