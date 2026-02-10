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

    # Modèles SANS régularisation (pour comparaison)
    MODELS_NO_REGULARIZATION = {
        'logistic_regression': LogisticRegression(max_iter=1000, random_state=42),
        'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'svm_linear': SVC(kernel='linear', random_state=42),
        'svm_rbf': SVC(kernel='rbf', random_state=42),
        'gradient_boosting': GradientBoostingClassifier(random_state=42),
    }

    # Modèles AVEC régularisation forte (anti-overfitting)
    MODELS = {
        'logistic_regression': LogisticRegression(
            max_iter=1000,
            C=0.1,                    # Régularisation L2 forte (défaut=1.0)
            penalty='l2',
            random_state=42
        ),
        'random_forest': RandomForestClassifier(
            n_estimators=100,
            max_depth=8,              # Limite la profondeur (défaut=None)
            min_samples_split=10,     # Min échantillons pour split (défaut=2)
            min_samples_leaf=5,       # Min échantillons par feuille (défaut=1)
            max_features='sqrt',      # Réduit les features par split
            random_state=42
        ),
        'svm_linear': SVC(
            kernel='linear',
            C=0.1,                    # Régularisation forte (défaut=1.0)
            random_state=42
        ),
        'svm_rbf': SVC(
            kernel='rbf',
            C=0.1,                    # Régularisation forte
            gamma='scale',
            random_state=42
        ),
        'gradient_boosting': GradientBoostingClassifier(
            n_estimators=100,
            max_depth=4,              # Arbres moins profonds (défaut=3)
            min_samples_split=10,
            min_samples_leaf=5,
            learning_rate=0.05,       # Apprentissage plus lent (défaut=0.1)
            subsample=0.8,            # Bagging pour réduire variance
            random_state=42
        ),
    }
    
    def __init__(self, model_name='random_forest', use_regularization=True):
        """
        Initialise le classifieur.

        Args:
            model_name: Nom du modèle à utiliser
            use_regularization: Si True, utilise les modèles régularisés (anti-overfitting)
        """
        self.model_name = model_name
        self.use_regularization = use_regularization
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()

        # Choisit les modèles avec ou sans régularisation
        models_dict = self.MODELS if use_regularization else self.MODELS_NO_REGULARIZATION

        # Pipeline
        self.pipeline = Pipeline([
            ('scaler', self.scaler),
            ('classifier', models_dict[model_name])
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