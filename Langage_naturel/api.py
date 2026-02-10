"""
API REST pour la classification des relations sémantiques.
Utilise FastAPI avec le modèle CamemBERT fine-tuné.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.models.deep_learning.camembert_classifier import CamemBERTClassifier

# Initialisation
app = FastAPI(
    title="API Classification Sémantique",
    description="Classifie les relations sémantiques dans les constructions génitives françaises (A de B)",
    version="1.0.0",
)

# Chargement du modèle au démarrage
print("Chargement du modèle CamemBERT...")
model = CamemBERTClassifier.load('models/camembert/best_model')
print("Modèle chargé !")

# Descriptions des relations
RELATION_DESCRIPTIONS = {
    'r_has_causatif': 'Cause',
    'r_has_property-1': 'Propriété',
    'r_objet>matiere': 'Matière',
    'r_lieu>origine': 'Origine géographique',
    'r_topic': 'Sujet/Thème',
    'r_depic': 'Représentation',
    'r_holo': 'Partie-Tout',
    'r_lieu': 'Localisation',
    'r_processus_agent': 'Agent du processus',
    'r_processus_patient': 'Patient du processus',
    'r_processus>instr-1': 'Instrument',
    'r_own-1': 'Possession',
    'r_quantificateur': 'Quantité',
    'r_social_tie': 'Lien social',
    'r_product_of': 'Produit de',
}


# Modèles de requête/réponse
class PhraseInput(BaseModel):
    phrase: str

    class Config:
        json_schema_extra = {
            "example": {"phrase": "la porte de la maison"}
        }


class BatchInput(BaseModel):
    phrases: list[str]

    class Config:
        json_schema_extra = {
            "example": {"phrases": ["la porte de la maison", "le livre de Marie"]}
        }


class PredictionResult(BaseModel):
    phrase: str
    relation: str
    description: str
    confidence: float
    probabilities: dict[str, float]


# Endpoints
@app.get("/")
def root():
    """Page d'accueil avec info sur l'API."""
    return {
        "message": "API Classification des Relations Sémantiques",
        "documentation": "/docs",
        "endpoints": {
            "POST /predict": "Prédire une seule phrase",
            "POST /predict/batch": "Prédire plusieurs phrases",
            "GET /relations": "Liste des relations possibles",
        }
    }


@app.get("/relations")
def get_relations():
    """Retourne la liste des 15 relations sémantiques possibles."""
    return {
        "count": len(RELATION_DESCRIPTIONS),
        "relations": RELATION_DESCRIPTIONS
    }


@app.post("/predict", response_model=PredictionResult)
def predict(input: PhraseInput):
    """
    Prédit la relation sémantique pour une phrase génitive.

    Exemple: "la porte de la maison" → r_holo (Partie-Tout)
    """
    if not input.phrase.strip():
        raise HTTPException(status_code=400, detail="La phrase ne peut pas être vide")

    # Prédiction
    prediction = model.predict([input.phrase])[0]
    proba = model.predict_proba([input.phrase])[0]

    # Trouver la confiance max
    max_idx = proba.argmax()
    confidence = float(proba[max_idx])

    # Top 5 probabilités
    proba_dict = {
        label: float(p)
        for label, p in sorted(
            zip(model.label_encoder.classes_, proba),
            key=lambda x: x[1],
            reverse=True
        )[:5]
    }

    return PredictionResult(
        phrase=input.phrase,
        relation=prediction,
        description=RELATION_DESCRIPTIONS.get(prediction, prediction),
        confidence=confidence,
        probabilities=proba_dict
    )


@app.post("/predict/batch")
def predict_batch(input: BatchInput):
    """
    Prédit les relations pour plusieurs phrases en une seule requête.
    Plus efficace que des appels individuels.
    """
    if not input.phrases:
        raise HTTPException(status_code=400, detail="La liste de phrases ne peut pas être vide")

    if len(input.phrases) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 phrases par requête")

    predictions = model.predict(input.phrases)
    probas = model.predict_proba(input.phrases)

    results = []
    for phrase, pred, proba in zip(input.phrases, predictions, probas):
        confidence = float(proba.max())
        results.append({
            "phrase": phrase,
            "relation": pred,
            "description": RELATION_DESCRIPTIONS.get(pred, pred),
            "confidence": confidence
        })

    return {"count": len(results), "predictions": results}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
