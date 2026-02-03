"""
Interface de démonstration pour la classification des relations sémantiques.
Utilise CamemBERT fine-tuné sur les constructions génitives françaises.
"""

import gradio as gr
from src.models.deep_learning.camembert_classifier import CamemBERTClassifier

# Chargement du modèle
print("Chargement du modèle CamemBERT...")
model = CamemBERTClassifier.load('models/camembert/best_model')
print("Modèle chargé !")

# Mapping des codes vers descriptions lisibles
RELATION_DESCRIPTIONS = {
    'r_has_causatif': 'Cause (ex: les retards de la pluie)',
    'r_has_property-1': 'Propriété (ex: la générosité du bénévole)',
    'r_objet>matiere': 'Matière (ex: une table de bois)',
    'r_lieu>origine': 'Origine géographique (ex: le vin de Bordeaux)',
    'r_topic': 'Sujet/Thème (ex: un livre d\'histoire)',
    'r_depic': 'Représentation (ex: le portrait de Marie)',
    'r_holo': 'Partie-Tout (ex: la porte de la maison)',
    'r_lieu': 'Localisation (ex: les gens de la ville)',
    'r_processus_agent': 'Agent du processus (ex: le discours du président)',
    'r_processus_patient': 'Patient du processus (ex: la sculpture du bois)',
    'r_processus>instr-1': 'Instrument (ex: le marteau de forgeron)',
    'r_own-1': 'Possession (ex: le livre de Marie)',
    'r_quantificateur': 'Quantité (ex: un kilo de pommes)',
    'r_social_tie': 'Lien social (ex: l\'ami de Pierre)',
    'r_product_of': 'Produit de (ex: le tableau de Picasso)',
}


def predict(phrase: str) -> tuple[str, dict]:
    """
    Prédit la relation sémantique pour une phrase génitive.

    Args:
        phrase: Construction génitive française (ex: "la porte de la maison")

    Returns:
        - Relation prédite avec description
        - Dictionnaire des probabilités par classe
    """
    if not phrase.strip():
        return "Veuillez entrer une phrase", {}

    # Prédiction
    prediction = model.predict([phrase])[0]
    proba = model.predict_proba([phrase])[0]

    # Formatage du résultat principal
    description = RELATION_DESCRIPTIONS.get(prediction, prediction)
    result = f"{prediction}\n{description}"

    # Formatage des probabilités avec descriptions
    proba_dict = {}
    for label, p in zip(model.label_encoder.classes_, proba):
        display_label = f"{label}"
        proba_dict[display_label] = float(p)

    return result, proba_dict


# Exemples pour l'interface
EXAMPLES = [
    ["la porte de la maison"],
    ["le livre de Marie"],
    ["le vin de Bordeaux"],
    ["un kilo de pommes"],
    ["le discours du président"],
    ["une table de bois"],
    ["l'ami de Pierre"],
    ["le portrait de Marie"],
    ["la générosité du bénévole"],
    ["un livre d'histoire"],
]

# Création de l'interface
demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(
        label="Phrase génitive",
        placeholder="Entrez une construction 'A de B' (ex: la porte de la maison)",
        lines=2,
    ),
    outputs=[
        gr.Textbox(label="Relation prédite", lines=2),
        gr.Label(label="Probabilités par classe", num_top_classes=5),
    ],
    title="🔍 Classification des Relations Sémantiques",
    description="""
    **Classifieur CamemBERT** pour les constructions génitives françaises.

    Entrez une phrase du type "A de B" pour identifier la relation sémantique
    entre A et B parmi 15 catégories possibles.

    *Accuracy: 100% sur le jeu de test (338 exemples)*
    """,
    examples=EXAMPLES,
    theme=gr.themes.Soft(),
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch(
        share=False,  # Mettre True pour générer un lien public temporaire
        server_name="0.0.0.0",  # Accessible sur le réseau local
        server_port=7860,
    )
