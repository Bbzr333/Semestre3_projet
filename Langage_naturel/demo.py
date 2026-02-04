"""
Interface de démonstration pour la classification des relations sémantiques.
Compare CamemBERT et le meilleur modèle baseline sur les constructions génitives françaises.
Inclut un Knowledge Graph navigable.

Version améliorée avec extraction de features JDM en temps réel.
Tous les modèles baseline sont évalués et le plus performant est affiché.
"""

import re
import os
import sys
import base64
import numpy as np
import pandas as pd
import gradio as gr
from pyvis.network import Network

sys.path.insert(0, 'src')

from src.models.deep_learning.camembert_classifier import CamemBERTClassifier
from src.models.baseline_models import BaselineClassifier
from src.preprocessing.preprocessor import GenitivePreprocessor
from src.features.feature_extractor import EnhancedFeatureExtractor

# Dossier pour les fichiers temporaires du graphe
GRAPH_DIR = os.path.join(os.path.dirname(__file__), "temp_graphs")
os.makedirs(GRAPH_DIR, exist_ok=True)

# Chargement des modèles
print("Chargement des modèles...")
print("  - CamemBERT...")
camembert_model = CamemBERTClassifier.load('models/camembert/best_model')

# Charger TOUS les modèles baseline
print("  - Modèles baseline...")
baseline_models = {}
MODEL_NAMES = ['random_forest', 'gradient_boosting', 'logistic_regression', 'svm_linear', 'svm_rbf']
for model_name in MODEL_NAMES:
    try:
        model_path = f'models/baseline/{model_name}.joblib'
        baseline_models[model_name] = BaselineClassifier.load(model_path)
        print(f"    ✓ {model_name}")
    except Exception as e:
        print(f"    ✗ {model_name}: {e}")

# Noms lisibles pour l'affichage
MODEL_DISPLAY_NAMES = {
    'random_forest': 'Random Forest',
    'gradient_boosting': 'Gradient Boosting',
    'logistic_regression': 'Régression Logistique',
    'svm_linear': 'SVM Linéaire',
    'svm_rbf': 'SVM RBF'
}
print(f"Modèles chargés: {len(baseline_models)} baseline + CamemBERT")

# Initialisation du préprocesseur et extracteur de features (avec JDM)
print("Initialisation de l'extracteur de features JDM...")
preprocessor = GenitivePreprocessor(use_jdm=True)
feature_extractor = EnhancedFeatureExtractor(use_jdm=True)
print("Extracteur JDM initialisé !")

# Charger les données pour les exemples
test_df = pd.read_csv('data/processed/test.csv')
train_df = pd.read_csv('data/processed/train.csv')

# Colonnes de features - récupérées DIRECTEMENT du premier modèle entraîné
# C'est critique pour éviter les erreurs de dimension
try:
    first_model = list(baseline_models.values())[0]
    FEATURE_COLUMNS = list(first_model.pipeline.named_steps['scaler'].feature_names_in_)
    print(f"Features baseline (depuis modèle): {len(FEATURE_COLUMNS)} colonnes")
except (AttributeError, IndexError):
    # Fallback: calculer depuis train.csv si le modèle n'a pas l'info
    excluded_cols = [
        'phrase_originale', 'nom1', 'nom2', 'determinant',
        'nom1_lemme', 'nom2_lemme', 'type_jdm', 'definitude',
        'est_valide', 'notes'
    ]
    numeric_cols = train_df.select_dtypes(include=['int64', 'float64', 'int32', 'float32', 'bool']).columns.tolist()
    FEATURE_COLUMNS = [col for col in numeric_cols if col not in excluded_cols]
    constant_cols = [col for col in FEATURE_COLUMNS if train_df[col].std() == 0]
    FEATURE_COLUMNS = [col for col in FEATURE_COLUMNS if col not in constant_cols]
    print(f"Features baseline (depuis train.csv): {len(FEATURE_COLUMNS)} colonnes")

print(f"Premières features: {FEATURE_COLUMNS[:5]}")
print(f"Dernières features: {FEATURE_COLUMNS[-5:]}")
del train_df  # Libérer la mémoire

# Mapping des codes vers descriptions lisibles
RELATION_DESCRIPTIONS = {
    'r_has_causatif': 'Cause',
    'r_has_property-1': 'Propriété',
    'r_objet>matiere': 'Matière',
    'r_lieu>origine': 'Origine',
    'r_topic': 'Sujet',
    'r_depic': 'Représentation',
    'r_holo': 'Partie-Tout',
    'r_lieu': 'Localisation',
    'r_processus_agent': 'Agent',
    'r_processus_patient': 'Patient',
    'r_processus>instr-1': 'Instrument',
    'r_own-1': 'Possession',
    'r_quantificateur': 'Quantité',
    'r_social_tie': 'Lien social',
    'r_product_of': 'Produit de',
}

# Couleurs par type de relation
RELATION_COLORS = {
    'r_has_causatif': '#e74c3c',
    'r_has_property-1': '#9b59b6',
    'r_objet>matiere': '#8b4513',
    'r_lieu>origine': '#27ae60',
    'r_topic': '#3498db',
    'r_depic': '#e91e63',
    'r_holo': '#f39c12',
    'r_lieu': '#1abc9c',
    'r_processus_agent': '#e67e22',
    'r_processus_patient': '#d35400',
    'r_processus>instr-1': '#7f8c8d',
    'r_own-1': '#2980b9',
    'r_quantificateur': '#16a085',
    'r_social_tie': '#c0392b',
    'r_product_of': '#8e44ad',
}


def extract_features_realtime(phrase: str) -> np.ndarray | None:
    """
    Extrait les features JDM en temps réel pour une nouvelle phrase.

    Returns:
        Array de features ou None si échec
    """
    try:
        # 1. Préprocesser la phrase
        construction = preprocessor.preprocess_construction(phrase)
        if not construction.est_valide:
            print(f"Preprocessing échoué pour: {phrase}")
            return None

        # 2. Créer un DataFrame avec les colonnes nécessaires
        row_data = {
            'phrase_originale': construction.phrase_originale,
            'nom1': construction.nom1,
            'nom2': construction.nom2,
            'determinant': construction.determinant,
            'nom1_lemme': construction.nom1_lemme,
            'nom2_lemme': construction.nom2_lemme,
            'definitude': construction.definitude,
        }
        df_single = pd.DataFrame([row_data])

        # 3. Extraire les features
        features_df = feature_extractor.extract_features(df_single, verbose=False)

        # 4. Aligner avec les colonnes attendues par le modèle
        # Créer un DataFrame avec exactement les colonnes attendues, initialisées à 0
        features_aligned_df = pd.DataFrame(0.0, index=[0], columns=FEATURE_COLUMNS)

        # Copier les valeurs des colonnes communes
        common_cols = [c for c in FEATURE_COLUMNS if c in features_df.columns]
        for col in common_cols:
            features_aligned_df[col] = features_df[col].values[0]

        features_aligned = features_aligned_df.values.reshape(1, -1)

        # Remplacer NaN et inf
        features_aligned = np.nan_to_num(features_aligned, nan=0.0, posinf=0.0, neginf=0.0)

        return features_aligned

    except Exception as e:
        print(f"Erreur extraction features: {e}")
        import traceback
        traceback.print_exc()
        return None


def get_test_example(phrase: str) -> dict | None:
    """Récupère un exemple du test set avec ses features et vraie classe."""
    matches = test_df[test_df['phrase_originale'].str.lower() == phrase.lower()]
    if len(matches) > 0:
        row = matches.iloc[0]
        # Aligner les features avec celles attendues par le modèle
        features_aligned = pd.DataFrame(0.0, index=[0], columns=FEATURE_COLUMNS)
        common_cols = [c for c in FEATURE_COLUMNS if c in test_df.columns]
        for col in common_cols:
            features_aligned[col] = row[col]
        features_array = features_aligned.values.reshape(1, -1)
        features_array = np.nan_to_num(features_array, nan=0.0, posinf=0.0, neginf=0.0)
        return {
            'features': features_array,
            'true_label': row['type_jdm'],
            'phrase': row['phrase_originale'],
            'from_test_set': True
        }
    return None


def extract_entities(phrase: str) -> tuple[str, str] | None:
    """Extrait les entités A et B d'une construction "A de B"."""
    patterns = [
        r"^(.+?)\s+de\s+la\s+(.+)$",
        r"^(.+?)\s+de\s+l['\u2019](.+)$",
        r"^(.+?)\s+du\s+(.+)$",
        r"^(.+?)\s+des\s+(.+)$",
        r"^(.+?)\s+de\s+(.+)$",
        r"^(.+?)\s+d['\u2019](.+)$",
    ]
    phrase = phrase.strip().lower()
    for pattern in patterns:
        match = re.match(pattern, phrase, re.IGNORECASE)
        if match:
            return match.group(1).strip(), match.group(2).strip()
    return None


def format_prediction_html(pred: str, proba: list, labels: list, true_label: str = "") -> str:
    """
    Formate les prédictions en HTML avec indicateurs visuels.
    """
    # Trier par probabilité décroissante
    sorted_preds = sorted(zip(labels, proba), key=lambda x: x[1], reverse=True)

    html = "<div style='font-family: Arial, sans-serif;'>"

    for i, (label, prob) in enumerate(sorted_preds[:8]):  # Top 8 prédictions
        desc = RELATION_DESCRIPTIONS.get(label, label)
        percentage = prob * 100

        # Indicateur correct/incorrect
        if true_label:
            if label == true_label:
                indicator = "<span style='color: #27ae60; font-weight: bold;'>✓</span>"
                bg_color = "#d4edda"  # Vert clair
            else:
                indicator = "<span style='color: #e74c3c;'>✗</span>"
                bg_color = "#f8f9fa" if i > 0 else "#f8d7da"  # Rouge clair pour top-1 incorrect
        else:
            indicator = ""
            bg_color = "#f8f9fa"

        # Style pour la prédiction principale
        if i == 0:
            html += f"""
            <div style='background: {bg_color}; padding: 10px; border-radius: 8px; margin-bottom: 10px; border: 2px solid #ddd;'>
                <div style='font-size: 16px; font-weight: bold;'>
                    {indicator} {desc}
                </div>
                <div style='color: #888; font-size: 12px; font-style: italic;'>{label}</div>
                <div style='margin-top: 5px;'>
                    <div style='background: #e9ecef; border-radius: 4px; overflow: hidden;'>
                        <div style='background: #3498db; height: 20px; width: {percentage}%;
                                    display: flex; align-items: center; justify-content: flex-end;
                                    padding-right: 5px; color: white; font-size: 12px;'>
                            {percentage:.1f}%
                        </div>
                    </div>
                </div>
            </div>
            """
        else:
            # Autres prédictions en plus compact
            bar_width = max(percentage * 2, 2)  # Scale pour visibilité
            html += f"""
            <div style='display: flex; align-items: center; margin: 4px 0; padding: 4px 8px;
                        background: {bg_color}; border-radius: 4px;'>
                <span style='width: 20px;'>{indicator}</span>
                <span style='flex: 1; font-size: 13px;'>{desc}</span>
                <span style='color: #888; font-size: 11px; font-style: italic; margin-right: 8px;'>{label}</span>
                <div style='width: 80px; background: #e9ecef; border-radius: 3px; overflow: hidden; margin-right: 5px;'>
                    <div style='background: #95a5a6; height: 12px; width: {bar_width}%;'></div>
                </div>
                <span style='font-size: 12px; width: 45px; text-align: right;'>{percentage:.1f}%</span>
            </div>
            """

    html += "</div>"
    return html


def predict_with_label(phrase: str, true_label: str) -> tuple[str, str, str]:
    """
    Prédit avec CamemBERT et le meilleur modèle baseline, retourne HTML formaté.
    """
    if not phrase.strip():
        empty = "<p style='color: #666;'>Entrez une phrase pour voir les prédictions</p>"
        return empty, empty, ""

    # Info sur la vraie classe
    if true_label:
        true_desc = RELATION_DESCRIPTIONS.get(true_label, true_label)
        label_info = f"**Vraie classe:** {true_desc} *({true_label})*"
    else:
        label_info = ""

    # Prédiction CamemBERT
    camembert_proba = camembert_model.predict_proba([phrase])[0]
    camembert_html = format_prediction_html(
        camembert_model.predict([phrase])[0],
        camembert_proba,
        camembert_model.label_encoder.classes_,
        true_label
    )

    # Obtenir les features (depuis test set ou extraction temps réel)
    test_example = get_test_example(phrase)
    if test_example:
        features = test_example['features']
        realtime_indicator = ""
    else:
        features = extract_features_realtime(phrase)
        realtime_indicator = """
        <div style='padding: 5px 10px; background: #d1ecf1; border-radius: 5px; margin-bottom: 10px; font-size: 12px; color: #0c5460;'>
            ⚡ Features JDM extraites en temps réel
        </div>
        """

    if features is not None:
        # Prédiction sur TOUS les modèles baseline
        all_predictions = {}
        for model_name, model in baseline_models.items():
            try:
                proba = model.predict_proba(features)[0]
                pred = model.predict(features)[0]
                max_confidence = float(proba.max())
                all_predictions[model_name] = {
                    'prediction': pred,
                    'proba': proba,
                    'confidence': max_confidence,
                    'labels': model.label_encoder.classes_
                }
            except Exception as e:
                print(f"Erreur {model_name}: {e}")

        # Trouver le meilleur modèle (plus haute confiance)
        if all_predictions:
            best_model_name = max(all_predictions.keys(),
                                  key=lambda k: all_predictions[k]['confidence'])
            best = all_predictions[best_model_name]
            best_display_name = MODEL_DISPLAY_NAMES.get(best_model_name, best_model_name)

            # Créer le HTML pour le meilleur modèle
            best_html = format_prediction_html(
                best['prediction'],
                best['proba'],
                best['labels'],
                true_label
            )

            # Ajouter un header indiquant quel modèle a été sélectionné
            model_header = f"""
            <div style='padding: 8px 12px; background: #e8f5e9; border-radius: 5px; margin-bottom: 10px; font-size: 13px; color: #2e7d32;'>
                🏆 <strong>{best_display_name}</strong> (confiance: {best['confidence']*100:.1f}%)
            </div>
            """
            best_html = realtime_indicator + model_header + best_html
        else:
            best_html = """
            <div style='padding: 20px; background: #f8d7da; border-radius: 8px; text-align: center;'>
                <div style='font-size: 24px;'>❌</div>
                <div style='margin-top: 10px; color: #721c24;'>Aucun modèle baseline disponible</div>
            </div>
            """
    else:
        best_html = """
        <div style='padding: 20px; background: #f8d7da; border-radius: 8px; text-align: center;'>
            <div style='font-size: 24px;'>❌</div>
            <div style='margin-top: 10px; color: #721c24;'>
                Impossible d'extraire les features<br>
                <small>Vérifiez que la phrase suit le format "A de B"</small>
            </div>
        </div>
        """

    return camembert_html, best_html, label_info


def build_knowledge_graph(phrases_text: str) -> str:
    """Construit un Knowledge Graph interactif à partir de plusieurs phrases."""
    phrases = [p.strip() for p in phrases_text.strip().split('\n') if p.strip()]

    if not phrases:
        return "<p style='text-align:center; color:#666;'>Entrez des phrases pour construire le graphe</p>"

    net = Network(
        height="500px",
        width="100%",
        bgcolor="#ffffff",
        font_color="#333333",
        directed=True,
    )

    net.set_options("""
    {
        "physics": {
            "forceAtlas2Based": {
                "gravitationalConstant": -100,
                "centralGravity": 0.01,
                "springLength": 200,
                "springConstant": 0.08
            },
            "solver": "forceAtlas2Based",
            "stabilization": {"iterations": 100}
        },
        "nodes": {
            "font": {"size": 14, "face": "arial"},
            "shape": "box",
            "margin": 10
        },
        "edges": {
            "font": {"size": 12, "align": "middle"},
            "arrows": {"to": {"enabled": true, "scaleFactor": 0.5}},
            "smooth": {"type": "curvedCW", "roundness": 0.2}
        },
        "interaction": {
            "hover": true,
            "navigationButtons": true,
            "keyboard": true
        }
    }
    """)

    nodes_added = set()
    edges_info = []

    for phrase in phrases:
        entities = extract_entities(phrase)
        if entities is None:
            continue

        entity_a, entity_b = entities
        prediction = camembert_model.predict([phrase])[0]
        proba = camembert_model.predict_proba([phrase])[0]
        confidence = float(proba.max())

        relation_label = RELATION_DESCRIPTIONS.get(prediction, prediction)
        color = RELATION_COLORS.get(prediction, '#95a5a6')

        for entity in [entity_a, entity_b]:
            if entity not in nodes_added:
                net.add_node(entity, label=entity, title=entity, color="#3498db", borderWidth=2)
                nodes_added.add(entity)

        edge_title = f"{phrase}\n{prediction} ({confidence:.1%})"
        net.add_edge(entity_a, entity_b, label=relation_label, title=edge_title,
                     color=color, width=2 + confidence * 3)

        edges_info.append({'relation': prediction, 'label': relation_label})

    if not nodes_added:
        return "<p style='text-align:center; color:#e74c3c;'>Aucune construction 'A de B' détectée</p>"

    graph_path = os.path.join(GRAPH_DIR, "knowledge_graph.html")
    net.save_graph(graph_path)

    with open(graph_path, 'r', encoding='utf-8') as f:
        graph_html = f.read()

    relations_used = set(e['relation'] for e in edges_info)
    legend_items = []
    for rel in relations_used:
        color = RELATION_COLORS.get(rel, '#95a5a6')
        label = RELATION_DESCRIPTIONS.get(rel, rel)
        legend_items.append(
            f'<span style="display:inline-block; margin:3px 10px 3px 0;">'
            f'<span style="display:inline-block; width:12px; height:12px; '
            f'background:{color}; margin-right:5px; border-radius:2px;"></span>'
            f'{label}</span>'
        )

    legend_html = "".join(legend_items)
    stats = f"<strong>{len(nodes_added)}</strong> entités, <strong>{len(edges_info)}</strong> relations"
    graph_b64 = base64.b64encode(graph_html.encode('utf-8')).decode('utf-8')

    return f"""
    <div style="margin-bottom:10px; padding:10px; background:#f8f9fa; border-radius:5px;">
        <strong>Légende:</strong> {legend_html}
    </div>
    <div style="margin-bottom:10px; color:#666;">{stats}</div>
    <iframe src="data:text/html;base64,{graph_b64}"
            width="100%" height="550px"
            style="border:1px solid #ddd; border-radius:5px;">
    </iframe>
    """


# Exemples avec vraies classes du test set (phrase, vraie_classe)
# Format interne: [phrase, code_technique] - le code est converti en français pour l'affichage
EXAMPLES_WITH_LABELS = [
    ["la porte de la maison", "r_holo"],
    ["le livre d'économie", "r_topic"],
    ["le vin de Bordeaux", "r_lieu>origine"],
    ["un morceau de fromage", "r_quantificateur"],
    ["la symphonie de mozart", "r_product_of"],
    ["le mentor de jules", "r_social_tie"],
    ["la poutre d'acier", "r_objet>matiere"],
    ["la montre du sportif", "r_own-1"],
    ["les embouteillages des travaux", "r_has_causatif"],
    ["la douceur de l'infirmière", "r_has_property-1"],
    ["le nettoyage de l'agent", "r_processus_agent"],
    ["la sculpture du bois", "r_processus_patient"],
    ["la scie de coupe", "r_processus>instr-1"],
    ["le marché de marrakech", "r_lieu"],
]

# Version avec labels français pour l'affichage dans l'interface
EXAMPLES_DISPLAY = [
    [phrase, RELATION_DESCRIPTIONS.get(label, label)]
    for phrase, label in EXAMPLES_WITH_LABELS
]

EXAMPLE_KG = """la porte de la maison
les fenêtres de la maison
le toit de la maison
le jardin de la maison
le moteur de la voiture
les roues de la voiture
le volant de la voiture
le livre de Marie
l'ami de Marie
le chien de Marie
la voiture de Pierre
le bureau de Paul
le vin de Bordeaux
le fromage de France
le champagne de Reims
les oranges d'Espagne
le café du Brésil
le tableau de Picasso
le portrait de Marie
la symphonie de Mozart
le roman de Zola
la sculpture de Rodin
la douceur du miel
la beauté du paysage
la force du vent
la chaleur du soleil
le bruit de la rue
la table de bois
le mur de pierre
la statue de bronze
le vase de cristal
le nettoyage de l'agent
la construction de l'ouvrier
la coupe du coiffeur
le travail de l'artisan
le couteau de cuisine
le marteau de forgeron
la scie du menuisier
le pinceau du peintre
le mentor de Jules
le collègue de Sophie
le voisin de Marc
l'élève du professeur"""


# Interface avec onglets
with gr.Blocks(title="Classification Sémantique") as demo:
    gr.Markdown("""
    # Classification des Relations Sémantiques
    **Comparaison CamemBERT vs Meilleur Modèle Baseline** sur les constructions génitives françaises

    ✓ = Prédiction correcte | Sélectionnez la vraie classe pour évaluer les prédictions
    """)

    with gr.Tabs():
        # Onglet Prédiction
        with gr.TabItem("Comparaison Modèles"):
            # Options pour le dropdown de vraie classe
            TRUE_CLASS_OPTIONS = [""] + [
                f"{code} ({RELATION_DESCRIPTIONS.get(code, code)})"
                for code in RELATION_DESCRIPTIONS.keys()
            ]

            with gr.Row():
                with gr.Column(scale=1):
                    input_phrase = gr.Textbox(
                        label="Phrase génitive",
                        placeholder="ex: la porte de la maison",
                        lines=2,
                    )
                    true_label_dropdown = gr.Dropdown(
                        label="Vraie classe (optionnel - pour évaluer)",
                        choices=TRUE_CLASS_OPTIONS,
                        value="",
                        allow_custom_value=True,
                    )
                    # Champ caché pour le code technique
                    true_label_input = gr.Textbox(visible=False)
                    predict_btn = gr.Button("Comparer les modèles", variant="primary")

                    gr.Markdown("### Exemples du test set")
                    gr.Markdown("*Cliquez pour charger (avec vraie classe)*")

                    examples_display = gr.Dataframe(
                        headers=["Phrase", "Relation"],
                        value=EXAMPLES_DISPLAY,
                        interactive=False,
                        wrap=True,
                    )

                with gr.Column(scale=2):
                    true_label_display = gr.Markdown("")

                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### 🤖 CamemBERT")
                            camembert_output = gr.HTML()

                        with gr.Column():
                            gr.Markdown("### 🏆 Meilleur Baseline")
                            best_baseline_output = gr.HTML()

            # Event handlers
            def on_example_select(evt: gr.SelectData):
                """Quand on clique sur un exemple, remplit la phrase et la vraie classe."""
                row = EXAMPLES_WITH_LABELS[evt.index[0]]
                phrase = row[0]
                code = row[1]
                # Format pour le dropdown: "r_holo (Partie-Tout)"
                dropdown_value = f"{code} ({RELATION_DESCRIPTIONS.get(code, code)})"
                return phrase, dropdown_value, code

            def extract_code_from_dropdown(dropdown_value: str) -> str:
                """Extrait le code technique du dropdown (ex: 'r_holo (Partie-Tout)' -> 'r_holo')."""
                if not dropdown_value or dropdown_value == "":
                    return ""
                # Le code est avant le premier espace ou parenthèse
                return dropdown_value.split(" ")[0].split("(")[0].strip()

            def predict_wrapper(phrase: str, dropdown_value: str):
                """Wrapper pour extraire le code avant prédiction."""
                true_label = extract_code_from_dropdown(dropdown_value)
                return predict_with_label(phrase, true_label)

            examples_display.select(
                on_example_select,
                outputs=[input_phrase, true_label_dropdown, true_label_input]
            )

            predict_btn.click(
                predict_wrapper,
                inputs=[input_phrase, true_label_dropdown],
                outputs=[camembert_output, best_baseline_output, true_label_display]
            )

            input_phrase.submit(
                predict_wrapper,
                inputs=[input_phrase, true_label_dropdown],
                outputs=[camembert_output, best_baseline_output, true_label_display]
            )

        # Onglet Knowledge Graph
        with gr.TabItem("Knowledge Graph"):
            gr.Markdown("""
            Entrez plusieurs phrases (une par ligne) pour construire un graphe de connaissances.

            **Navigation:** Zoom molette, glisser pour déplacer, cliquer pour fixer un nœud.
            """)

            with gr.Row():
                with gr.Column(scale=1):
                    input_phrases = gr.Textbox(
                        label="Phrases (une par ligne)",
                        placeholder="la porte de la maison\nle livre de Marie\n...",
                        lines=12,
                        value=EXAMPLE_KG,
                    )
                    build_btn = gr.Button("Construire le graphe", variant="primary")

                with gr.Column(scale=2):
                    output_graph = gr.HTML(label="Knowledge Graph")

            build_btn.click(build_knowledge_graph, inputs=input_phrases, outputs=output_graph)
            demo.load(build_knowledge_graph, inputs=input_phrases, outputs=output_graph)


if __name__ == "__main__":
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860, theme=gr.themes.Soft())
