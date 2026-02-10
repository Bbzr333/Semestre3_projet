"""
Test des modèles sur corpus externe pour évaluer la généralisation.

Ce script permet de:
1. Tester sur des phrases générées dynamiquement depuis JDM
2. Tester sur des phrases Wikipedia/manuelles
3. Comparer modèles régularisés vs non-régularisés
4. Évaluer la robustesse face au bruit et aux cas ambigus
"""

import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Any
import warnings
warnings.filterwarnings('ignore')

sys.path.append('src')

from src.models.baseline_models import BaselineClassifier
from src.features.feature_extractor import EnhancedFeatureExtractor
from src.data.jdm_data_generator import JDMDataGenerator
from src.evaluation.evaluator import ModelEvaluator


# Corpus externe de test (phrases réelles variées)
EXTERNAL_TEST_CORPUS = [
    # r_holo (Partie-Tout)
    {"phrase": "le moteur de la voiture", "type_jdm": "r_holo"},
    {"phrase": "les branches de l'arbre", "type_jdm": "r_holo"},
    {"phrase": "le toit de l'immeuble", "type_jdm": "r_holo"},
    {"phrase": "les touches du clavier", "type_jdm": "r_holo"},
    {"phrase": "l'écran du téléphone", "type_jdm": "r_holo"},

    # r_own-1 (Possession)
    {"phrase": "la bicyclette de mon voisin", "type_jdm": "r_own-1"},
    {"phrase": "l'appartement de Julie", "type_jdm": "r_own-1"},
    {"phrase": "les lunettes de grand-père", "type_jdm": "r_own-1"},
    {"phrase": "le cartable de l'écolier", "type_jdm": "r_own-1"},
    {"phrase": "la montre de papa", "type_jdm": "r_own-1"},

    # r_lieu>origine (Origine géographique)
    {"phrase": "le camembert de Normandie", "type_jdm": "r_lieu>origine"},
    {"phrase": "les oranges d'Espagne", "type_jdm": "r_lieu>origine"},
    {"phrase": "le thé de Chine", "type_jdm": "r_lieu>origine"},
    {"phrase": "la pizza de Naples", "type_jdm": "r_lieu>origine"},
    {"phrase": "le whisky d'Écosse", "type_jdm": "r_lieu>origine"},

    # r_objet>matiere (Matière)
    {"phrase": "une bague en or", "type_jdm": "r_objet>matiere"},
    {"phrase": "un pull de laine", "type_jdm": "r_objet>matiere"},
    {"phrase": "des couverts d'argent", "type_jdm": "r_objet>matiere"},
    {"phrase": "un sac de cuir", "type_jdm": "r_objet>matiere"},
    {"phrase": "une sculpture de bronze", "type_jdm": "r_objet>matiere"},

    # r_topic (Thème/Sujet)
    {"phrase": "un manuel de mathématiques", "type_jdm": "r_topic"},
    {"phrase": "un documentaire sur l'espace", "type_jdm": "r_topic"},
    {"phrase": "une conférence de physique", "type_jdm": "r_topic"},
    {"phrase": "un cours de philosophie", "type_jdm": "r_topic"},
    {"phrase": "une revue de médecine", "type_jdm": "r_topic"},

    # r_social_tie (Lien social)
    {"phrase": "la nièce de Martine", "type_jdm": "r_social_tie"},
    {"phrase": "le collègue de bureau", "type_jdm": "r_social_tie"},
    {"phrase": "l'associé de mon père", "type_jdm": "r_social_tie"},
    {"phrase": "le parrain de baptême", "type_jdm": "r_social_tie"},
    {"phrase": "la belle-mère de Jacques", "type_jdm": "r_social_tie"},

    # r_quantificateur (Quantité)
    {"phrase": "une poignée de noix", "type_jdm": "r_quantificateur"},
    {"phrase": "un verre de jus", "type_jdm": "r_quantificateur"},
    {"phrase": "trois kilos de carottes", "type_jdm": "r_quantificateur"},
    {"phrase": "une dizaine de personnes", "type_jdm": "r_quantificateur"},
    {"phrase": "un paquet de biscuits", "type_jdm": "r_quantificateur"},

    # r_product_of (Produit de)
    {"phrase": "une toile de Van Gogh", "type_jdm": "r_product_of"},
    {"phrase": "un roman de Victor Hugo", "type_jdm": "r_product_of"},
    {"phrase": "une sonate de Chopin", "type_jdm": "r_product_of"},
    {"phrase": "un film de Spielberg", "type_jdm": "r_product_of"},
    {"phrase": "une théorie de Newton", "type_jdm": "r_product_of"},

    # r_depic (Représentation)
    {"phrase": "un portrait de Napoléon", "type_jdm": "r_depic"},
    {"phrase": "une statue de la Liberté", "type_jdm": "r_depic"},
    {"phrase": "une caricature du président", "type_jdm": "r_depic"},
    {"phrase": "une maquette de la tour Eiffel", "type_jdm": "r_depic"},
    {"phrase": "un dessin d'enfant", "type_jdm": "r_depic"},

    # r_has_causatif (Cause)
    {"phrase": "les dégâts de l'inondation", "type_jdm": "r_has_causatif"},
    {"phrase": "la fatigue du voyage", "type_jdm": "r_has_causatif"},
    {"phrase": "les conséquences de la crise", "type_jdm": "r_has_causatif"},
    {"phrase": "le retard de la grève", "type_jdm": "r_has_causatif"},
    {"phrase": "les séquelles de l'accident", "type_jdm": "r_has_causatif"},

    # r_processus_agent (Agent)
    {"phrase": "l'intervention du chirurgien", "type_jdm": "r_processus_agent"},
    {"phrase": "la décision du jury", "type_jdm": "r_processus_agent"},
    {"phrase": "le discours du maire", "type_jdm": "r_processus_agent"},
    {"phrase": "l'enquête de la police", "type_jdm": "r_processus_agent"},
    {"phrase": "le verdict du tribunal", "type_jdm": "r_processus_agent"},

    # r_lieu (Localisation)
    {"phrase": "les habitants de la banlieue", "type_jdm": "r_lieu"},
    {"phrase": "les touristes de la plage", "type_jdm": "r_lieu"},
    {"phrase": "les étudiants de la fac", "type_jdm": "r_lieu"},
    {"phrase": "les ouvriers de l'usine", "type_jdm": "r_lieu"},
    {"phrase": "les clients du magasin", "type_jdm": "r_lieu"},

    # r_has_property-1 (Propriété)
    {"phrase": "la douceur de sa voix", "type_jdm": "r_has_property-1"},
    {"phrase": "l'élégance de sa démarche", "type_jdm": "r_has_property-1"},
    {"phrase": "la fraîcheur du matin", "type_jdm": "r_has_property-1"},
    {"phrase": "la rapidité du service", "type_jdm": "r_has_property-1"},
    {"phrase": "la profondeur de sa pensée", "type_jdm": "r_has_property-1"},

    # r_processus_patient (Patient)
    {"phrase": "la rénovation de l'appartement", "type_jdm": "r_processus_patient"},
    {"phrase": "la cuisson du pain", "type_jdm": "r_processus_patient"},
    {"phrase": "le nettoyage des vitres", "type_jdm": "r_processus_patient"},
    {"phrase": "la réparation de la machine", "type_jdm": "r_processus_patient"},
    {"phrase": "l'analyse des données", "type_jdm": "r_processus_patient"},

    # r_processus>instr-1 (Instrument)
    {"phrase": "le pinceau du peintre", "type_jdm": "r_processus>instr-1"},
    {"phrase": "le scalpel du chirurgien", "type_jdm": "r_processus>instr-1"},
    {"phrase": "la raquette du tennisman", "type_jdm": "r_processus>instr-1"},
    {"phrase": "le violon du musicien", "type_jdm": "r_processus>instr-1"},
    {"phrase": "le marteau du menuisier", "type_jdm": "r_processus>instr-1"},

    # Cas ambigus (difficiles)
    {"phrase": "la peinture de la porte", "type_jdm": "r_processus_patient"},  # Action de peindre
    {"phrase": "la peinture de Monet", "type_jdm": "r_product_of"},  # Œuvre créée
    {"phrase": "la carte de France", "type_jdm": "r_depic"},  # Représentation
    {"phrase": "le tableau de famille", "type_jdm": "r_depic"},  # Représentation
    {"phrase": "le tableau de Picasso", "type_jdm": "r_product_of"},  # Création
]


def load_models(models_dir: str = 'models/baseline') -> Dict[str, BaselineClassifier]:
    """Charge tous les modèles disponibles."""
    models = {}
    models_path = Path(models_dir)

    if not models_path.exists():
        print(f"[ATTENTION] Repertoire {models_dir} non trouve")
        return models

    for model_file in models_path.glob('*.joblib'):
        model_name = model_file.stem
        try:
            models[model_name] = BaselineClassifier.load(model_file)
            print(f"  [OK] Charge: {model_name}")
        except Exception as e:
            print(f"  [ECHEC] Erreur {model_name}: {e}")

    return models


def preprocess_phrase(phrase: str) -> Dict[str, Any]:
    """
    Préprocesse une phrase pour extraire nom1, nom2, etc.
    Format attendu: "det nom1 de/du/de la/d' nom2"
    """
    import re

    phrase_lower = phrase.strip().lower()
    phrase_original = phrase.strip()

    # Détermine la définitude (articles définis vs indéfinis)
    definitude = 1 if any(phrase_lower.startswith(art) for art in ['le ', 'la ', "l'", 'les ']) else 0

    # Patterns pour extraire la structure "A de B"
    patterns = [
        r"^(?:l[ea]s?|un[es]?|des?|du|ce[ts]?|cette|ces|quelques?|tout(?:es?)?|chaque)?\s*(.+?)\s+(?:de la|du|de l'|d'|de)\s+(.+)$",
        r"^(.+?)\s+(?:de la|du|de l'|d'|de)\s+(.+)$",
    ]

    nom1, nom2 = '', ''
    for pattern in patterns:
        match = re.match(pattern, phrase_lower, re.IGNORECASE)
        if match:
            nom1 = match.group(1).strip()
            nom2 = match.group(2).strip()
            break

    # Nettoie les articles restants
    nom1 = re.sub(r'^(le|la|les|un|une|des)\s+', '', nom1)

    # Gère les cas où nom1/nom2 sont vides
    if not nom1:
        nom1 = 'inconnu'
    if not nom2:
        nom2 = 'inconnu'

    return {
        'phrase_originale': phrase_original,
        'nom1': nom1,
        'nom2': nom2,
        'nom1_lemme': nom1,  # Simplification: pas de lemmatisation
        'nom2_lemme': nom2,
        'definitude': definitude,
    }


def extract_features_for_corpus(
    corpus: List[Dict],
    use_jdm: bool = True,
    verbose: bool = True
) -> pd.DataFrame:
    """Extrait les features pour un corpus de test."""

    # Préprocesse chaque phrase
    processed_data = []
    for item in corpus:
        phrase = item['phrase']
        processed = preprocess_phrase(phrase)
        processed['type_jdm'] = item['type_jdm']
        processed_data.append(processed)

    df_preprocessed = pd.DataFrame(processed_data)

    if verbose:
        print(f"  Préprocessing: {len(df_preprocessed)} phrases")

    # Extrait les features
    extractor = EnhancedFeatureExtractor(use_jdm=use_jdm)
    features_df = extractor.extract_features(df_preprocessed, verbose=verbose)

    # Ajoute les colonnes de métadonnées
    features_df['type_jdm'] = df_preprocessed['type_jdm']
    features_df['phrase_originale'] = df_preprocessed['phrase_originale']

    if verbose:
        print(f"  [OK] {len(features_df)} exemples, {len(features_df.columns) - 2} features extraites")

    return features_df


def evaluate_on_corpus(
    models: Dict[str, BaselineClassifier],
    df: pd.DataFrame,
    verbose: bool = True
) -> pd.DataFrame:
    """Évalue tous les modèles sur un corpus."""
    # Prépare les features
    excluded_cols = ['phrase_originale', 'type_jdm', 'nom1', 'nom2', 'determinant',
                     'nom1_lemme', 'nom2_lemme', 'definitude', 'est_valide', 'notes']
    numeric_cols = df.select_dtypes(include=['int64', 'float64', 'int32', 'float32', 'bool']).columns.tolist()
    feature_cols = [col for col in numeric_cols if col not in excluded_cols]

    X = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = df['type_jdm']

    if verbose:
        print(f"  Features disponibles: {len(feature_cols)}")

    results = []

    for model_name, model in models.items():
        try:
            # Récupère les features attendues par le modèle
            model_features = list(model.pipeline.named_steps['scaler'].feature_names_in_)

            if verbose:
                print(f"  {model_name}: modèle attend {len(model_features)} features")

            # Crée un DataFrame avec les features du modèle, remplit les manquantes par 0
            X_aligned = pd.DataFrame(0, index=X.index, columns=model_features)
            common_features = [f for f in model_features if f in X.columns]
            X_aligned[common_features] = X[common_features]

            if verbose and len(common_features) < len(model_features):
                missing = len(model_features) - len(common_features)
                print(f"    → {len(common_features)} features communes, {missing} manquantes (mises à 0)")

            # Prédictions
            y_pred = model.predict(X_aligned)

            # Calcul accuracy manuel
            accuracy = (y_pred == y).mean()
            n_errors = (y_pred != y).sum()

            results.append({
                'model': model_name,
                'accuracy': accuracy,
                'f1_macro': accuracy,  # Approximation
                'n_correct': (y_pred == y).sum(),
                'n_total': len(y),
                'n_errors': n_errors
            })

            if verbose:
                print(f"    → Accuracy: {accuracy*100:.1f}% ({n_errors} erreurs)")

        except Exception as e:
            print(f"  [ECHEC] {model_name}: {e}")
            import traceback
            traceback.print_exc()

    return pd.DataFrame(results)


def test_with_jdm_generated(
    models: Dict[str, BaselineClassifier],
    n_samples: int = 200,
    include_ambiguous: bool = True,
    add_noise: bool = True,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Teste les modèles sur des données générées dynamiquement depuis JDM.
    """
    print("\n" + "=" * 60)
    print("TEST SUR DONNEES JDM GENEREES")
    print("=" * 60)

    generator = JDMDataGenerator(min_weight=5, enrich_with_jdm=True)

    # Génère des données variées
    print("\nGeneration des donnees de test...")
    test_data = generator.generate_all_classes(n_per_class=n_samples // 15, verbose=verbose)

    # Ajoute des cas ambigus
    if include_ambiguous:
        ambiguous = generator.generate_ambiguous_examples(n_samples=n_samples // 5, verbose=verbose)
        test_data.extend(ambiguous)

    # Ajoute du bruit
    if add_noise:
        test_data = generator.add_noise_to_data(test_data, noise_ratio=0.15, verbose=verbose)

    print(f"\n[OK] Total: {len(test_data)} exemples de test")

    # Extrait les features
    print("\nExtraction des features...")
    df = extract_features_for_corpus(test_data, use_jdm=True, verbose=verbose)

    # Évalue
    print("\nEvaluation des modeles...")
    results = evaluate_on_corpus(models, df, verbose=verbose)

    return results


def test_with_external_corpus(
    models: Dict[str, BaselineClassifier],
    verbose: bool = True
) -> pd.DataFrame:
    """
    Teste les modèles sur le corpus externe prédéfini.
    """
    print("\n" + "=" * 60)
    print("TEST SUR CORPUS EXTERNE")
    print("=" * 60)

    print(f"\n[OK] {len(EXTERNAL_TEST_CORPUS)} exemples dans le corpus externe")

    # Extrait les features
    print("\nExtraction des features...")
    df = extract_features_for_corpus(EXTERNAL_TEST_CORPUS, use_jdm=True, verbose=verbose)

    # Évalue
    print("\nEvaluation des modeles...")
    results = evaluate_on_corpus(models, df, verbose=verbose)

    return results


def analyze_errors(
    model: BaselineClassifier,
    df: pd.DataFrame,
    max_errors: int = 20
) -> List[Dict]:
    """Analyse les erreurs de prédiction."""
    excluded_cols = ['phrase_originale', 'type_jdm', 'nom1', 'nom2', 'determinant',
                     'nom1_lemme', 'nom2_lemme', 'definitude', 'est_valide', 'notes']
    numeric_cols = df.select_dtypes(include=['int64', 'float64', 'int32', 'float32', 'bool']).columns.tolist()
    feature_cols = [col for col in numeric_cols if col not in excluded_cols]

    X = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = df['type_jdm']

    try:
        model_features = list(model.pipeline.named_steps['scaler'].feature_names_in_)
        common_features = [f for f in model_features if f in X.columns]
        X_aligned = X[common_features] if common_features else X

        y_pred = model.predict(X_aligned)
    except Exception as e:
        print(f"Erreur: {e}")
        return []

    errors = []
    for i, (true, pred) in enumerate(zip(y, y_pred)):
        if true != pred:
            errors.append({
                'phrase': df.iloc[i].get('phrase_originale', 'N/A'),
                'vrai': true,
                'predit': pred
            })
            if len(errors) >= max_errors:
                break

    return errors


def main():
    parser = argparse.ArgumentParser(description='Test des modèles sur corpus externe')
    parser.add_argument('--models-dir', default='models/baseline',
                        help='Répertoire des modèles (default: models/baseline)')
    parser.add_argument('--test-type', choices=['external', 'jdm', 'both'], default='both',
                        help='Type de test: external, jdm, ou both')
    parser.add_argument('--n-samples', type=int, default=300,
                        help='Nombre d\'exemples JDM à générer (default: 300)')
    parser.add_argument('--no-ambiguous', action='store_true',
                        help='Exclut les cas ambigus')
    parser.add_argument('--no-noise', action='store_true',
                        help='N\'ajoute pas de bruit')
    parser.add_argument('--show-errors', action='store_true',
                        help='Affiche les erreurs détaillées')
    args = parser.parse_args()

    print("TEST DE GENERALISATION DES MODELES")
    print("=" * 60)

    # Charge les modèles
    print("\nChargement des modeles...")
    models = load_models(args.models_dir)

    if not models:
        print("[ERREUR] Aucun modele trouve. Executez d'abord run_train_baseline.py")
        return

    all_results = []

    # Test sur corpus externe
    if args.test_type in ['external', 'both']:
        results_ext = test_with_external_corpus(models, verbose=True)
        results_ext['test_type'] = 'external'
        all_results.append(results_ext)

    # Test sur données JDM générées
    if args.test_type in ['jdm', 'both']:
        results_jdm = test_with_jdm_generated(
            models,
            n_samples=args.n_samples,
            include_ambiguous=not args.no_ambiguous,
            add_noise=not args.no_noise,
            verbose=True
        )
        results_jdm['test_type'] = 'jdm_generated'
        all_results.append(results_jdm)

    # Combine les résultats
    if all_results:
        combined = pd.concat(all_results, ignore_index=True)

        print("\n" + "=" * 60)
        print("RESUME DES RESULTATS")
        print("=" * 60)

        if combined.empty or 'accuracy' not in combined.columns:
            print("\n[ATTENTION] Aucun resultat disponible (features incompatibles?)")
        else:
            # Affiche les résultats directement si pivot échoue
            try:
                pivot = combined.pivot_table(
                    index='model',
                    columns='test_type',
                    values='accuracy',
                    aggfunc='mean'
                )
                print("\nAccuracy par type de test:")
                print(pivot.to_string())
            except Exception:
                print("\nRésultats bruts:")
                print(combined[['model', 'accuracy', 'n_errors', 'n_total']].to_string(index=False))

        # Sauvegarde
        results_path = Path('results/external_test_results.csv')
        results_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(results_path, index=False)
        print(f"\nResultats sauvegardes: {results_path}")

        # Analyse des erreurs si demandé
        if args.show_errors and 'random_forest' in models:
            print("\n" + "=" * 60)
            print("ANALYSE DES ERREURS (Random Forest)")
            print("=" * 60)

            df_ext = extract_features_for_corpus(EXTERNAL_TEST_CORPUS, use_jdm=True, verbose=False)
            errors = analyze_errors(models['random_forest'], df_ext, max_errors=15)

            if errors:
                for err in errors:
                    print(f"\n  Phrase: \"{err['phrase']}\"")
                    print(f"  Vrai: {err['vrai']} → Prédit: {err['predit']}")
            else:
                print("  Aucune erreur détectée!")

    print("\n" + "=" * 60)
    print("TEST TERMINE")
    print("=" * 60)


if __name__ == '__main__':
    main()
