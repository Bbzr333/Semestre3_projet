"""
Comparaison avec ChatGPT (GPT-3.5)
Évaluation via API OpenAI avec few-shot prompting
"""

import pandas as pd
import numpy as np
import sys
import json
import time
from pathlib import Path
from openai import OpenAI
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

sys.path.append('src')

# Configuration
RELATIONS_DEFINITIONS = {
    'r_has_causatif': "Relation de cause (A cause B ou B cause A)",
    'r_has_property-1': "A possède la propriété B",
    'r_objet>matiere': "A est fait de la matière B",
    'r_lieu>origine': "A provient du lieu B",
    'r_topic': "A a pour sujet/thème B",
    'r_depic': "A représente/dépeint B",
    'r_holo': "A est une partie de B (relation partie-tout)",
    'r_lieu': "A est situé dans/à B",
    'r_processus_agent': "A est l'agent qui effectue le processus B",
    'r_processus_patient': "A subit le processus B",
    'r_processus>instr-1': "A est l'instrument utilisé pour B",
    'r_own-1': "A appartient à B (possession)",
    'r_quantificateur': "A est une quantité de B",
    'r_social_tie': "Lien social entre A et B",
    'r_product_of': "A est le produit/résultat de B"
}

def create_few_shot_prompt(train_df, n_examples_per_class=2):
    """
    Crée un prompt few-shot avec des exemples pour chaque relation
    """
    prompt = """Tu es un expert en analyse linguistique française. Ta tâche est d'identifier la relation sémantique dans les constructions génitives "A de B".

RELATIONS POSSIBLES:
"""
    
    # Ajouter les définitions
    for rel, definition in RELATIONS_DEFINITIONS.items():
        prompt += f"\n• {rel}: {definition}"
    
    prompt += "\n\nEXEMPLES:\n"
    
    # Ajouter des exemples pour chaque classe
    for relation in sorted(train_df['type_jdm'].unique()):
        examples = train_df[train_df['type_jdm'] == relation].sample(
            min(n_examples_per_class, len(train_df[train_df['type_jdm'] == relation]))
        )
        
        for _, row in examples.iterrows():
            prompt += f'\nPhrase: "{row["phrase_originale"]}"\nRelation: {relation}\n'
    
    prompt += """\n\nINSTRUCTIONS:
1. Analyse la construction "A de B" dans la phrase donnée
2. Identifie quelle relation sémantique lie A et B
3. Réponds UNIQUEMENT avec le nom exact de la relation (ex: r_lieu>origine)
4. Ne donne aucune explication, juste le nom de la relation

Format de réponse: juste le nom de la relation, rien d'autre.
"""
    
    return prompt

def query_chatgpt(client, phrase, system_prompt, model="gpt-4", max_retries=3):
    """
    Interroge ChatGPT avec gestion des erreurs et retry
    """
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f'Phrase: "{phrase}"\nRelation:'}
                ],
                temperature=0,  # Déterministe
                max_tokens=50
            )
            
            prediction = response.choices[0].message.content.strip()
            
            # Nettoyer la réponse (enlever ponctuations, espaces, etc.)
            prediction = prediction.replace('.', '').replace(',', '').strip()
            
            # Vérifier si c'est une relation valide
            if prediction in RELATIONS_DEFINITIONS.keys():
                return prediction
            else:
                # Essayer de trouver la relation dans la réponse
                for rel in RELATIONS_DEFINITIONS.keys():
                    if rel in prediction:
                        return rel
                
                # Si aucune relation trouvée, retourner la prédiction brute
                return prediction
        
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"  [ATTENTION] Erreur (tentative {attempt + 1}/{max_retries}): {e}")
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                print(f"  [ERREUR] Echec apres {max_retries} tentatives: {e}")
                return "ERROR"
    
    return "ERROR"

def evaluate_chatgpt(client, test_df, train_df, model="gpt-4", n_samples=None):
    """
    Évalue ChatGPT sur le test set
    """
    print(f"\n{'='*70}")
    print(f"EVALUATION DE {model.upper()}")
    print(f"{'='*70}")
    
    # Échantillonner si nécessaire
    if n_samples and n_samples < len(test_df):
        test_sample = test_df.sample(n_samples, random_state=42)
        print(f"\nEchantillon: {n_samples} exemples (pour economiser les tokens)")
    else:
        test_sample = test_df
        print(f"\nTest complet: {len(test_sample)} exemples")
    
    # Créer le prompt few-shot
    print(f"\nCreation du prompt few-shot...")
    system_prompt = create_few_shot_prompt(train_df, n_examples_per_class=2)
    prompt_tokens = len(system_prompt.split())
    print(f"  [OK] Prompt: ~{prompt_tokens} mots ({len(system_prompt)} caracteres)")
    
    # Interroger ChatGPT pour chaque exemple
    print(f"\nInterrogation de {model}...")
    predictions = []
    y_true = []
    errors = []
    
    start_time = time.time()
    
    for idx, row in tqdm(test_sample.iterrows(), total=len(test_sample), desc="Prédictions"):
        phrase = row['phrase_originale']
        true_label = row['type_jdm']
        
        prediction = query_chatgpt(client, phrase, system_prompt, model=model)
        
        predictions.append(prediction)
        y_true.append(true_label)
        
        if prediction != true_label:
            errors.append({
                'phrase': phrase,
                'true': true_label,
                'pred': prediction
            })
        
        # Rate limiting léger
        time.sleep(0.1)
    
    elapsed_time = time.time() - start_time
    
    # Calcul des métriques
    print(f"\nRESULTATS")
    print(f"{'='*70}")
    
    # Filtrer les erreurs
    valid_predictions = [p for p in predictions if p != "ERROR"]
    valid_true = [y_true[i] for i, p in enumerate(predictions) if p != "ERROR"]
    
    if len(valid_predictions) < len(predictions):
        print(f"[ATTENTION] {len(predictions) - len(valid_predictions)} erreurs API (reponses invalides)")
    
    # Accuracy
    accuracy = accuracy_score(valid_true, valid_predictions)
    print(f"\n[OK] Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")

    # Temps
    print(f"Temps total: {elapsed_time:.1f}s ({elapsed_time/len(test_sample):.2f}s/exemple)")

    # Rapport détaillé
    print(f"\nRapport par Classe:")
    report = classification_report(valid_true, valid_predictions, zero_division=0)
    print(report)
    
    # Matrice de confusion
    cm = confusion_matrix(valid_true, valid_predictions, labels=list(RELATIONS_DEFINITIONS.keys()))
    
    # Analyse des erreurs
    if errors:
        print(f"\nAnalyse des Erreurs ({len(errors)} erreurs):")
        
        # Top confusions
        error_df = pd.DataFrame(errors)
        confusions = error_df.groupby(['true', 'pred']).size().sort_values(ascending=False).head(10)
        
        print(f"\n  Top 10 Confusions:")
        for (true_label, pred_label), count in confusions.items():
            print(f"    • {true_label} → {pred_label}: {count} fois")
            example = error_df[(error_df['true'] == true_label) & (error_df['pred'] == pred_label)].iloc[0]
            print(f"      Exemple: \"{example['phrase']}\"")
    
    return {
        'model': model,
        'accuracy': accuracy,
        'predictions': predictions,
        'y_true': y_true,
        'errors': errors,
        'confusion_matrix': cm,
        'elapsed_time': elapsed_time,
        'n_samples': len(test_sample),
        'classification_report': report
    }

def plot_comparison(results_dict, save_path):
    """
    Compare les résultats de tous les modèles
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 1. Comparaison des accuracies
    ax = axes[0]
    models = list(results_dict.keys())
    accuracies = [results_dict[m]['accuracy'] for m in models]
    colors = ['#3498db' if 'GPT' in m or 'gpt' in m else '#2ecc71' for m in models]
    
    bars = ax.barh(models, accuracies, color=colors, alpha=0.7)
    ax.set_xlabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Comparaison des Performances', fontsize=14, fontweight='bold')
    ax.set_xlim([0.8, 1.0])
    ax.grid(axis='x', alpha=0.3)
    
    # Ajouter les valeurs
    for i, (model, acc) in enumerate(zip(models, accuracies)):
        ax.text(acc + 0.005, i, f'{acc:.3f}', va='center', fontweight='bold')
    
    # 2. Temps d'exécution
    ax = axes[1]
    times = [results_dict[m].get('elapsed_time', 0) / results_dict[m].get('n_samples', 1) 
             for m in models]
    
    bars = ax.barh(models, times, color=colors, alpha=0.7)
    ax.set_xlabel('Temps par Exemple (secondes)', fontsize=12, fontweight='bold')
    ax.set_title('Efficacité Computationnelle', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # Ajouter les valeurs
    for i, (model, t) in enumerate(zip(models, times)):
        ax.text(t + 0.01, i, f'{t:.3f}s', va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Comparaison sauvegardee: {save_path}")

def main():
    print("=" * 70)
    print("COMPARAISON AVEC CHATGPT")
    print("=" * 70)
    
    # Vérifier la clé API
    api_key = None

    # 1. Essayer de charger depuis config.json
    try:
        with open('data/apiKey/config.json', 'r') as f:
            config = json.load(f)
            api_key = config.get('openai_api_key')
            if api_key:
                print(f"[OK] Cle API chargee depuis config.json")
    except FileNotFoundError:
        print("[ATTENTION] Fichier config.json non trouve")
    except json.JSONDecodeError:
        print("[ATTENTION] Erreur de lecture du fichier config.json")

    if not api_key:
        print("\nCle API OpenAI non trouvee dans les variables d'environnement")
        api_key = input("Entrez votre clé API OpenAI (sk-...): ").strip()
        
        if not api_key.startswith('sk-'):
            print("[ERREUR] Cle API invalide (doit commencer par 'sk-')")
            return
    
    print(f"[OK] Cle API configuree")

    if not api_key:
        print("\n[ERREUR] Cle API OpenAI non trouvee!")
        print("\nPour configurer la cle API:")
        print("   export OPENAI_API_KEY='votre-clé-api'")
        print("\nOu créer un fichier .env avec:")
        print("   OPENAI_API_KEY=votre-clé-api")
        return
    
    # Initialiser le client OpenAI
    client = OpenAI(api_key=api_key)
    print(f"[OK] Client OpenAI initialise")
    
    # Charger les données
    print(f"\nChargement des donnees...")
    train = pd.read_csv('data/processed/train.csv')
    test = pd.read_csv('data/processed/test.csv')

    print(f"[OK] Train: {len(train)} exemples")
    print(f"[OK] Test: {len(test)} exemples")
    
    # Charger les résultats des modèles baseline
    baseline_results = pd.read_csv('results/test_results.csv', index_col=0)
    
    # Configuration
    MODELS_TO_TEST = ['gpt-3.5-turbo']
    N_SAMPLES = 100  # Échantillon pour économiser les tokens (modifiable)
    
    print(f"\nConfiguration:")
    print(f"  • Modèles: {', '.join(MODELS_TO_TEST)}")
    print(f"  • Échantillon de test: {N_SAMPLES} exemples")
    print(f"  • Coût estimé: ~${(N_SAMPLES * 0.002 * len(MODELS_TO_TEST)):.2f}")
    
    # Demander confirmation
    response = input("\nContinuer? (y/n): ")
    if response.lower() != 'y':
        print("Annule.")
        return
    
    # Évaluer chaque modèle
    all_results = {}
    
    for model_name in MODELS_TO_TEST:
        try:
            results = evaluate_chatgpt(
                client, 
                test, 
                train, 
                model=model_name,
                n_samples=N_SAMPLES
            )
            all_results[model_name] = results
            
            # Sauvegarder les erreurs
            if results['errors']:
                error_df = pd.DataFrame(results['errors'])
                error_path = Path('results') / f'errors_{model_name.replace("-", "_")}.csv'
                error_df.to_csv(error_path, index=False)
                print(f"  Erreurs sauvegardees: {error_path}")
        
        except Exception as e:
            print(f"\n[ERREUR] Erreur avec {model_name}: {e}")
            continue
    
    # Ajouter les résultats baseline pour comparaison
    print(f"\nAjout des resultats baseline...")
    for model in baseline_results.index:
        if model in ['random_forest', 'gradient_boosting', 'svm_linear']:
            all_results[model] = {
                'accuracy': baseline_results.loc[model, 'accuracy'],
                'elapsed_time': 0.001,  # Très rapide
                'n_samples': 338  # Test complet
            }
    
    # Comparaison finale
    print(f"\n{'='*70}")
    print(f"COMPARAISON FINALE")
    print(f"{'='*70}")
    
    comparison = []
    for model_name, results in all_results.items():
        comparison.append({
            'Modèle': model_name,
            'Accuracy': f"{results['accuracy']:.3f}",
            'Temps/exemple': f"{results.get('elapsed_time', 0) / results.get('n_samples', 1):.3f}s",
            'N échantillon': results.get('n_samples', '-')
        })
    
    df_comparison = pd.DataFrame(comparison)
    df_comparison = df_comparison.sort_values('Accuracy', ascending=False)
    print("\n" + df_comparison.to_string(index=False))
    
    # Sauvegarder
    results_dir = Path('results')
    plots_dir = results_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    comparison_path = results_dir / 'chatgpt_comparison.csv'
    df_comparison.to_csv(comparison_path, index=False)
    print(f"\nComparaison sauvegardee: {comparison_path}")
    
    # Graphique de comparaison
    plot_path = plots_dir / 'chatgpt_vs_baseline.png'
    plot_comparison(all_results, plot_path)
    
    # Conclusion
    print(f"\n{'='*70}")
    print(f"CONCLUSIONS")
    print(f"{'='*70}")
    
    best_model = df_comparison.iloc[0]['Modèle']
    best_acc = float(df_comparison.iloc[0]['Accuracy'])
    
    print(f"\nMeilleur modele: {best_model} ({best_acc:.3f})")
    
    if 'gpt' in best_model.lower():
        print(f"\nChatGPT surpasse les modeles classiques!")
        print(f"   Mais au prix d'un temps d'exécution ~100x plus lent")
    else:
        print(f"\nLes modeles classiques restent competitifs!")
        print(f"   Avec l'avantage d'être beaucoup plus rapides et gratuits")
    
    print(f"\n{'='*70}")
    print(f"EVALUATION TERMINEE")
    print(f"{'='*70}")

if __name__ == '__main__':
    main()