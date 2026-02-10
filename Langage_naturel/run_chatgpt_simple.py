"""
Version simplifiée - Comparaison avec ChatGPT
Utilise requests au lieu de la bibliothèque OpenAI
"""

import pandas as pd
import numpy as np
import sys
import json
import time
import requests
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
import os

sys.path.append('src')

# Configuration
API_URL = "https://api.openai.com/v1/chat/completions"

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
    """Crée un prompt few-shot"""
    prompt = """Tu es un expert en analyse linguistique française. Ta tâche est d'identifier la relation sémantique dans les constructions génitives "A de B".

RELATIONS POSSIBLES:
"""
    for rel, definition in RELATIONS_DEFINITIONS.items():
        prompt += f"\n• {rel}: {definition}"
    
    prompt += "\n\nEXEMPLES:\n"
    
    for relation in sorted(train_df['type_jdm'].unique()):
        examples = train_df[train_df['type_jdm'] == relation].sample(
            min(n_examples_per_class, len(train_df[train_df['type_jdm'] == relation]))
        )
        for _, row in examples.iterrows():
            prompt += f'\nPhrase: "{row["phrase_originale"]}"\nRelation: {relation}\n'
    
    prompt += """\n\nINSTRUCTIONS:
1. Analyse la construction "A de B"
2. Identifie la relation sémantique
3. Réponds UNIQUEMENT avec le nom exact de la relation
4. Ne donne aucune explication

Format: juste le nom de la relation.
"""
    return prompt

def query_chatgpt_api(api_key, phrase, system_prompt, model="gpt-3.5-turbo", max_retries=3):
    """Interroge ChatGPT via requests"""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f'Phrase: "{phrase}"\nRelation:'}
        ],
        "temperature": 0,
        "max_tokens": 50
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.post(API_URL, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            prediction = result['choices'][0]['message']['content'].strip()
            prediction = prediction.replace('.', '').replace(',', '').strip()
            
            # Vérifier si c'est une relation valide
            if prediction in RELATIONS_DEFINITIONS.keys():
                return prediction
            else:
                for rel in RELATIONS_DEFINITIONS.keys():
                    if rel in prediction:
                        return rel
                return prediction
        
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                print(f"  [ATTENTION] Erreur (tentative {attempt + 1}/{max_retries}): {e}")
                time.sleep(2 ** attempt)
            else:
                print(f"  [ERREUR] Echec: {e}")
                return "ERROR"
    
    return "ERROR"

def main():
    print("=" * 70)
    print("COMPARAISON AVEC CHATGPT (Version Simplifiee)")
    print("=" * 70)
    
    # Demander la clé API
    
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
    
    # Charger les données
    print(f"\nChargement des donnees...")
    train = pd.read_csv('data/processed/train.csv')
    test = pd.read_csv('data/processed/test.csv')

    print(f"[OK] Train: {len(train)} exemples")
    print(f"[OK] Test: {len(test)} exemples")
    
    # Configuration
    MODEL = 'gpt-3.5-turbo'  # Moins cher pour commencer
    N_SAMPLES = 50  # Petit échantillon
    
    print(f"\nConfiguration:")
    print(f"  • Modèle: {MODEL}")
    print(f"  • Échantillon: {N_SAMPLES} exemples")
    print(f"  • Coût estimé: ~${(N_SAMPLES * 0.002):.2f}")
    
    response = input("\nContinuer? (y/n): ")
    if response.lower() != 'y':
        print("Annule.")
        return
    
    # Échantillonner
    test_sample = test.sample(N_SAMPLES, random_state=42)
    
    # Créer le prompt
    print(f"\nCreation du prompt few-shot...")
    system_prompt = create_few_shot_prompt(train, n_examples_per_class=2)
    
    # Évaluer
    print(f"\nInterrogation de {MODEL}...")
    predictions = []
    y_true = []
    errors = []
    
    start_time = time.time()
    
    for idx, row in tqdm(test_sample.iterrows(), total=len(test_sample)):
        phrase = row['phrase_originale']
        true_label = row['type_jdm']
        
        prediction = query_chatgpt_api(api_key, phrase, system_prompt, model=MODEL)
        
        predictions.append(prediction)
        y_true.append(true_label)
        
        if prediction != true_label:
            errors.append({
                'phrase': phrase,
                'true': true_label,
                'pred': prediction
            })
        
        time.sleep(0.2)  # Rate limiting
    
    elapsed_time = time.time() - start_time
    
    # Résultats
    print(f"\n{'='*70}")
    print(f"RESULTATS")
    print(f"{'='*70}")
    
    valid_predictions = [p for p in predictions if p != "ERROR"]
    valid_true = [y_true[i] for i, p in enumerate(predictions) if p != "ERROR"]
    
    accuracy = accuracy_score(valid_true, valid_predictions)
    print(f"\n[OK] Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
    print(f"Temps: {elapsed_time:.1f}s ({elapsed_time/N_SAMPLES:.2f}s/exemple)")
    print(f"Erreurs: {len(errors)}/{N_SAMPLES}")
    
    if errors:
        print(f"\nExemples d'erreurs:")
        for i, err in enumerate(errors[:5]):
            print(f"  {i+1}. \"{err['phrase']}\"")
            print(f"     Vrai: {err['true']} | Prédit: {err['pred']}")
    
    # Comparaison avec baseline
    baseline_results = pd.read_csv('results/test_results.csv', index_col=0)
    
    print(f"\n{'='*70}")
    print(f"COMPARAISON AVEC BASELINE")
    print(f"{'='*70}")
    print(f"\nModèle                Accuracy    Temps/exemple")
    print(f"-" * 50)
    print(f"Random Forest         1.000       0.001s")
    print(f"{MODEL:20s}  {accuracy:.3f}       {elapsed_time/N_SAMPLES:.3f}s")
    
    print(f"\nConclusion:")
    if accuracy > 0.95:
        print(f"ChatGPT excellent ({accuracy:.1%}) mais {elapsed_time/N_SAMPLES/0.001:.0f}x plus lent")
    elif accuracy > 0.85:
        print(f"ChatGPT bon ({accuracy:.1%}) mais moins performant que RF (100%)")
    else:
        print(f"[ATTENTION] ChatGPT sous-performe ({accuracy:.1%}) vs RF (100%)")
    
    print(f"\n{'='*70}")
    print(f"EVALUATION TERMINEE")
    print(f"{'='*70}")

if __name__ == '__main__':
    main()