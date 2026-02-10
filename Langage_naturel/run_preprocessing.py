#!/usr/bin/env python3
"""
Script exemple pour preprocesser votre corpus.
Adaptez les chemins selon votre structure de projet.
"""

import sys
from pathlib import Path
import argparse

# Ajoute src au path si nécessaire
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.preprocessing.preprocessor import GenitivePreprocessor


def main(args):
    """Lance le preprocessing sur le corpus."""
    
    # Configuration des chemins
    INPUT_CSV = "data/raw/corpus_initial/corpus_A_de_B_relations_150.csv"
    OUTPUT_CSV = "data/processed/corpus_preprocessed.csv"
    
    print("Lancement du preprocessing")
    print("="*60)
    print(f"Entree  : {INPUT_CSV}")
    print(f"Sortie  : {OUTPUT_CSV}")
    print(f"API JDM : {'Activee' if args.use_jdm else 'Desactivee'}")
    print("="*60)
    
    # Crée le dossier de sortie s'il n'existe pas
    output_path = Path(OUTPUT_CSV)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Initialise le preprocessor
    preprocessor = GenitivePreprocessor(use_jdm=args.use_jdm)
    
    # Lance le preprocessing
    try:
        df = preprocessor.preprocess_csv(INPUT_CSV, OUTPUT_CSV)
        
        print("\n" + "="*60)
        print("PREPROCESSING TERMINE AVEC SUCCES")
        print("="*60)
        print(f"{len(df)} constructions traitees")
        print(f"[OK]  {df['est_valide'].sum()} constructions valides")
        print(f"[ECHEC]  {(~df['est_valide']).sum()} echecs")
        print(f"Taux de succes : {df['est_valide'].sum()/len(df)*100:.1f}%")
        
        # Affiche quelques exemples
        print("\nApercu des resultats (5 premieres lignes):")
        print("-"*60)
        display_df = df[df['est_valide']][['phrase_originale', 'nom1_lemme', 'nom2_lemme', 'definitude', 'type_jdm']].head(5)
        print(display_df.to_string(index=False))
        
        print(f"\nFichier sauvegarde : {OUTPUT_CSV}")
        print("="*60)
        
        return 0
        
    except FileNotFoundError:
        print(f"\nERREUR: Fichier non trouve : {INPUT_CSV}")
        print("Vérifiez que le chemin est correct.")
        return 1
        
    except Exception as e:
        print(f"\nERREUR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocessing du corpus génitif.')
    parser.add_argument(
        '--use-jdm',
        action='store_true',
        help='Activer l\'utilisation de l\'API JeuxDeMots pour une meilleure lemmatisation.'
    )
    args = parser.parse_args()
    exit_code = main(args)
    sys.exit(exit_code)