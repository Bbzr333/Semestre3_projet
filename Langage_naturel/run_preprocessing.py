#!/usr/bin/env python3
"""
Script exemple pour preprocesser votre corpus.
Adaptez les chemins selon votre structure de projet.
"""

import sys
from pathlib import Path

# Ajoute src au path si nécessaire
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.preprocessing.preprocessor import GenitivePreprocessor


def main():
    """Lance le preprocessing sur le corpus."""
    
    # Configuration des chemins
    INPUT_CSV = "data/raw/corpus_initial/corpus_A_de_B_relations_150.csv"
    OUTPUT_CSV = "data/processed/corpus_preprocessed.csv"
    
    print("🚀 Lancement du preprocessing")
    print("="*60)
    print(f"📁 Entrée  : {INPUT_CSV}")
    print(f"💾 Sortie  : {OUTPUT_CSV}")
    print(f"🌐 API JDM : Activée")
    print("="*60)
    
    # Crée le dossier de sortie s'il n'existe pas
    output_path = Path(OUTPUT_CSV)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Initialise le preprocessor
    # preprocessor = GenitivePreprocessor(use_jdm=True)

    # Initialise le preprocessor sans JDM pour tests hors ligne
    preprocessor = GenitivePreprocessor(use_jdm=False)
    
    # Lance le preprocessing
    try:
        df = preprocessor.preprocess_csv(INPUT_CSV, OUTPUT_CSV)
        
        print("\n" + "="*60)
        print("✅ PREPROCESSING TERMINÉ AVEC SUCCÈS")
        print("="*60)
        print(f"📊 {len(df)} constructions traitées")
        print(f"✓  {df['est_valide'].sum()} constructions valides")
        print(f"✗  {(~df['est_valide']).sum()} échecs")
        print(f"📈 Taux de succès : {df['est_valide'].sum()/len(df)*100:.1f}%")
        
        # Affiche quelques exemples
        print("\n📋 Aperçu des résultats (5 premières lignes):")
        print("-"*60)
        display_df = df[df['est_valide']][['phrase_originale', 'nom1_lemme', 'nom2_lemme', 'definitude', 'type_jdm']].head(5)
        print(display_df.to_string(index=False))
        
        print(f"\n💾 Fichier sauvegardé : {OUTPUT_CSV}")
        print("="*60)
        
        return 0
        
    except FileNotFoundError:
        print(f"\n❌ ERREUR: Fichier non trouvé : {INPUT_CSV}")
        print("Vérifiez que le chemin est correct.")
        return 1
        
    except Exception as e:
        print(f"\n❌ ERREUR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)