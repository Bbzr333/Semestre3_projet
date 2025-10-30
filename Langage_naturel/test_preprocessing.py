"""
Script de test pour le module de preprocessing.
Teste les différentes fonctionnalités sans nécessiter le CSV complet.
"""

import sys
from pathlib import Path

# Ajoute le dossier src au path
sys.path.insert(0, str(Path(__file__).parent))

from src.preprocessing.preprocessor import GenitivePreprocessor
from src.utils.jdm_api import get_jdm_api


def test_extraction():
    """Test l'extraction des constructions génitives."""
    print("="*60)
    print("TEST 1: Extraction de constructions")
    print("="*60)
    
    preprocessor = GenitivePreprocessor(use_jdm=False)  # Mode hors ligne
    
    test_cases = [
        "désert d'Algérie",
        "maison de Pierre",
        "chapeau du monsieur",
        "voiture de la professeure",
        "groupe des étudiants",
        "lunettes de soleil",  # Cas simple
        "lunettes de soleil de marque",  # Mot composé
    ]
    
    for phrase in test_cases:
        result = preprocessor.preprocess_construction(phrase)
        if result.est_valide:
            print(f"\n✓ '{phrase}'")
            print(f"  → Nom1: {result.nom1_lemme}")
            print(f"  → Nom2: {result.nom2_lemme}")
            print(f"  → Déterminant: {result.determinant}")
            print(f"  → Définitude: {result.definitude}")
        else:
            print(f"\n✗ '{phrase}' - Échec: {result.notes}")


def test_lemmatization():
    """Test la lemmatisation."""
    print("\n" + "="*60)
    print("TEST 2: Lemmatisation")
    print("="*60)
    
    preprocessor = GenitivePreprocessor(use_jdm=False)
    
    test_words = [
        ("films", "film"),
        ("chevaux", "cheval"),
        ("maisons", "maison"),
        ("animaux", "animal"),
    ]
    
    for word, expected in test_words:
        result = preprocessor.lemmatize(word)
        status = "✓" if result == expected else "✗"
        print(f"{status} {word} → {result} (attendu: {expected})")


def test_jdm_api():
    """Test l'API JeuxDeMots."""
    print("\n" + "="*60)
    print("TEST 3: API JeuxDeMots")
    print("="*60)
    
    try:
        jdm = get_jdm_api()
        
        # Test existence
        test_terms = ["désert", "maison", "termeInexistant123"]
        
        print("\n🔍 Test d'existence des termes:")
        for term in test_terms:
            exists = jdm.term_exists(term)
            status = "✓" if exists else "✗"
            print(f"{status} '{term}': {exists}")
        
        # Test hyperonymes
        print("\n📚 Hyperonymes de 'désert':")
        hypernyms = jdm.get_hypernyms("désert", max_results=5)
        for i, hyp in enumerate(hypernyms, 1):
            print(f"  {i}. {hyp}")
        
        # Test signature complète
        print("\n🔖 Signature de 'désert':")
        signature = jdm.get_signature("désert")
        print(f"  Hyperonymes: {signature['hypernyms'][:3]}")
        print(f"  Types sémantiques: {signature['semantic_types']}")
        print(f"  Types relations: {list(signature['relation_types'].keys())[:5]}")
        
    except Exception as e:
        print(f"⚠️  Erreur API JDM: {e}")
        print("   (Assurez-vous d'avoir une connexion internet)")


def test_full_preprocessing():
    """Test preprocessing complet."""
    print("\n" + "="*60)
    print("TEST 4: Preprocessing complet")
    print("="*60)
    
    preprocessor = GenitivePreprocessor(use_jdm=True)
    
    exemples = [
        ("désert d'Algérie", "lieu"),
        ("portrait de Van Gogh", "auteur_createur"),
        ("tasse de café", "contenant"),
        ("roue de vélo", "partie"),
    ]
    
    for phrase, type_jdm in exemples:
        print(f"\n📝 '{phrase}' ({type_jdm})")
        try:
            result = preprocessor.preprocess_construction(phrase, type_jdm)
            if result.est_valide:
                print(f"  ✓ Extraction réussie")
                print(f"    A: {result.nom1_lemme}")
                print(f"    B: {result.nom2_lemme}")
                print(f"    DEF: {result.definitude}")
            else:
                print(f"  ✗ Échec: {result.notes}")
        except Exception as e:
            print(f"  ⚠️  Erreur: {e}")


def main():
    """Lance tous les tests."""
    print("\n🧪 TESTS DU MODULE DE PREPROCESSING\n")
    
    # test_extraction()
    # test_lemmatization()
    
    # Tests JDM (nécessite connexion internet)
    try:
        test_jdm_api()
        # test_full_preprocessing()
    except Exception as e:
        print(f"\n⚠️  Tests JDM ignorés (pas de connexion): {e}")
    
    print("\n" + "="*60)
    print("✅ Tests terminés !")
    print("="*60)


if __name__ == '__main__':
    main()