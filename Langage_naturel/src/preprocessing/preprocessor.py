"""
Preprocessor pour les constructions génitives "A de B".

Implémente les étapes de preprocessing décrites dans l'article GRASP-it :
- Extraction de la construction A de B (et variantes)
- Identification des mots composés
- Lemmatisation
- Normalisation de la casse
- Désambiguïsation
- Gestion des entités nommées
- Préservation du trait de définitude (DEF)
"""

import re
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import sys
from pathlib import Path

# Ajoute le dossier parent au path pour les imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.jdm_api import get_jdm_api


@dataclass
class GenitiveConstruction:
    """Représente une construction génitive extraite et preprocessée."""
    
    phrase_originale: str
    nom1: str  # Tête nominale (A)
    nom2: str  # Complément (B)
    determinant: str  # de/d'/du/de la/des
    nom1_lemme: str
    nom2_lemme: str
    definitude: bool  # True si déterminant défini (du/de la/des)
    type_jdm: Optional[str] = None
    est_valide: bool = True
    notes: str = ""


class GenitivePreprocessor:
    """Preprocesseur pour les constructions génitives françaises."""
    
    # Patterns pour détecter les constructions génitives
    # Note: Capturer plusieurs mots pour nom2 (noms composés type "Van Gogh")
    GENITIVE_PATTERNS = [
        r'\b(\w+)\s+d\'([\w\s]+?)(?:\s+de\s+|\s+d\'|\s+du\s+|\s+des\s+|$)',           # A d'B
        r'\b(\w+)\s+du\s+([\w\s]+?)(?:\s+de\s+|\s+d\'|\s+du\s+|\s+des\s+|$)',          # A du B
        r'\b(\w+)\s+de\s+la\s+([\w\s]+?)(?:\s+de\s+|\s+d\'|\s+du\s+|\s+des\s+|$)',     # A de la B
        r'\b(\w+)\s+des\s+([\w\s]+?)(?:\s+de\s+|\s+d\'|\s+du\s+|\s+des\s+|$)',         # A des B
        r'\b(\w+)\s+de\s+l\'([\w\s]+?)(?:\s+de\s+|\s+d\'|\s+du\s+|\s+des\s+|$)',       # A de l'B
        r'\b(\w+)\s+de\s+([\w\s]+?)(?:\s+de\s+|\s+d\'|\s+du\s+|\s+des\s+|$)',          # A de B (en dernier)
    ]
    
    # Mapping déterminants contractés
    DETERMINANTS = {
        "d'": {'form': "d'", 'defini': False},
        'de': {'form': 'de', 'defini': False},
        'du': {'form': 'du', 'defini': True},
        'de la': {'form': 'de la', 'defini': True},
        'des': {'form': 'des', 'defini': True},
        "de l'": {'form': "de l'", 'defini': False},
    }
    
    def __init__(self, use_jdm: bool = True):
        """
        Initialise le preprocesseur.
        
        Args:
            use_jdm: Si True, utilise l'API JDM pour la validation et la désambiguïsation
        """
        self.use_jdm = use_jdm
        self.jdm = get_jdm_api() if use_jdm else None
        self.stats = {
            'total_traites': 0,
            'extractions_reussies': 0,
            'mots_composes_detectes': 0,
            'entites_remplacees': 0
        }
    
    def extract_genitive_construction(self, phrase: str) -> Optional[Tuple[str, str, str]]:
        """
        Extrait la construction "A de B" d'une phrase.
        Gère les variantes : de/d'/du/de la/des/de l'
        Gère les noms composés dans B (ex: "Van Gogh")
        
        Args:
            phrase: Phrase à analyser
            
        Returns:
            Tuple (nom1, determinant, nom2) ou None
        """
        phrase_lower = phrase.lower().strip()
        
        # Teste chaque pattern (ordre important: du plus spécifique au plus général)
        for pattern in self.GENITIVE_PATTERNS:
            match = re.search(pattern, phrase_lower)
            if match:
                nom1 = match.group(1).strip()
                nom2 = match.group(2).strip()

                # Enlève les articles définis capturés avec nom2
                if nom2.startswith('la '):
                    nom2 = nom2[3:]
                elif nom2.startswith('le '):
                    nom2 = nom2[3:]
                elif nom2.startswith('les '):
                    nom2 = nom2[4:]
                elif nom2.startswith("l'"):
                    nom2 = nom2[2:]

                
                # Nettoie nom2 : enlève les mots après un second "de"
                # Ex: "Van Gogh de Paris" → "Van Gogh"
                if ' de ' in nom2:
                    nom2 = nom2.split(' de ')[0].strip()
                if " d'" in nom2:
                    nom2 = nom2.split(" d'")[0].strip()
                if ' du ' in nom2:
                    nom2 = nom2.split(' du ')[0].strip()
                if ' des ' in nom2:
                    nom2 = nom2.split(' des ')[0].strip()
                
                # Extrait le déterminant utilisé
                det_text = phrase_lower[match.start():match.end()]
                determinant = self._identify_determinant(det_text)
                
                return (nom1, determinant, nom2)
        
        return None
    
    def _identify_determinant(self, text: str) -> str:
        """Identifie le déterminant dans le texte extrait."""
        for det in self.DETERMINANTS.keys():
            if det in text:
                return det
        return 'de'  # Par défaut
    
    def handle_compound_words(self, phrase: str) -> Tuple[str, str, str]:
        """
        Gère les phrases avec plusieurs "de" en identifiant les mots composés.
        Exemple : "lunettes de soleil de marque" → (lunettes_de_soleil, de, marque)
        
        Args:
            phrase: Phrase à analyser
            
        Returns:
            Tuple (nom1, determinant, nom2) avec mots composés identifiés
        """
        # Compte le nombre de "de" et variantes
        de_variants = [' de ', " d'", ' du ', ' des ', ' de la ', " de l'"]
        de_count = sum(phrase.lower().count(variant) for variant in de_variants)
        
        if de_count < 2:
            # Cas simple : un seul "de"
            result = self.extract_genitive_construction(phrase)
            return result if result else (None, None, None)
        
        # Cas complexe : plusieurs "de"
        # Stratégie : tester les différentes décompositions possibles
        # en privilégiant les mots composés connus dans JDM
        
        words = phrase.lower().split()
        best_split = None
        
        # Parcourt toutes les positions possibles de split
        for i in range(1, len(words)):
            if words[i] not in ['de', "d'", 'du', 'des', 'de la', "de l'"]:
                continue
            
            # Teste la décomposition à la position i
            candidate_a = ' '.join(words[:i])
            remaining = ' '.join(words[i:])
            
            # Vérifie si A existe comme mot composé dans JDM
            if self.use_jdm and self.jdm.is_compound_word(candidate_a):
                # A est un mot composé connu, on prend ce split
                result = self.extract_genitive_construction(remaining)
                if result:
                    _, det, nom2 = result
                    self.stats['mots_composes_detectes'] += 1
                    return (candidate_a.replace(' ', '_'), det, nom2)
                best_split = (candidate_a, words[i], ' '.join(words[i+1:]))
        
        # Si aucun mot composé trouvé dans JDM, prendre le dernier "de"
        # Trouve la dernière occurrence de "de"
        last_de_idx = -1
        for i, word in enumerate(words):
            if word in ['de', "d'", 'du', 'des']:
                last_de_idx = i
        
        if last_de_idx > 0:
            nom1 = ' '.join(words[:last_de_idx])
            det = words[last_de_idx]
            nom2 = ' '.join(words[last_de_idx+1:])
            return (nom1, det, nom2)
        
        # Fallback : extraction simple
        result = self.extract_genitive_construction(phrase)
        return result if result else (None, None, None)
    
    def normalize_case(self, text: str) -> str:
        """
        Normalise la casse en minuscules.
        
        Args:
            text: Texte à normaliser
            
        Returns:
            Texte en minuscules
        """
        return text.lower().strip()
    
    def lemmatize(self, term: str) -> str:
        """
        Lemmatise un terme (forme canonique).
        Pour l'instant, implémentation basique. Peut être améliorée avec Spacy.
        
        Args:
            term: Terme à lemmatiser
            
        Returns:
            Forme lemmatisée
        """
        # Règles basiques de lemmatisation française
        term = term.lower()
        
        # Pluriels réguliers
        if term.endswith('aux'):
            return term[:-3] + 'al'
        elif term.endswith('s') and len(term) > 2:
            return term[:-1]
        elif term.endswith('x') and len(term) > 2:
            return term[:-1]
        
        return term
    
    def disambiguate(self, term: str, context: Optional[str] = None) -> str:
        """
        Désambiguïse un terme polysémique en sélectionnant le sens approprié.
        
        Args:
            term: Terme à désambiguïser
            context: Contexte pour aider à la désambiguïsation
            
        Returns:
            Terme désambiguïsé (pour l'instant, retourne le terme tel quel)
        """
        # TODO: Implémenter la désambiguïsation avec JDM
        # Pour l'instant, retourne le terme normalisé
        return self.normalize_case(term)
    
    def handle_named_entities(self, term: str) -> str:
        """
        Gère les entités nommées inconnues en les remplaçant par des équivalents connus.
        
        Args:
            term: Terme à vérifier
            
        Returns:
            Terme original ou équivalent connu
        """
        if not self.use_jdm:
            return term
        
        # Vérifie si le terme existe dans JDM
        if not self.jdm.term_exists(term):
            # Stratégies de remplacement
            # 1. Si c'est un prénom/nom propre, remplacer par un équivalent connu
            if term[0].isupper():
                # TODO: Implémenter mapping prénoms/noms connus
                # Pour l'instant, on garde le terme tel quel
                self.stats['entites_remplacees'] += 1
                return term
        
        return term
    
    def preprocess_construction(self, phrase: str, type_jdm: Optional[str] = None) -> GenitiveConstruction:
        """
        Preprocess complet d'une construction génitive.
        
        Args:
            phrase: Phrase à preprocesser
            type_jdm: Type de relation (optionnel, pour le corpus annoté)
            
        Returns:
            Objet GenitiveConstruction preprocessé
        """
        self.stats['total_traites'] += 1
        
        # 1. Extraction de la construction
        extraction = self.handle_compound_words(phrase)
        
        if not extraction or None in extraction:
            return GenitiveConstruction(
                phrase_originale=phrase,
                nom1="", nom2="", determinant="",
                nom1_lemme="", nom2_lemme="",
                definitude=False,
                type_jdm=type_jdm,
                est_valide=False,
                notes="Échec extraction construction"
            )
        
        nom1, determinant, nom2 = extraction
        
        # 2. Normalisation de la casse
        nom1 = self.normalize_case(nom1)
        nom2 = self.normalize_case(nom2)
        
        # 3. Gestion des entités nommées
        nom1 = self.handle_named_entities(nom1)
        nom2 = self.handle_named_entities(nom2)
        
        # 4. Lemmatisation
        nom1_lemme = self.lemmatize(nom1)
        nom2_lemme = self.lemmatize(nom2)
        
        # 5. Désambiguïsation (si nécessaire)
        nom1_lemme = self.disambiguate(nom1_lemme, phrase)
        nom2_lemme = self.disambiguate(nom2_lemme, phrase)
        
        # 6. Extraction du trait de définitude
        definitude = self.DETERMINANTS.get(determinant, {}).get('defini', False)
        
        self.stats['extractions_reussies'] += 1
        
        return GenitiveConstruction(
            phrase_originale=phrase,
            nom1=nom1,
            nom2=nom2,
            determinant=determinant,
            nom1_lemme=nom1_lemme,
            nom2_lemme=nom2_lemme,
            definitude=definitude,
            type_jdm=type_jdm,
            est_valide=True,
            notes=""
        )
    
    def preprocess_csv(self, csv_path: str, output_path: Optional[str] = None) -> pd.DataFrame:
        """
        Preprocess un fichier CSV contenant des constructions génitives.
        
        Format attendu du CSV :
        - Colonnes : phrase, type_jdm
        
        Args:
            csv_path: Chemin vers le fichier CSV d'entrée
            output_path: Chemin de sortie (optionnel)
            
        Returns:
            DataFrame preprocessé
        """
        print(f"📖 Lecture du CSV : {csv_path}")
        
        # Lecture du CSV avec gestion de différents séparateurs
        df = None
        for sep in [',', ';', '\t']:
            try:
                df = pd.read_csv(csv_path, sep=sep, encoding='utf-8')
                if len(df.columns) >= 2:
                    break
            except Exception:
                continue

        if df is None or len(df) == 0:
            raise FileNotFoundError(f"Impossible de lire le CSV : {csv_path}")

        print(f"✓ {len(df)} lignes chargées")
        
        # Détecte les noms des colonnes
        col_phrase = self._find_column(df, ['phrase', 'construction', 'texte', 'exemple'])
        col_type = self._find_column(df, ['type_jdm', 'type', 'relation', 'classe'])
        
        if not col_phrase:
            raise ValueError("Aucune colonne 'phrase' trouvée dans le CSV")
        
        print(f"📝 Preprocessing des constructions...")
        
        # Preprocess chaque ligne
        constructions = []
        for idx, row in df.iterrows():
            phrase = str(row[col_phrase])
            type_jdm = str(row[col_type]) if col_type else None
            
            construction = self.preprocess_construction(phrase, type_jdm)
            constructions.append(construction)
            
            if (idx + 1) % 50 == 0:
                print(f"  ... {idx + 1}/{len(df)} lignes traitées")
        
        # Conversion en DataFrame
        df_result = pd.DataFrame([
            {
                'phrase_originale': c.phrase_originale,
                'nom1': c.nom1,
                'nom2': c.nom2,
                'determinant': c.determinant,
                'nom1_lemme': c.nom1_lemme,
                'nom2_lemme': c.nom2_lemme,
                'definitude': c.definitude,
                'type_jdm': c.type_jdm,
                'est_valide': c.est_valide,
                'notes': c.notes
            }
            for c in constructions
        ])
        
        # Affiche les statistiques
        self._print_stats(df_result)
        
        # Sauvegarde si chemin fourni
        if output_path:
            df_result.to_csv(output_path, index=False, encoding='utf-8')
            print(f"\n💾 Résultats sauvegardés : {output_path}")
        
        return df_result
    
    def _find_column(self, df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
        """Trouve la première colonne correspondant aux candidats."""
        for col in df.columns:
            if col.lower() in [c.lower() for c in candidates]:
                return col
        return None
    
    def _print_stats(self, df: pd.DataFrame):
        """Affiche les statistiques de preprocessing."""
        print("\n" + "="*60)
        print("📊 STATISTIQUES DE PREPROCESSING")
        print("="*60)
        print(f"Total traité:              {self.stats['total_traites']}")
        print(f"Extractions réussies:      {self.stats['extractions_reussies']}")
        print(f"Mots composés détectés:    {self.stats['mots_composes_detectes']}")
        print(f"Entités remplacées:        {self.stats['entites_remplacees']}")
        print(f"\nLignes valides:            {df['est_valide'].sum()} / {len(df)}")
        print(f"Taux de réussite:          {df['est_valide'].sum() / len(df) * 100:.1f}%")
        
        if 'type_jdm' in df.columns:
            print(f"\n📈 Distribution des types:")
            print(df[df['est_valide']]['type_jdm'].value_counts())
        
        print("="*60)


def main():
    """Fonction principale pour tester le preprocessor."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocessing des constructions génitives')
    parser.add_argument('input_csv', help='Fichier CSV d\'entrée')
    parser.add_argument('-o', '--output', help='Fichier CSV de sortie')
    parser.add_argument('--no-jdm', action='store_true', help='Désactive l\'API JDM')
    
    args = parser.parse_args()
    
    # Initialise le preprocessor
    preprocessor = GenitivePreprocessor(use_jdm=not args.no_jdm)
    
    # Lance le preprocessing
    df_result = preprocessor.preprocess_csv(args.input_csv, args.output)
    
    print("\n✅ Preprocessing terminé !")
    
    # Affiche quelques exemples
    print("\n📋 Aperçu des résultats:")
    print(df_result[df_result['est_valide']].head(10))


if __name__ == '__main__':
    main()