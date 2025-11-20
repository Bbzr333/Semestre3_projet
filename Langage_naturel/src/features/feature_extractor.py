"""
Extracteur de features sans dépendance JDM
Features morpho-syntaxiques et sémantiques basiques
"""

import pandas as pd
import numpy as np
from collections import Counter
import re

class BasicFeatureExtractor:
    """
    Extrait des features basiques pour classification
    """
    
    def __init__(self):
        # Listes sémantiques prédéfinies (à enrichir)
        self.person_words = {'homme', 'femme', 'enfant', 'père', 'mère', 
                            'frère', 'soeur', 'ami', 'voisin', 'professeur'}
        self.location_words = {'ville', 'pays', 'région', 'quartier', 'rue',
                              'maison', 'appartement', 'bureau'}
        self.temporal_words = {'jour', 'mois', 'année', 'siècle', 'époque',
                              'moment', 'période', 'saison'}
        self.abstract_words = {'idée', 'concept', 'théorie', 'pensée', 'sentiment'}
        self.material_words = {'bois', 'métal', 'pierre', 'verre', 'plastique',
                              'or', 'argent', 'fer'}
        
    def extract_features(self, df):
        """
        Extrait toutes les features du dataframe
        """
        features = []
        
        for idx, row in df.iterrows():
            feat_dict = {}
            
            # Features morphologiques
            feat_dict.update(self._morphological_features(row))
            
            # Features lexicales
            feat_dict.update(self._lexical_features(row))
            
            # Features de longueur
            feat_dict.update(self._length_features(row))
            
            # Features de structure
            feat_dict.update(self._structural_features(row))
            
            features.append(feat_dict)
        
        return pd.DataFrame(features)
    
    def _morphological_features(self, row):
        """Features morphologiques"""
        nom1 = str(row['nom1_lemme']).lower()
        nom2 = str(row['nom2_lemme']).lower()
        
        return {
            'nom1_starts_with_vowel': int(nom1[0] in 'aeiouyàéèêë'),
            'nom2_starts_with_vowel': int(nom2[0] in 'aeiouyàéèêë'),
            'nom1_ends_with_e': int(nom1.endswith('e')),
            'nom2_ends_with_e': int(nom2.endswith('e')),
            'nom1_ends_with_s': int(nom1.endswith('s')),
            'nom2_ends_with_s': int(nom2.endswith('s')),
            'definitude': int(row['definitude']),
            'has_article_le': int('le' in str(row['phrase_originale']).lower()),
            'has_article_un': int('un' in str(row['phrase_originale']).lower()),
        }
    
    def _lexical_features(self, row):
        """Features lexicales/sémantiques basiques"""
        nom1 = str(row['nom1_lemme']).lower()
        nom2 = str(row['nom2_lemme']).lower()
        
        return {
            'nom1_is_person': int(any(w in nom1 for w in self.person_words)),
            'nom2_is_person': int(any(w in nom2 for w in self.person_words)),
            'nom1_is_location': int(any(w in nom1 for w in self.location_words)),
            'nom2_is_location': int(any(w in nom2 for w in self.location_words)),
            'nom1_is_temporal': int(any(w in nom1 for w in self.temporal_words)),
            'nom2_is_temporal': int(any(w in nom2 for w in self.temporal_words)),
            'nom1_is_abstract': int(any(w in nom1 for w in self.abstract_words)),
            'nom2_is_abstract': int(any(w in nom2 for w in self.abstract_words)),
            'nom1_is_material': int(any(w in nom1 for w in self.material_words)),
            'nom2_is_material': int(any(w in nom2 for w in self.material_words)),
        }
    
    def _length_features(self, row):
        """Features de longueur"""
        nom1 = str(row['nom1_lemme'])
        nom2 = str(row['nom2_lemme'])
        phrase = str(row['phrase_originale'])
        
        return {
            'nom1_length': len(nom1),
            'nom2_length': len(nom2),
            'phrase_length': len(phrase),
            'nom1_syllables_approx': self._count_vowels(nom1),
            'nom2_syllables_approx': self._count_vowels(nom2),
            'length_ratio': len(nom1) / max(len(nom2), 1) if len(nom2) > 0 else 0,
        }
    
    def _structural_features(self, row):
        """Features de structure de la phrase"""
        phrase = str(row['phrase_originale']).lower()
        
        return {
            'has_determinant': int(bool(re.search(r'\b(le|la|les|un|une|des|du|de la|d\')\b', phrase))),
            'word_count': len(phrase.split()),
            'has_adjective': int(bool(re.search(r'\b(grand|petit|beau|bon|nouveau|vieux)\b', phrase))),
        }
    
    @staticmethod
    def _count_vowels(word):
        """Compte approximatif des syllabes via voyelles"""
        return len(re.findall(r'[aeiouyàéèêëîïôùû]', word.lower()))