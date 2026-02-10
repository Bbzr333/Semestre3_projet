"""
Extracteur de features pour la classification des relations génitives.

Ce module contient:
- BasicFeatureExtractor: Features morpho-syntaxiques basiques (sans JDM)
- JDMFeatureExtractor: Features sémantiques via l'API JeuxDeMots
- EnhancedFeatureExtractor: Combinaison des deux extracteurs
"""

import pandas as pd
import numpy as np
from collections import Counter
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Any

# Ajoute le dossier parent au path pour les imports
sys.path.insert(0, str(Path(__file__).parent.parent))

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


class JDMFeatureExtractor:
    """
    Extracteur de features sémantiques via l'API JeuxDeMots.

    Implémente les features:
    - H (Hyperonymes): hyperonymes des termes
    - SST (Semantic Standard Types): types sémantiques
    - TRT (Target Relation Types): types de relations entrantes
    """

    # Catégories sémantiques (conventions JDM)
    SEMANTIC_CATEGORIES = {
        'person': {'_INFO-SEM-Personne', '_INFO-SEM-Humain', 'humain', 'personne', 'être humain'},
        'location': {'_INFO-SEM-Lieu', '_INFO-SEM-Pays', '_INFO-SEM-Ville', 'lieu', 'endroit'},
        'temporal': {'_INFO-SEM-Temps', '_INFO-SEM-Date', 'moment', 'durée', 'période'},
        'abstract': {'_INFO-SEM-Abstrait', '_INFO-SEM-Concept', 'abstraction', 'concept'},
        'concrete': {'_INFO-SEM-Concret', '_INFO-SEM-Objet', 'objet', 'chose'},
        'action': {'_INFO-SEM-Action', '_INFO-SEM-Proces', 'action', 'activité'},
        'animal': {'_INFO-SEM-Animal', 'animal', 'bête'},
        'plant': {'_INFO-SEM-Plante', 'plante', 'végétal'},
        'substance': {'_INFO-SEM-Matiere', '_INFO-SEM-Substance', 'matière', 'substance'},
    }

    # Types de relations pertinents pour la classification
    RELEVANT_RELATIONS = [
        'r_isa',           # Hyperonymie
        'r_holo',          # Holonymie (partie-tout)
        'r_has_part',      # Méronymie (tout-partie)
        'r_lieu',          # Lieu
        'r_agent',         # Agent
        'r_patient',       # Patient
        'r_carac',         # Caractéristique
        'r_domain',        # Domaine
        'r_hypo',          # Hyponymes
        'r_syn',           # Synonymes
    ]

    def __init__(self, use_jdm: bool = True, cache_signatures: bool = True):
        """
        Initialise l'extracteur JDM.

        Args:
            use_jdm: Active les appels API JDM
            cache_signatures: Cache les signatures des termes
        """
        self.use_jdm = use_jdm
        self.jdm = None
        self._signature_cache: Dict[str, Dict] = {} if cache_signatures else None

        if use_jdm:
            try:
                from utils.jdm_api import get_jdm_api
                self.jdm = get_jdm_api()
            except ImportError:
                print("Warning: JDM API non disponible, features JDM désactivées")
                self.use_jdm = False

    def _get_cached_signature(self, term: str) -> Optional[Dict]:
        """Récupère la signature d'un terme avec cache."""
        if not self.use_jdm or not self.jdm:
            return None

        if self._signature_cache is not None:
            if term not in self._signature_cache:
                try:
                    self._signature_cache[term] = self.jdm.get_signature(term)
                except Exception as e:
                    print(f"Erreur JDM pour '{term}': {e}")
                    self._signature_cache[term] = {'exists': False}
            return self._signature_cache[term]

        try:
            return self.jdm.get_signature(term)
        except Exception:
            return {'exists': False}

    def extract_jdm_features(self, row: pd.Series) -> Dict[str, float]:
        """
        Extrait les features JDM pour une ligne.

        Args:
            row: Ligne du DataFrame avec 'nom1_lemme' et 'nom2_lemme'

        Returns:
            Dict de features {nom: valeur}
        """
        features = {}

        nom1 = str(row.get('nom1_lemme', '')).lower().strip()
        nom2 = str(row.get('nom2_lemme', '')).lower().strip()

        # Récupère les signatures
        sig1 = self._get_cached_signature(nom1) if nom1 else None
        sig2 = self._get_cached_signature(nom2) if nom2 else None

        # --- Features d'existence ---
        features['nom1_exists_jdm'] = int(sig1 and sig1.get('exists', False))
        features['nom2_exists_jdm'] = int(sig2 and sig2.get('exists', False))
        features['both_exist_jdm'] = features['nom1_exists_jdm'] * features['nom2_exists_jdm']

        # --- Features d'hyperonymes (H) ---
        features.update(self._extract_hypernym_features(sig1, sig2))

        # --- Features de types sémantiques (SST) ---
        features.update(self._extract_semantic_type_features(sig1, sig2))

        # --- Features de types de relations (TRT) ---
        features.update(self._extract_relation_type_features(sig1, sig2))

        # --- Features de compatibilité croisée ---
        features.update(self._extract_compatibility_features(nom1, nom2, sig1, sig2))

        return features

    def _extract_hypernym_features(
        self,
        sig1: Optional[Dict],
        sig2: Optional[Dict]
    ) -> Dict[str, float]:
        """Extrait les features basées sur les hyperonymes."""
        features = {}

        hyp1 = set(sig1.get('hypernyms', [])) if sig1 else set()
        hyp2 = set(sig2.get('hypernyms', [])) if sig2 else set()

        # Nombre d'hyperonymes
        features['nom1_hypernym_count'] = len(hyp1)
        features['nom2_hypernym_count'] = len(hyp2)

        # Hyperonymes partagés (indicateur de similarité)
        shared = hyp1 & hyp2
        features['shared_hypernym_count'] = len(shared)
        features['has_shared_hypernym'] = int(len(shared) > 0)

        # Ratio de chevauchement
        total = len(hyp1 | hyp2)
        features['hypernym_overlap_ratio'] = len(shared) / total if total > 0 else 0.0

        return features

    def _extract_semantic_type_features(
        self,
        sig1: Optional[Dict],
        sig2: Optional[Dict]
    ) -> Dict[str, float]:
        """Extrait les features de types sémantiques (SST)."""
        features = {}

        types1 = set(sig1.get('semantic_types', [])) if sig1 else set()
        types2 = set(sig2.get('semantic_types', [])) if sig2 else set()

        # Ajoute aussi les hyperonymes comme indicateurs de type
        hyp1 = set(sig1.get('hypernyms', [])) if sig1 else set()
        hyp2 = set(sig2.get('hypernyms', [])) if sig2 else set()
        all_types1 = types1 | hyp1
        all_types2 = types2 | hyp2

        # Vérifie chaque catégorie sémantique
        for category, keywords in self.SEMANTIC_CATEGORIES.items():
            # Vérifie si nom1 appartient à la catégorie
            nom1_in_cat = int(any(
                any(kw.lower() in t.lower() for kw in keywords)
                for t in all_types1
            )) if all_types1 else 0
            features[f'nom1_is_{category}_jdm'] = nom1_in_cat

            # Vérifie si nom2 appartient à la catégorie
            nom2_in_cat = int(any(
                any(kw.lower() in t.lower() for kw in keywords)
                for t in all_types2
            )) if all_types2 else 0
            features[f'nom2_is_{category}_jdm'] = nom2_in_cat

            # Les deux dans la même catégorie
            features[f'both_are_{category}_jdm'] = nom1_in_cat * nom2_in_cat

        # Nombre total de types sémantiques
        features['nom1_semantic_type_count'] = len(types1)
        features['nom2_semantic_type_count'] = len(types2)

        return features

    def _extract_relation_type_features(
        self,
        sig1: Optional[Dict],
        sig2: Optional[Dict]
    ) -> Dict[str, float]:
        """Extrait les features de types de relations (TRT)."""
        features = {}

        rel1 = sig1.get('relation_types', {}) if sig1 else {}
        rel2 = sig2.get('relation_types', {}) if sig2 else {}

        # Compte pour chaque type de relation pertinent
        for rel_type in self.RELEVANT_RELATIONS:
            features[f'nom1_has_{rel_type}'] = int(rel_type in rel1)
            features[f'nom2_has_{rel_type}'] = int(rel_type in rel2)
            features[f'nom1_{rel_type}_count'] = rel1.get(rel_type, 0)
            features[f'nom2_{rel_type}_count'] = rel2.get(rel_type, 0)

        # Nombre total de relations entrantes
        features['nom1_total_relations'] = sum(rel1.values()) if rel1 else 0
        features['nom2_total_relations'] = sum(rel2.values()) if rel2 else 0

        return features

    def _extract_compatibility_features(
        self,
        nom1: str,
        nom2: str,
        sig1: Optional[Dict],
        sig2: Optional[Dict]
    ) -> Dict[str, float]:
        """Extrait les features de compatibilité entre les deux termes."""
        features = {}

        # Vérifie si nom2 est hyperonyme de nom1
        hyp1 = sig1.get('hypernyms', []) if sig1 else []
        features['nom2_is_hypernym_of_nom1'] = int(nom2 in [h.lower() for h in hyp1])

        # Vérifie si nom1 est hyperonyme de nom2
        hyp2 = sig2.get('hypernyms', []) if sig2 else []
        features['nom1_is_hypernym_of_nom2'] = int(nom1 in [h.lower() for h in hyp2])

        # Indicateur de relation hiérarchique
        features['has_hierarchical_relation'] = int(
            features['nom2_is_hypernym_of_nom1'] or features['nom1_is_hypernym_of_nom2']
        )

        return features

    def extract_features_batch(self, df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
        """
        Extrait les features JDM pour tout un DataFrame.

        Args:
            df: DataFrame avec colonnes 'nom1_lemme' et 'nom2_lemme'
            verbose: Affiche la progression

        Returns:
            DataFrame avec les features JDM
        """
        features_list = []
        total = len(df)

        for idx, row in df.iterrows():
            feat = self.extract_jdm_features(row)
            features_list.append(feat)

            if verbose and (idx + 1) % 100 == 0:
                print(f"  Features JDM: {idx + 1}/{total} traitées")

        if verbose:
            print(f"  Features JDM: {total}/{total} terminées")

        return pd.DataFrame(features_list)


class EnhancedFeatureExtractor:
    """
    Extracteur combiné: features basiques + features JDM.

    Maintient la rétrocompatibilité tout en ajoutant les features sémantiques.
    """

    def __init__(self, use_jdm: bool = True):
        """
        Initialise l'extracteur combiné.

        Args:
            use_jdm: Active les features JDM (sinon uniquement basiques)
        """
        self.basic_extractor = BasicFeatureExtractor()
        self.jdm_extractor = JDMFeatureExtractor(use_jdm=use_jdm) if use_jdm else None
        self.use_jdm = use_jdm

    def extract_features(self, df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
        """
        Extrait toutes les features (basiques + JDM).

        Args:
            df: DataFrame préprocessé
            verbose: Affiche la progression

        Returns:
            DataFrame avec toutes les features
        """
        if verbose:
            print("Extraction des features basiques...")
        basic_features = self.basic_extractor.extract_features(df)

        if self.use_jdm and self.jdm_extractor:
            if verbose:
                print("Extraction des features JDM sémantiques...")
            jdm_features = self.jdm_extractor.extract_features_batch(df, verbose=verbose)

            # Combine les features
            all_features = pd.concat([basic_features, jdm_features], axis=1)
        else:
            all_features = basic_features

        if verbose:
            print(f"Total features extraites: {len(all_features.columns)}")

        return all_features