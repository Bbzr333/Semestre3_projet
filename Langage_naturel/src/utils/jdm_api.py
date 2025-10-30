"""
Client API JeuxDeMots pour l'extraction de connaissances sémantiques.
Basé sur l'API REST de JeuxDeMots : http://www.jeuxdemots.org/rezo-dump.php
"""

import requests
import time
from typing import Dict, List, Optional, Set
from functools import lru_cache


class JeuxDeMotsAPI:
    """Interface pour interroger l'API JeuxDeMots."""
    
    BASE_URL = "http://www.jeuxdemots.org/rezo.php"
    
    # Types de relations JDM pertinents pour les signatures
    RELATION_TYPES = {
        'r_isa': 'hyperonyme',
        'r_hypo': 'hyponyme',
        'r_agent': 'agent',
        'r_patient': 'patient',
        'r_lieu': 'lieu',
        'r_charac': 'caractéristique',
        'r_has_part': 'partie',
        'r_holo': 'holonyme',
        'r_product_of': 'produit_de',
    }
    
    def __init__(self, cache_size: int = 1000, delay: float = 0.1):
        """
        Initialise le client API.
        
        Args:
            cache_size: Taille du cache LRU pour les requêtes
            delay: Délai entre les requêtes (rate limiting)
        """
        self.delay = delay
        self.last_request_time = 0
        self._init_cache(cache_size)
    
    def _init_cache(self, size: int):
        """Configure le cache LRU."""
        self._get_term_info = lru_cache(maxsize=size)(self._get_term_info_uncached)
    
    def _rate_limit(self):
        """Applique un délai entre les requêtes."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)
        self.last_request_time = time.time()
    
    def _get_term_info_uncached(self, term: str) -> Optional[Dict]:
        """
        Récupère les informations d'un terme depuis JDM (version non cachée).
        
        Args:
            term: Terme à rechercher
            
        Returns:
            Dictionnaire avec les relations ou None si le terme n'existe pas
        """
        self._rate_limit()
        
        # Format texte brut au lieu de HTML
        params = {
            'gotermsubmit': term,
            'rel': '',
            'output': 'text'  # Force le format texte
        }
        
        try:
            # Essaie 2 fois en cas d'échec
            for attempt in range(2):
                try:
                    response = requests.get(self.BASE_URL, params=params, timeout=30)
                    response.raise_for_status()

                    # AJOUTEZ CE DEBUG :
                    print(f"\n=== DEBUG JDM pour '{term}' ===")
                    print(f"URL: {response.url}")
                    print(f"Status: {response.status_code}")
                    print(f"Réponse (100 premiers chars): {response.text[:100]}")
                    print("="*40)

                    data = self._parse_jdm_response(response.text)
                    return data if data else None
                except requests.exceptions.Timeout:
                    if attempt == 0:
                        continue  # Retry
                    raise
                    
        except requests.exceptions.RequestException as e:
            print(f"Erreur API JDM pour '{term}': {e}")
            return None
    
    def _parse_jdm_response(self, text: str) -> Dict:
        """
        Parse la réponse texte de l'API JDM.
        
        Format JDM :
        eid|term|type|weight
        rid|node_id1|node_id2|type|weight
        
        Returns:
            Dictionnaire structuré avec les relations
        """
        lines = text.strip().split('\n')
        
        nodes = {}  # id -> term
        relations = {rel: [] for rel in self.RELATION_TYPES.keys()}
        term_info = None
        
        for line in lines:
            if not line.strip():
                continue
                
            parts = line.split('|')
            
            # Ligne de noeud (eid)
            if line.startswith('eid'):
                if len(parts) >= 4:
                    node_id = parts[0].replace('eid=', '')
                    term = parts[1].replace("'", "")  # Nettoie les quotes
                    node_type = parts[2]
                    nodes[node_id] = term
                    
                    # Premier noeud = terme recherché
                    if term_info is None:
                        term_info = {
                            'term': term,
                            'type': node_type,
                            'id': node_id
                        }
            
            # Ligne de relation (rid)
            elif line.startswith('rid'):
                if len(parts) >= 5:
                    node1_id = parts[1]
                    node2_id = parts[2]
                    rel_type = parts[3]
                    weight = int(parts[4]) if parts[4].isdigit() else 0
                    
                    if rel_type in relations:
                        if node2_id in nodes:
                            relations[rel_type].append({
                                'target': nodes[node2_id],
                                'weight': weight
                            })
        
        if term_info:
            term_info['relations'] = relations
            return term_info
        
        return {}
    
    def get_hypernyms(self, term: str, max_results: int = 10) -> List[str]:
        """
        Récupère les hyperonymes d'un terme (r_isa).
        
        Args:
            term: Terme à analyser
            max_results: Nombre max de résultats
            
        Returns:
            Liste des hyperonymes triés par poids
        """
        info = self._get_term_info(term)
        if not info:
            return []
        
        hypernyms = info.get('relations', {}).get('r_isa', [])
        # Trie par poids décroissant
        hypernyms.sort(key=lambda x: x['weight'], reverse=True)
        
        return [h['target'] for h in hypernyms[:max_results]]
    
    def get_semantic_types(self, term: str) -> Set[str]:
        """
        Récupère les types sémantiques standard (SST) d'un terme.
        Les types sont identifiés par le préfixe _INFO-SEM ou similaire.
        
        Args:
            term: Terme à analyser
            
        Returns:
            Ensemble des types sémantiques
        """
        info = self._get_term_info(term)
        if not info:
            return set()
        
        types = set()
        
        # Les types sémantiques sont souvent dans r_isa avec le préfixe _
        for rel_type, relations in info.get('relations', {}).items():
            for rel in relations:
                target = rel['target']
                if target.startswith('_'):  # Convention JDM pour types sémantiques
                    types.add(target)
        
        return types
    
    def get_incoming_relations(self, term: str) -> Dict[str, int]:
        """
        Compte les types de relations pointant vers le terme (TRT).
        
        Args:
            term: Terme à analyser
            
        Returns:
            Dictionnaire {type_relation: count}
        """
        info = self._get_term_info(term)
        if not info:
            return {}
        
        relation_counts = {}
        
        for rel_type, relations in info.get('relations', {}).items():
            if relations:
                relation_counts[rel_type] = len(relations)
        
        return relation_counts
    
    def term_exists(self, term: str) -> bool:
        """
        Vérifie si un terme existe dans JDM.
        
        Args:
            term: Terme à vérifier
            
        Returns:
            True si le terme existe
        """
        info = self._get_term_info(term)
        return info is not None
    
    def is_compound_word(self, phrase: str) -> bool:
        """
        Vérifie si une phrase est un mot composé connu dans JDM.
        
        Args:
            phrase: Expression à vérifier (ex: "lunettes de soleil")
            
        Returns:
            True si le mot composé existe dans JDM
        """
        # Remplace les espaces par underscore (convention JDM)
        compound = phrase.replace(' ', '_')
        return self.term_exists(compound)
    
    def get_signature(self, term: str) -> Dict:
        """
        Construit la signature sémantique complète d'un terme.
        Correspond aux features H, TRT, SST de GRASP-it.
        
        Args:
            term: Terme à analyser
            
        Returns:
            Dictionnaire avec la signature complète
        """
        return {
            'term': term,
            'hypernyms': self.get_hypernyms(term),
            'semantic_types': list(self.get_semantic_types(term)),
            'relation_types': self.get_incoming_relations(term),
            'exists': self.term_exists(term)
        }


# Instance globale (singleton pattern)
_jdm_instance = None

def get_jdm_api() -> JeuxDeMotsAPI:
    """Retourne l'instance singleton de l'API JDM."""
    global _jdm_instance
    if _jdm_instance is None:
        _jdm_instance = JeuxDeMotsAPI()
    return _jdm_instance