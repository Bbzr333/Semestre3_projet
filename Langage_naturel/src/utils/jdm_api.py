"""
Client API REST JeuxDeMots pour l'extraction de connaissances sémantiques.
Basé sur la nouvelle API REST : https://jdm-api.demo.lirmm.fr

Ce module fournit une interface pour interroger la base de connaissances
JeuxDeMots afin d'extraire des informations sémantiques sur les termes
français (hyperonymes, types sémantiques, relations).
"""

import requests
import time
import logging
from typing import Dict, List, Optional, Set, Any, Tuple
from functools import lru_cache
from dataclasses import dataclass


# --- Data Classes ---

@dataclass
class JDMNode:
    """Représente un noeud JDM (terme/concept)."""
    id: int
    name: str
    type: int
    weight: int
    confidence: float
    level: int

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'JDMNode':
        """Crée un JDMNode depuis un dict de l'API."""
        return cls(
            id=data.get('id', 0),
            name=data.get('name', ''),
            type=data.get('type', 0),
            weight=data.get('w', 0),
            confidence=data.get('c', 0.0),
            level=data.get('level', 0)
        )


@dataclass
class JDMRelation:
    """Représente une relation JDM entre deux noeuds."""
    id: int
    node1: int  # ID du noeud source
    node2: int  # ID du noeud cible
    type: int   # ID du type de relation
    weight: int
    confidence: float
    normalized_weight: float

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'JDMRelation':
        """Crée une JDMRelation depuis un dict de l'API."""
        return cls(
            id=data.get('id', 0),
            node1=data.get('node1', 0),
            node2=data.get('node2', 0),
            type=data.get('type', 0),
            weight=data.get('w', 0),
            confidence=data.get('c', 0.0),
            normalized_weight=data.get('nw', 0.0)
        )


@dataclass
class RelationsResponse:
    """Réponse de l'API pour les endpoints de relations."""
    nodes: List[JDMNode]
    relations: List[JDMRelation]

    def get_node_by_id(self, node_id: int) -> Optional[JDMNode]:
        """Trouve un noeud par son ID."""
        for node in self.nodes:
            if node.id == node_id:
                return node
        return None

    def get_target_names(self) -> List[Tuple[str, int]]:
        """Retourne les noms des noeuds cibles avec leur poids."""
        results = []
        for rel in self.relations:
            node = self.get_node_by_id(rel.node2)
            if node:
                results.append((node.name, rel.weight))
        return sorted(results, key=lambda x: x[1], reverse=True)


# --- Exceptions ---

class JDMAPIError(Exception):
    """Exception de base pour les erreurs de l'API JDM."""
    pass


class JDMConnectionError(JDMAPIError):
    """Erreur de connexion réseau."""
    pass


class JDMRateLimitError(JDMAPIError):
    """Limite de requêtes dépassée."""
    pass


# --- Client API Principal ---

class JeuxDeMotsAPI:
    """Client REST pour l'API JeuxDeMots."""

    BASE_URL = "https://jdm-api.demo.lirmm.fr/v0"

    # IDs des types de relations courantes (depuis l'API)
    RELATION_TYPE_IDS = {
        'r_isa': 6,           # Hyperonyme (est-un)
        'r_hypo': 8,          # Hyponyme
        'r_has_part': 9,      # A pour partie
        'r_holo': 10,         # Fait partie de (holonyme)
        'r_agent': 13,        # Agent
        'r_patient': 14,      # Patient
        'r_lieu': 15,         # Lieu
        'r_carac': 17,        # Caractéristique
        'r_syn': 5,           # Synonyme
        'r_domain': 18,       # Domaine
        'r_pos': 4,           # Partie du discours
    }

    # Mapping inverse (ID -> nom)
    RELATION_NAMES = {v: k for k, v in RELATION_TYPE_IDS.items()}

    def __init__(
        self,
        cache_size: int = 1000,
        delay: float = 0.05,
        timeout: int = 30,
        base_url: Optional[str] = None
    ):
        """
        Initialise le client API.

        Args:
            cache_size: Taille du cache LRU pour les requêtes
            delay: Délai entre les requêtes (rate limiting, en secondes)
            timeout: Timeout des requêtes HTTP (en secondes)
            base_url: URL de base (pour tests ou autre instance)
        """
        self.base_url = base_url or self.BASE_URL
        self.delay = delay
        self.timeout = timeout
        self.last_request_time = 0.0
        self._session = requests.Session()
        self._logger = logging.getLogger(__name__)

        # Initialise les caches
        self._init_cache(cache_size)

        # Métadonnées chargées à la demande
        self._relation_types: Optional[Dict[int, str]] = None
        self._node_types: Optional[Dict[int, str]] = None

    def _init_cache(self, size: int):
        """Configure les caches LRU."""
        self._get_node_cached = lru_cache(maxsize=size)(self._get_node_uncached)
        self._get_outgoing_relations_cached = lru_cache(maxsize=size)(
            self._get_outgoing_relations_uncached
        )
        self._get_incoming_relations_cached = lru_cache(maxsize=size)(
            self._get_incoming_relations_uncached
        )

    def _rate_limit(self):
        """Applique le rate limiting entre les requêtes."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)
        self.last_request_time = time.time()

    def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict] = None,
        max_retries: int = 3
    ) -> Optional[Dict]:
        """
        Effectue une requête HTTP vers l'API.

        Args:
            method: Méthode HTTP (GET, POST, etc.)
            endpoint: Chemin de l'endpoint
            params: Paramètres de la requête
            max_retries: Nombre max de tentatives

        Returns:
            Réponse JSON ou None si terme non trouvé

        Raises:
            JDMConnectionError: Erreur de connexion
            JDMRateLimitError: Limite de requêtes dépassée
        """
        self._rate_limit()
        url = f"{self.base_url}{endpoint}"

        for attempt in range(max_retries):
            try:
                response = self._session.request(
                    method=method,
                    url=url,
                    params=params,
                    timeout=self.timeout
                )

                # Terme non trouvé (404 ou 500 pour cette API) - pas une erreur
                if response.status_code in (404, 500):
                    return None

                # Rate limit (429) - attendre et réessayer
                if response.status_code == 429:
                    wait_time = float(response.headers.get('Retry-After', 5))
                    self._logger.warning(f"Rate limited, attente {wait_time}s")
                    time.sleep(wait_time)
                    continue

                response.raise_for_status()
                return response.json()

            except requests.exceptions.Timeout:
                self._logger.warning(f"Timeout pour {url}, tentative {attempt + 1}")
                if attempt == max_retries - 1:
                    raise JDMConnectionError(f"Timeout après {max_retries} tentatives")
                time.sleep(2 ** attempt)  # Backoff exponentiel

            except requests.exceptions.ConnectionError as e:
                self._logger.error(f"Erreur de connexion: {e}")
                if attempt == max_retries - 1:
                    raise JDMConnectionError(str(e))
                time.sleep(2 ** attempt)

            except requests.exceptions.HTTPError as e:
                self._logger.error(f"Erreur HTTP pour {url}: {e}")
                raise JDMAPIError(str(e))

        return None

    # --- Méthodes pour les noeuds ---

    def _get_node_uncached(self, term: str) -> Optional[JDMNode]:
        """Récupère un noeud par nom (version non cachée)."""
        data = self._request('GET', f'/node_by_name/{term}')
        if data:
            return JDMNode.from_dict(data)
        return None

    def get_node(self, term: str) -> Optional[JDMNode]:
        """
        Récupère un noeud JDM par son nom.

        Args:
            term: Terme à rechercher

        Returns:
            JDMNode ou None si le terme n'existe pas
        """
        return self._get_node_cached(term)

    def term_exists(self, term: str) -> bool:
        """
        Vérifie si un terme existe dans JDM.

        Args:
            term: Terme à vérifier

        Returns:
            True si le terme existe
        """
        return self.get_node(term) is not None

    def is_compound_word(self, phrase: str) -> bool:
        """
        Vérifie si une expression est un mot composé connu dans JDM.

        Args:
            phrase: Expression à vérifier (ex: "lunettes de soleil")

        Returns:
            True si le mot composé existe dans JDM
        """
        # Convention JDM: espaces remplacés par underscore
        compound = phrase.replace(' ', '_')
        return self.term_exists(compound)

    # --- Méthodes pour les relations ---

    def _get_outgoing_relations_uncached(
        self,
        term: str,
        types_ids: Optional[str] = None,
        min_weight: Optional[int] = None,
        limit: int = 100
    ) -> Optional[RelationsResponse]:
        """Récupère les relations sortantes (version non cachée)."""
        params = {'limit': limit}
        if types_ids:
            params['types_ids'] = types_ids
        if min_weight is not None:
            params['min_weight'] = min_weight

        data = self._request('GET', f'/relations/from/{term}', params)
        if data:
            nodes = [JDMNode.from_dict(n) for n in data.get('nodes', [])]
            relations = [JDMRelation.from_dict(r) for r in data.get('relations', [])]
            return RelationsResponse(nodes=nodes, relations=relations)
        return None

    def _get_incoming_relations_uncached(
        self,
        term: str,
        types_ids: Optional[str] = None,
        min_weight: Optional[int] = None,
        limit: int = 100
    ) -> Optional[RelationsResponse]:
        """Récupère les relations entrantes (version non cachée)."""
        params = {'limit': limit}
        if types_ids:
            params['types_ids'] = types_ids
        if min_weight is not None:
            params['min_weight'] = min_weight

        data = self._request('GET', f'/relations/to/{term}', params)
        if data:
            nodes = [JDMNode.from_dict(n) for n in data.get('nodes', [])]
            relations = [JDMRelation.from_dict(r) for r in data.get('relations', [])]
            return RelationsResponse(nodes=nodes, relations=relations)
        return None

    def get_outgoing_relations(
        self,
        term: str,
        types_ids: Optional[str] = None,
        min_weight: Optional[int] = None,
        limit: int = 100
    ) -> Optional[RelationsResponse]:
        """
        Récupère les relations sortantes d'un terme.

        Args:
            term: Terme source
            types_ids: IDs des types de relations à filtrer (ex: "6,8")
            min_weight: Poids minimum
            limit: Nombre max de résultats

        Returns:
            RelationsResponse ou None
        """
        return self._get_outgoing_relations_cached(term, types_ids, min_weight, limit)

    def get_incoming_relations(
        self,
        term: str,
        types_ids: Optional[str] = None,
        min_weight: Optional[int] = None,
        limit: int = 100
    ) -> Optional[RelationsResponse]:
        """
        Récupère les relations entrantes vers un terme.

        Args:
            term: Terme cible
            types_ids: IDs des types de relations à filtrer
            min_weight: Poids minimum
            limit: Nombre max de résultats

        Returns:
            RelationsResponse ou None
        """
        return self._get_incoming_relations_cached(term, types_ids, min_weight, limit)

    # --- Méthodes sémantiques de haut niveau ---

    def get_hypernyms(self, term: str, max_results: int = 10) -> List[str]:
        """
        Récupère les hyperonymes d'un terme (relation r_isa).

        Args:
            term: Terme à analyser
            max_results: Nombre max de résultats

        Returns:
            Liste des hyperonymes triés par poids décroissant
        """
        type_id = str(self.RELATION_TYPE_IDS['r_isa'])
        response = self.get_outgoing_relations(term, types_ids=type_id, limit=max_results * 2)

        if not response:
            return []

        results = response.get_target_names()
        return [name for name, _ in results[:max_results]]

    def get_semantic_types(self, term: str) -> Set[str]:
        """
        Récupère les types sémantiques d'un terme (termes commençant par '_').

        Args:
            term: Terme à analyser

        Returns:
            Ensemble des types sémantiques
        """
        type_id = str(self.RELATION_TYPE_IDS['r_isa'])
        response = self.get_outgoing_relations(term, types_ids=type_id, limit=50)

        if not response:
            return set()

        types = set()
        for rel in response.relations:
            node = response.get_node_by_id(rel.node2)
            if node and node.name.startswith('_'):
                types.add(node.name)

        return types

    def get_relation_type_counts(self, term: str) -> Dict[str, int]:
        """
        Compte les relations entrantes par type.

        Args:
            term: Terme à analyser

        Returns:
            Dict {nom_relation: count}
        """
        response = self.get_incoming_relations(term, limit=500)

        if not response:
            return {}

        counts: Dict[int, int] = {}
        for rel in response.relations:
            counts[rel.type] = counts.get(rel.type, 0) + 1

        # Convertit en noms lisibles
        return {
            self.RELATION_NAMES.get(t, f'type_{t}'): c
            for t, c in counts.items()
        }

    def get_signature(self, term: str) -> Dict[str, Any]:
        """
        Construit la signature sémantique complète d'un terme.
        Compatible avec l'ancienne API pour la rétrocompatibilité.

        Args:
            term: Terme à analyser

        Returns:
            Dict avec la signature complète
        """
        return {
            'term': term,
            'exists': self.term_exists(term),
            'hypernyms': self.get_hypernyms(term),
            'semantic_types': list(self.get_semantic_types(term)),
            'relation_types': self.get_relation_type_counts(term),
        }

    # --- Méthodes de métadonnées ---

    def _load_relation_types(self):
        """Charge les types de relations depuis l'API."""
        data = self._request('GET', '/relations_types')
        if data:
            self._relation_types = {
                item['id']: item['name']
                for item in data
            }
        else:
            self._relation_types = {}

    def _load_node_types(self):
        """Charge les types de noeuds depuis l'API."""
        data = self._request('GET', '/nodes_types')
        if data:
            self._node_types = {
                item['id']: item['name']
                for item in data
            }
        else:
            self._node_types = {}

    def get_relation_types(self) -> Dict[int, str]:
        """
        Retourne tous les types de relations disponibles.

        Returns:
            Dict {id: nom}
        """
        if self._relation_types is None:
            self._load_relation_types()
        return self._relation_types or {}

    def get_node_types(self) -> Dict[int, str]:
        """
        Retourne tous les types de noeuds disponibles.

        Returns:
            Dict {id: nom}
        """
        if self._node_types is None:
            self._load_node_types()
        return self._node_types or {}

    def clear_cache(self):
        """Vide tous les caches."""
        self._get_node_cached.cache_clear()
        self._get_outgoing_relations_cached.cache_clear()
        self._get_incoming_relations_cached.cache_clear()


# --- Singleton Pattern (rétrocompatibilité) ---

_jdm_instance: Optional[JeuxDeMotsAPI] = None

def get_jdm_api() -> JeuxDeMotsAPI:
    """Retourne l'instance singleton du client API JDM."""
    global _jdm_instance
    if _jdm_instance is None:
        _jdm_instance = JeuxDeMotsAPI()
    return _jdm_instance
