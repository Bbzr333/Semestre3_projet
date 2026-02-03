"""
Generateur de donnees d'entrainement a partir de l'API JeuxDeMots.

Ce module permet d'augmenter le corpus en extrayant des paires de mots
depuis JDM et en generant des constructions "A de B" automatiquement.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
import random

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.jdm_api import get_jdm_api, JeuxDeMotsAPI


class JDMDataGenerator:
    """Genere des donnees d'entrainement depuis JeuxDeMots."""

    # Mapping classe projet -> ID relation JDM
    CLASS_TO_JDM_RELATION = {
        'r_holo': 10,              # r_holo (partie-tout)
        'r_lieu': 15,              # r_lieu
        'r_processus_agent': 24,   # r_agent-1
        'r_processus_patient': 26, # r_patient-1
        'r_processus>instr-1': 25, # r_instr-1
        'r_has_property-1': 17,    # r_carac
        'r_topic': 3,              # r_domain
    }

    # Termes graines pour chaque classe (noms communs frequents)
    SEED_TERMS = {
        'r_holo': ['maison', 'voiture', 'arbre', 'corps', 'livre', 'table',
                   'ordinateur', 'ville', 'pays', 'animal', 'oiseau', 'fleur',
                   'montagne', 'riviere', 'foret', 'batiment', 'machine', 'avion'],
        'r_lieu': ['homme', 'femme', 'animal', 'plante', 'objet', 'personne',
                   'habitant', 'touriste', 'travailleur', 'etudiant', 'artiste'],
        'r_processus_agent': ['discours', 'decision', 'travail', 'creation',
                              'intervention', 'action', 'oeuvre', 'projet', 'film'],
        'r_processus_patient': ['destruction', 'construction', 'transformation',
                                'cuisson', 'lecture', 'ecriture', 'analyse'],
        'r_processus>instr-1': ['outil', 'instrument', 'machine', 'appareil',
                                'ustensile', 'materiel', 'equipement'],
        'r_has_property-1': ['beaute', 'force', 'intelligence', 'courage',
                             'gentillesse', 'sagesse', 'talent', 'qualite'],
        'r_topic': ['livre', 'film', 'cours', 'article', 'emission', 'conference',
                    'documentaire', 'these', 'etude', 'recherche'],
    }

    # Listes pour les classes sans mapping direct
    MATERIALS = {'bois', 'pierre', 'metal', 'verre', 'plastique', 'or', 'argent',
                 'fer', 'acier', 'cuivre', 'bronze', 'marbre', 'granit', 'beton',
                 'cuir', 'tissu', 'laine', 'coton', 'soie', 'papier', 'carton'}

    LOCATIONS = {'paris', 'france', 'lyon', 'marseille', 'bordeaux', 'toulouse',
                 'nice', 'nantes', 'strasbourg', 'montpellier', 'lille', 'rennes',
                 'italie', 'espagne', 'allemagne', 'angleterre', 'japon', 'chine',
                 'afrique', 'amerique', 'europe', 'asie', 'bretagne', 'normandie',
                 'provence', 'alsace', 'bourgogne', 'champagne', 'corse'}

    PERSONS = {'marie', 'jean', 'pierre', 'paul', 'jacques', 'philippe', 'michel',
               'andre', 'louis', 'henri', 'charles', 'francois', 'nicolas', 'antoine',
               'sophie', 'isabelle', 'catherine', 'anne', 'claire', 'julie'}

    SOCIAL_RELATIONS = {'ami', 'frere', 'soeur', 'pere', 'mere', 'oncle', 'tante',
                        'cousin', 'cousine', 'voisin', 'voisine', 'collegue',
                        'patron', 'employe', 'professeur', 'eleve', 'medecin', 'patient'}

    QUANTIFIERS = {'kilo', 'gramme', 'litre', 'metre', 'centimetre', 'tonne',
                   'tas', 'groupe', 'ensemble', 'serie', 'collection', 'lot',
                   'paquet', 'boite', 'sac', 'bouteille', 'verre', 'tasse',
                   'portion', 'part', 'morceau', 'tranche', 'bout'}

    DEPICTION_TERMS = {'portrait', 'photo', 'image', 'tableau', 'dessin',
                       'peinture', 'sculpture', 'statue', 'representation',
                       'illustration', 'gravure', 'esquisse', 'croquis'}

    CAUSE_EFFECT_TERMS = {'consequence', 'resultat', 'effet', 'impact', 'suite',
                          'retard', 'probleme', 'difficulte', 'accident', 'incident'}

    def __init__(self, min_weight: int = 10):
        """
        Initialise le generateur.

        Args:
            min_weight: Poids minimum des relations JDM a considerer
        """
        self.api = get_jdm_api()
        self.min_weight = min_weight
        self.generated_pairs: Set[Tuple[str, str, str]] = set()

    def generate_for_class(
        self,
        class_name: str,
        n_samples: int = 100,
        verbose: bool = True
    ) -> List[Dict]:
        """
        Genere des exemples pour une classe donnee.

        Args:
            class_name: Nom de la classe (ex: 'r_holo')
            n_samples: Nombre d'exemples a generer
            verbose: Affiche la progression

        Returns:
            Liste de dictionnaires {phrase, nom1, nom2, type_jdm}
        """
        if class_name in self.CLASS_TO_JDM_RELATION:
            return self._generate_from_jdm_relation(class_name, n_samples, verbose)
        else:
            return self._generate_special_class(class_name, n_samples, verbose)

    def _generate_from_jdm_relation(
        self,
        class_name: str,
        n_samples: int,
        verbose: bool
    ) -> List[Dict]:
        """Genere des exemples via les relations JDM directes."""
        relation_id = self.CLASS_TO_JDM_RELATION[class_name]
        seeds = self.SEED_TERMS.get(class_name, [])
        results = []

        if verbose:
            print(f"  Generation pour {class_name} (relation JDM {relation_id})...")

        for seed in seeds:
            if len(results) >= n_samples:
                break

            # Recupere les relations sortantes du type voulu
            response = self.api.get_outgoing_relations(
                seed,
                types_ids=str(relation_id),
                min_weight=self.min_weight,
                limit=50
            )

            if not response:
                continue

            for rel in response.relations:
                if len(results) >= n_samples:
                    break

                target_node = response.get_node_by_id(rel.node2)
                if not target_node:
                    continue

                nom1 = seed
                nom2 = target_node.name

                # Filtre les termes invalides
                if not self._is_valid_term(nom1) or not self._is_valid_term(nom2):
                    continue

                # Evite les doublons
                if (nom1, nom2, class_name) in self.generated_pairs:
                    continue

                phrase = self._generate_phrase(nom1, nom2)
                self.generated_pairs.add((nom1, nom2, class_name))

                results.append({
                    'phrase': phrase,
                    'nom1': nom1,
                    'nom2': nom2,
                    'type_jdm': class_name,
                    'source': 'jdm_api',
                    'weight': rel.weight
                })

        if verbose:
            print(f"    -> {len(results)} exemples generes")

        return results

    def _generate_special_class(
        self,
        class_name: str,
        n_samples: int,
        verbose: bool
    ) -> List[Dict]:
        """Genere des exemples pour les classes sans mapping JDM direct."""
        results = []

        if verbose:
            print(f"  Generation pour {class_name} (strategie speciale)...")

        if class_name == 'r_objet>matiere':
            results = self._generate_material_class(n_samples)
        elif class_name == 'r_lieu>origine':
            results = self._generate_origin_class(n_samples)
        elif class_name == 'r_own-1':
            results = self._generate_ownership_class(n_samples)
        elif class_name == 'r_social_tie':
            results = self._generate_social_class(n_samples)
        elif class_name == 'r_quantificateur':
            results = self._generate_quantifier_class(n_samples)
        elif class_name == 'r_depic':
            results = self._generate_depiction_class(n_samples)
        elif class_name == 'r_has_causatif':
            results = self._generate_causative_class(n_samples)
        elif class_name == 'r_product_of':
            results = self._generate_product_class(n_samples)

        if verbose:
            print(f"    -> {len(results)} exemples generes")

        return results

    def _generate_material_class(self, n_samples: int) -> List[Dict]:
        """Genere des exemples pour r_objet>matiere."""
        results = []
        objects = ['table', 'chaise', 'porte', 'fenetre', 'escalier', 'pont',
                   'statue', 'bijou', 'couteau', 'bol', 'vase', 'cadre',
                   'banc', 'armoire', 'coffre', 'bateau', 'maison', 'mur']

        for obj in objects:
            for material in random.sample(list(self.MATERIALS), min(5, len(self.MATERIALS))):
                if len(results) >= n_samples:
                    break
                if (obj, material, 'r_objet>matiere') not in self.generated_pairs:
                    phrase = self._generate_phrase(obj, material, definite=False)
                    self.generated_pairs.add((obj, material, 'r_objet>matiere'))
                    results.append({
                        'phrase': phrase,
                        'nom1': obj,
                        'nom2': material,
                        'type_jdm': 'r_objet>matiere',
                        'source': 'generated',
                        'weight': 50
                    })
            if len(results) >= n_samples:
                break

        return results[:n_samples]

    def _generate_origin_class(self, n_samples: int) -> List[Dict]:
        """Genere des exemples pour r_lieu>origine."""
        results = []
        products = ['vin', 'fromage', 'chocolat', 'parfum', 'voiture', 'montre',
                    'biere', 'cafe', 'the', 'huile', 'jambon', 'saucisson',
                    'moutarde', 'champagne', 'cognac', 'cidre', 'calvados']

        for product in products:
            for location in random.sample(list(self.LOCATIONS), min(8, len(self.LOCATIONS))):
                if len(results) >= n_samples:
                    break
                if (product, location, 'r_lieu>origine') not in self.generated_pairs:
                    phrase = self._generate_phrase(product, location, definite=False)
                    self.generated_pairs.add((product, location, 'r_lieu>origine'))
                    results.append({
                        'phrase': phrase,
                        'nom1': product,
                        'nom2': location,
                        'type_jdm': 'r_lieu>origine',
                        'source': 'generated',
                        'weight': 50
                    })
            if len(results) >= n_samples:
                break

        return results[:n_samples]

    def _generate_ownership_class(self, n_samples: int) -> List[Dict]:
        """Genere des exemples pour r_own-1 (possession)."""
        results = []
        possessions = ['livre', 'voiture', 'maison', 'chien', 'chat', 'velo',
                       'telephone', 'ordinateur', 'sac', 'montre', 'chapeau',
                       'jardin', 'bureau', 'appartement', 'bateau', 'piano']

        for possession in possessions:
            for person in random.sample(list(self.PERSONS), min(8, len(self.PERSONS))):
                if len(results) >= n_samples:
                    break
                if (possession, person, 'r_own-1') not in self.generated_pairs:
                    phrase = self._generate_phrase(possession, person, definite=False)
                    self.generated_pairs.add((possession, person, 'r_own-1'))
                    results.append({
                        'phrase': phrase,
                        'nom1': possession,
                        'nom2': person,
                        'type_jdm': 'r_own-1',
                        'source': 'generated',
                        'weight': 50
                    })
            if len(results) >= n_samples:
                break

        return results[:n_samples]

    def _generate_social_class(self, n_samples: int) -> List[Dict]:
        """Genere des exemples pour r_social_tie."""
        results = []

        for relation in self.SOCIAL_RELATIONS:
            for person in random.sample(list(self.PERSONS), min(8, len(self.PERSONS))):
                if len(results) >= n_samples:
                    break
                if (relation, person, 'r_social_tie') not in self.generated_pairs:
                    phrase = self._generate_phrase(relation, person, definite=False)
                    self.generated_pairs.add((relation, person, 'r_social_tie'))
                    results.append({
                        'phrase': phrase,
                        'nom1': relation,
                        'nom2': person,
                        'type_jdm': 'r_social_tie',
                        'source': 'generated',
                        'weight': 50
                    })
            if len(results) >= n_samples:
                break

        return results[:n_samples]

    def _generate_quantifier_class(self, n_samples: int) -> List[Dict]:
        """Genere des exemples pour r_quantificateur."""
        results = []
        countables = ['pommes', 'oranges', 'livres', 'fleurs', 'pierres',
                      'bonbons', 'billes', 'lettres', 'photos', 'pieces',
                      'farine', 'sucre', 'sel', 'eau', 'lait', 'riz', 'sable']

        for quantifier in self.QUANTIFIERS:
            for item in random.sample(countables, min(6, len(countables))):
                if len(results) >= n_samples:
                    break
                if (quantifier, item, 'r_quantificateur') not in self.generated_pairs:
                    phrase = self._generate_phrase(quantifier, item, definite=False)
                    self.generated_pairs.add((quantifier, item, 'r_quantificateur'))
                    results.append({
                        'phrase': phrase,
                        'nom1': quantifier,
                        'nom2': item,
                        'type_jdm': 'r_quantificateur',
                        'source': 'generated',
                        'weight': 50
                    })
            if len(results) >= n_samples:
                break

        return results[:n_samples]

    def _generate_depiction_class(self, n_samples: int) -> List[Dict]:
        """Genere des exemples pour r_depic."""
        results = []
        subjects = ['roi', 'reine', 'femme', 'homme', 'enfant', 'paysage',
                    'nature', 'bataille', 'scene', 'personnage', 'animal',
                    'ville', 'montagne', 'mer', 'foret', 'jardin']

        for depiction in self.DEPICTION_TERMS:
            for subject in random.sample(subjects, min(8, len(subjects))):
                if len(results) >= n_samples:
                    break
                if (depiction, subject, 'r_depic') not in self.generated_pairs:
                    phrase = self._generate_phrase(depiction, subject, definite=True)
                    self.generated_pairs.add((depiction, subject, 'r_depic'))
                    results.append({
                        'phrase': phrase,
                        'nom1': depiction,
                        'nom2': subject,
                        'type_jdm': 'r_depic',
                        'source': 'generated',
                        'weight': 50
                    })
            if len(results) >= n_samples:
                break

        return results[:n_samples]

    def _generate_causative_class(self, n_samples: int) -> List[Dict]:
        """Genere des exemples pour r_has_causatif."""
        results = []
        causes = ['pluie', 'neige', 'vent', 'orage', 'tempete', 'greve',
                  'accident', 'incendie', 'inondation', 'seisme', 'guerre',
                  'maladie', 'fatigue', 'stress', 'travail', 'chaleur', 'froid']

        for effect in self.CAUSE_EFFECT_TERMS:
            for cause in random.sample(causes, min(10, len(causes))):
                if len(results) >= n_samples:
                    break
                if (effect, cause, 'r_has_causatif') not in self.generated_pairs:
                    phrase = self._generate_phrase(effect, cause, definite=True)
                    self.generated_pairs.add((effect, cause, 'r_has_causatif'))
                    results.append({
                        'phrase': phrase,
                        'nom1': effect,
                        'nom2': cause,
                        'type_jdm': 'r_has_causatif',
                        'source': 'generated',
                        'weight': 50
                    })
            if len(results) >= n_samples:
                break

        return results[:n_samples]

    def _generate_product_class(self, n_samples: int) -> List[Dict]:
        """Genere des exemples pour r_product_of."""
        results = []
        products = ['tableau', 'livre', 'film', 'chanson', 'poeme', 'roman',
                    'sculpture', 'symphonie', 'opera', 'piece', 'oeuvre',
                    'invention', 'decouverte', 'theorie', 'recette']
        creators = ['picasso', 'mozart', 'einstein', 'hugo', 'balzac', 'monet',
                    'renoir', 'beethoven', 'bach', 'moliere', 'voltaire',
                    'rousseau', 'descartes', 'pasteur', 'curie']

        for product in products:
            for creator in random.sample(creators, min(8, len(creators))):
                if len(results) >= n_samples:
                    break
                if (product, creator, 'r_product_of') not in self.generated_pairs:
                    phrase = self._generate_phrase(product, creator, definite=False)
                    self.generated_pairs.add((product, creator, 'r_product_of'))
                    results.append({
                        'phrase': phrase,
                        'nom1': product,
                        'nom2': creator,
                        'type_jdm': 'r_product_of',
                        'source': 'generated',
                        'weight': 50
                    })
            if len(results) >= n_samples:
                break

        return results[:n_samples]

    def _generate_phrase(
        self,
        nom1: str,
        nom2: str,
        definite: Optional[bool] = None
    ) -> str:
        """
        Genere une phrase "A de B" avec le bon determinant.

        Args:
            nom1: Premier nom (tete)
            nom2: Deuxieme nom (complement)
            definite: Force la definitude (None = aleatoire)

        Returns:
            Phrase generee
        """
        nom2_lower = nom2.lower()

        # Determine si on utilise un determinant defini
        if definite is None:
            definite = random.choice([True, False])

        # Choisit le determinant
        if nom2_lower[0] in 'aeiouhàâäéèêëïîôùûü':
            det = "d'" if not definite else "de l'"
        elif definite:
            # Simplifie: utilise "du" pour masculin, "de la" pour feminin
            # (approximation car on n'a pas le genre)
            if nom2_lower.endswith('e') or nom2_lower.endswith('ie'):
                det = "de la"
            else:
                det = "du"
        else:
            det = "de"

        return f"{nom1} {det}{nom2}"

    def _is_valid_term(self, term: str) -> bool:
        """Verifie si un terme est valide pour la generation."""
        if not term or len(term) < 2:
            return False
        if term.startswith('_'):  # Types semantiques JDM
            return False
        if term.startswith('::'):  # Relations internes JDM
            return False
        if any(c in term for c in ['>', '<', ':', '|', '/']):
            return False
        return True

    def generate_all_classes(
        self,
        n_per_class: int = 100,
        verbose: bool = True
    ) -> List[Dict]:
        """
        Genere des exemples pour toutes les classes.

        Args:
            n_per_class: Nombre d'exemples par classe
            verbose: Affiche la progression

        Returns:
            Liste de tous les exemples generes
        """
        all_classes = [
            'r_holo', 'r_lieu', 'r_processus_agent', 'r_processus_patient',
            'r_processus>instr-1', 'r_has_property-1', 'r_topic', 'r_product_of',
            'r_objet>matiere', 'r_lieu>origine', 'r_own-1', 'r_social_tie',
            'r_quantificateur', 'r_depic', 'r_has_causatif'
        ]

        all_results = []

        if verbose:
            print(f"Generation de {n_per_class} exemples par classe...")
            print("=" * 60)

        for class_name in all_classes:
            results = self.generate_for_class(class_name, n_per_class, verbose)
            all_results.extend(results)

        if verbose:
            print("=" * 60)
            print(f"Total: {len(all_results)} exemples generes")

        return all_results
