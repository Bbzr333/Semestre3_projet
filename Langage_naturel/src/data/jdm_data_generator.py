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

    # Termes ambigus qui peuvent appartenir à plusieurs classes
    AMBIGUOUS_TERMS = {
        'peinture': ['r_processus_patient', 'r_depic', 'r_objet>matiere'],  # action, tableau, ou matière
        'carte': ['r_depic', 'r_lieu', 'r_topic'],  # représentation, lieu, ou sujet
        'sculpture': ['r_processus_patient', 'r_depic', 'r_product_of'],  # action, objet, ou création
        'photo': ['r_depic', 'r_processus_patient', 'r_own-1'],  # image, action, ou possession
        'construction': ['r_processus_patient', 'r_holo', 'r_lieu'],  # action, partie, ou lieu
        'histoire': ['r_topic', 'r_processus_patient', 'r_product_of'],  # sujet, récit, ou création
        'portrait': ['r_depic', 'r_product_of', 'r_own-1'],  # représentation, création, possession
        'dessin': ['r_depic', 'r_processus_patient', 'r_product_of'],
        'tableau': ['r_depic', 'r_product_of', 'r_own-1', 'r_objet>matiere'],
        'ecriture': ['r_processus_patient', 'r_topic', 'r_has_property-1'],
    }

    # Patterns de paraphrase pour diversité syntaxique
    PARAPHRASE_PATTERNS = [
        lambda n1, n2, det: f"{det}{n1} de {n2}",
        lambda n1, n2, det: f"{det}{n1} du {n2}" if not n2[0] in 'aeiouh' else f"{det}{n1} de l'{n2}",
        lambda n1, n2, det: f"ce {n1} de {n2}",
        lambda n1, n2, det: f"un certain {n1} de {n2}",
        lambda n1, n2, det: f"tout {n1} de {n2}",
        lambda n1, n2, det: f"chaque {n1} de {n2}",
        lambda n1, n2, det: f"quel {n1} de {n2}",
    ]

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

    # Listes enrichies pour les classes sans mapping direct
    MATERIALS = {'bois', 'pierre', 'metal', 'verre', 'plastique', 'or', 'argent',
                 'fer', 'acier', 'cuivre', 'bronze', 'marbre', 'granit', 'beton',
                 'cuir', 'tissu', 'laine', 'coton', 'soie', 'papier', 'carton',
                 'aluminium', 'platine', 'titane', 'zinc', 'etain', 'plomb',
                 'cristal', 'porcelaine', 'ceramique', 'terre cuite', 'ardoise',
                 'caoutchouc', 'resine', 'fibre', 'bambou', 'osier', 'rotin',
                 'ivoire', 'nacre', 'corne', 'email', 'faience', 'gres',
                 'velours', 'satin', 'lin', 'chanvre', 'jute', 'nylon', 'polyester'}

    LOCATIONS = {'paris', 'france', 'lyon', 'marseille', 'bordeaux', 'toulouse',
                 'nice', 'nantes', 'strasbourg', 'montpellier', 'lille', 'rennes',
                 'italie', 'espagne', 'allemagne', 'angleterre', 'japon', 'chine',
                 'afrique', 'amerique', 'europe', 'asie', 'bretagne', 'normandie',
                 'provence', 'alsace', 'bourgogne', 'champagne', 'corse',
                 'suisse', 'belgique', 'canada', 'mexique', 'bresil', 'argentine',
                 'grece', 'turquie', 'inde', 'russie', 'australie', 'egypte',
                 'maroc', 'tunisie', 'senegal', 'vietnam', 'thailande', 'coree',
                 'perigord', 'savoie', 'vendee', 'auvergne', 'aquitaine', 'lorraine',
                 'touraine', 'gascogne', 'languedoc', 'roussillon', 'limousin',
                 'londres', 'rome', 'berlin', 'madrid', 'lisbonne', 'vienne',
                 'amsterdam', 'bruxelles', 'geneve', 'zurich', 'milan', 'florence'}

    PERSONS = {'marie', 'jean', 'pierre', 'paul', 'jacques', 'philippe', 'michel',
               'andre', 'louis', 'henri', 'charles', 'francois', 'nicolas', 'antoine',
               'sophie', 'isabelle', 'catherine', 'anne', 'claire', 'julie',
               'thomas', 'marc', 'luc', 'mathieu', 'david', 'alexandre', 'olivier',
               'vincent', 'sylvain', 'bruno', 'alain', 'bernard', 'claude', 'daniel',
               'eric', 'fabrice', 'guillaume', 'julien', 'laurent', 'pascal',
               'nathalie', 'sandrine', 'valerie', 'christine', 'martine', 'monique',
               'laurence', 'veronique', 'patricia', 'sylvie', 'brigitte', 'helene',
               'emma', 'lea', 'chloe', 'camille', 'lucas', 'hugo', 'nathan', 'theo',
               'artiste', 'auteur', 'ecrivain', 'peintre', 'musicien', 'sculpteur'}

    SOCIAL_RELATIONS = {'ami', 'frere', 'soeur', 'pere', 'mere', 'oncle', 'tante',
                        'cousin', 'cousine', 'voisin', 'voisine', 'collegue',
                        'patron', 'employe', 'professeur', 'eleve', 'medecin', 'patient',
                        'grand-pere', 'grand-mere', 'neveu', 'niece', 'beau-frere',
                        'belle-soeur', 'gendre', 'belle-fille', 'parrain', 'marraine',
                        'filleul', 'filleule', 'compagnon', 'compagne', 'epoux', 'epouse',
                        'associe', 'partenaire', 'collaborateur', 'superieur', 'subalterne',
                        'mentor', 'tuteur', 'eleve', 'disciple', 'maitre', 'apprenti',
                        'client', 'fournisseur', 'avocat', 'notaire', 'banquier',
                        'colocataire', 'camarade', 'condisciple', 'rival', 'adversaire'}

    QUANTIFIERS = {'kilo', 'gramme', 'litre', 'metre', 'centimetre', 'tonne',
                   'tas', 'groupe', 'ensemble', 'serie', 'collection', 'lot',
                   'paquet', 'boite', 'sac', 'bouteille', 'verre', 'tasse',
                   'portion', 'part', 'morceau', 'tranche', 'bout',
                   'pincee', 'poignee', 'brassee', 'goutte', 'filet', 'nuage',
                   'cuillere', 'assiette', 'bol', 'corbeille', 'caisse', 'carton',
                   'douzaine', 'centaine', 'millier', 'million', 'milliard',
                   'moitie', 'tiers', 'quart', 'cinquieme', 'dixieme',
                   'pile', 'amas', 'monticule', 'foule', 'multitude', 'masse',
                   'bande', 'equipe', 'troupe', 'escouade', 'escadron', 'regiment'}

    DEPICTION_TERMS = {'portrait', 'photo', 'image', 'tableau', 'dessin',
                       'peinture', 'sculpture', 'statue', 'representation',
                       'illustration', 'gravure', 'esquisse', 'croquis',
                       'photographie', 'cliche', 'instantane', 'selfie',
                       'fresque', 'mosaique', 'vitrail', 'tapisserie',
                       'caricature', 'silhouette', 'effigie', 'buste', 'masque',
                       'figurine', 'maquette', 'modele', 'reproduction', 'copie',
                       'affiche', 'poster', 'carte postale', 'vignette', 'miniature'}

    CAUSE_EFFECT_TERMS = {'consequence', 'resultat', 'effet', 'impact', 'suite',
                          'retard', 'probleme', 'difficulte', 'accident', 'incident',
                          'dommage', 'degat', 'destruction', 'perte', 'prejudice',
                          'benefice', 'avantage', 'progres', 'amelioration', 'succes',
                          'echec', 'fiasco', 'desastre', 'catastrophe', 'crise',
                          'sequelle', 'repercussion', 'contrecoup', 'reaction', 'reponse',
                          'symptome', 'signe', 'manifestation', 'expression', 'marque'}

    # Objets supplementaires pour plus de diversite
    OBJECTS_EXTENDED = {'table', 'chaise', 'porte', 'fenetre', 'escalier', 'pont',
                        'statue', 'bijou', 'couteau', 'bol', 'vase', 'cadre',
                        'banc', 'armoire', 'coffre', 'bateau', 'maison', 'mur',
                        'lit', 'bureau', 'etagere', 'commode', 'buffet', 'bahut',
                        'lampe', 'lustre', 'miroir', 'horloge', 'pendule', 'tableau',
                        'tapis', 'rideau', 'coussin', 'couverture', 'drap', 'oreiller',
                        'assiette', 'plat', 'casserole', 'poele', 'marmite', 'theiere',
                        'fourchette', 'cuillere', 'louche', 'spatule', 'rape', 'passoire',
                        'bracelet', 'collier', 'bague', 'boucle', 'pendentif', 'broche',
                        'violon', 'piano', 'guitare', 'flute', 'harpe', 'orgue'}

    def __init__(self, min_weight: int = 10, enrich_with_jdm: bool = True):
        """
        Initialise le generateur.

        Args:
            min_weight: Poids minimum des relations JDM a considerer
            enrich_with_jdm: Enrichit les listes avec synonymes/hyponymes JDM
        """
        self.api = get_jdm_api()
        self.min_weight = min_weight
        self.generated_pairs: Set[Tuple[str, str, str]] = set()
        self.enrich_with_jdm = enrich_with_jdm
        self._enriched_cache: Dict[str, Set[str]] = {}

    def _enrich_terms_jdm(self, terms: Set[str], max_additions: int = 50) -> Set[str]:
        """
        Enrichit une liste de termes avec synonymes et hyponymes via JDM.

        Args:
            terms: Set de termes a enrichir
            max_additions: Nombre max de termes a ajouter

        Returns:
            Set enrichi de termes
        """
        if not self.enrich_with_jdm:
            return terms

        cache_key = frozenset(terms)
        if cache_key in self._enriched_cache:
            return self._enriched_cache[cache_key]

        enriched = set(terms)
        additions = 0

        for term in list(terms)[:20]:  # Limite pour eviter trop d'appels API
            if additions >= max_additions:
                break

            try:
                # Recupere synonymes (r_syn = relation type 5)
                syn_response = self.api.get_outgoing_relations(
                    term, types_ids='5', min_weight=5, limit=10
                )
                if syn_response:
                    for rel in syn_response.relations:
                        node = syn_response.get_node_by_id(rel.node2)
                        if node and self._is_valid_term(node.name):
                            enriched.add(node.name.lower())
                            additions += 1
                            if additions >= max_additions:
                                break

                # Recupere hyponymes (r_hypo = relation type 8)
                hypo_response = self.api.get_outgoing_relations(
                    term, types_ids='8', min_weight=5, limit=10
                )
                if hypo_response:
                    for rel in hypo_response.relations:
                        node = hypo_response.get_node_by_id(rel.node2)
                        if node and self._is_valid_term(node.name):
                            enriched.add(node.name.lower())
                            additions += 1
                            if additions >= max_additions:
                                break

            except Exception:
                continue

        self._enriched_cache[cache_key] = enriched
        return enriched

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

        # Utilise la liste etendue d'objets
        objects = list(self.OBJECTS_EXTENDED)
        random.shuffle(objects)

        # Enrichit les materiaux avec JDM
        materials = self._enrich_terms_jdm(self.MATERIALS, max_additions=30)
        materials_list = list(materials)
        random.shuffle(materials_list)

        for obj in objects:
            for material in materials_list[:15]:  # Plus de materiaux par objet
                if len(results) >= n_samples:
                    break

                # Genere plusieurs variations pour chaque paire
                variations = self._generate_phrase_variations(
                    obj, material, 'r_objet>matiere', max_variations=2
                )
                results.extend(variations)

            if len(results) >= n_samples:
                break

        return results[:n_samples]

    def _generate_origin_class(self, n_samples: int) -> List[Dict]:
        """Genere des exemples pour r_lieu>origine."""
        results = []
        products = ['vin', 'fromage', 'chocolat', 'parfum', 'voiture', 'montre',
                    'biere', 'cafe', 'the', 'huile', 'jambon', 'saucisson',
                    'moutarde', 'champagne', 'cognac', 'cidre', 'calvados',
                    'whisky', 'vodka', 'rhum', 'tequila', 'sake', 'porto',
                    'dentelle', 'porcelaine', 'soie', 'tapis', 'epice', 'safran',
                    'olive', 'orange', 'citron', 'banane', 'mangue', 'ananas']

        # Enrichit les lieux
        locations = self._enrich_terms_jdm(self.LOCATIONS, max_additions=40)
        locations_list = list(locations)
        random.shuffle(locations_list)

        for product in products:
            for location in locations_list[:15]:
                if len(results) >= n_samples:
                    break

                variations = self._generate_phrase_variations(
                    product, location, 'r_lieu>origine', max_variations=2
                )
                results.extend(variations)

            if len(results) >= n_samples:
                break

        return results[:n_samples]

    def _generate_ownership_class(self, n_samples: int) -> List[Dict]:
        """Genere des exemples pour r_own-1 (possession)."""
        results = []
        possessions = ['livre', 'voiture', 'maison', 'chien', 'chat', 'velo',
                       'telephone', 'ordinateur', 'sac', 'montre', 'chapeau',
                       'jardin', 'bureau', 'appartement', 'bateau', 'piano',
                       'moto', 'camion', 'tracteur', 'yacht', 'avion', 'villa',
                       'chalet', 'studio', 'loft', 'ferme', 'domaine', 'chateau',
                       'camera', 'tablette', 'console', 'imprimante', 'robot',
                       'perroquet', 'hamster', 'lapin', 'cheval', 'poney', 'tortue']

        # Enrichit les personnes
        persons = self._enrich_terms_jdm(self.PERSONS, max_additions=30)
        persons_list = list(persons)
        random.shuffle(persons_list)

        for possession in possessions:
            for person in persons_list[:12]:
                if len(results) >= n_samples:
                    break

                variations = self._generate_phrase_variations(
                    possession, person, 'r_own-1', max_variations=2
                )
                results.extend(variations)

            if len(results) >= n_samples:
                break

        return results[:n_samples]

    def _generate_social_class(self, n_samples: int) -> List[Dict]:
        """Genere des exemples pour r_social_tie."""
        results = []

        # Enrichit les relations sociales et les personnes
        relations = self._enrich_terms_jdm(self.SOCIAL_RELATIONS, max_additions=20)
        relations_list = list(relations)
        random.shuffle(relations_list)

        persons = self._enrich_terms_jdm(self.PERSONS, max_additions=30)
        persons_list = list(persons)
        random.shuffle(persons_list)

        for relation in relations_list:
            for person in persons_list[:10]:
                if len(results) >= n_samples:
                    break

                variations = self._generate_phrase_variations(
                    relation, person, 'r_social_tie', max_variations=2
                )
                results.extend(variations)

            if len(results) >= n_samples:
                break

        return results[:n_samples]

    def _generate_quantifier_class(self, n_samples: int) -> List[Dict]:
        """Genere des exemples pour r_quantificateur."""
        results = []
        countables = ['pommes', 'oranges', 'livres', 'fleurs', 'pierres',
                      'bonbons', 'billes', 'lettres', 'photos', 'pieces',
                      'farine', 'sucre', 'sel', 'eau', 'lait', 'riz', 'sable',
                      'tomates', 'cerises', 'fraises', 'raisins', 'noix', 'amandes',
                      'chocolat', 'cafe', 'the', 'vin', 'biere', 'jus',
                      'pain', 'gateau', 'biscuit', 'croissant', 'brioche',
                      'viande', 'poisson', 'poulet', 'boeuf', 'porc', 'agneau',
                      'legumes', 'fruits', 'cereales', 'pates', 'soupe', 'salade',
                      'argent', 'or', 'monnaie', 'billets', 'cartes', 'jetons']

        # Enrichit les quantificateurs
        quantifiers = self._enrich_terms_jdm(self.QUANTIFIERS, max_additions=20)
        quantifiers_list = list(quantifiers)
        random.shuffle(quantifiers_list)

        for quantifier in quantifiers_list:
            for item in random.sample(countables, min(10, len(countables))):
                if len(results) >= n_samples:
                    break

                variations = self._generate_phrase_variations(
                    quantifier, item, 'r_quantificateur', max_variations=2
                )
                results.extend(variations)

            if len(results) >= n_samples:
                break

        return results[:n_samples]

    def _generate_depiction_class(self, n_samples: int) -> List[Dict]:
        """Genere des exemples pour r_depic."""
        results = []
        subjects = ['roi', 'reine', 'femme', 'homme', 'enfant', 'paysage',
                    'nature', 'bataille', 'scene', 'personnage', 'animal',
                    'ville', 'montagne', 'mer', 'foret', 'jardin',
                    'saint', 'ange', 'demon', 'dieu', 'deesse', 'heros',
                    'chevalier', 'soldat', 'paysan', 'noble', 'marchand',
                    'coucher de soleil', 'lever de soleil', 'tempete', 'orage',
                    'printemps', 'ete', 'automne', 'hiver', 'nuit', 'jour',
                    'chasse', 'peche', 'danse', 'fete', 'mariage', 'funerailles',
                    'christ', 'vierge', 'napoleon', 'cesar', 'alexandre']

        # Enrichit les termes de depiction
        depictions = self._enrich_terms_jdm(self.DEPICTION_TERMS, max_additions=15)
        depictions_list = list(depictions)
        random.shuffle(depictions_list)

        for depiction in depictions_list:
            for subject in random.sample(subjects, min(12, len(subjects))):
                if len(results) >= n_samples:
                    break

                variations = self._generate_phrase_variations(
                    depiction, subject, 'r_depic', max_variations=2
                )
                results.extend(variations)

            if len(results) >= n_samples:
                break

        return results[:n_samples]

    def _generate_causative_class(self, n_samples: int) -> List[Dict]:
        """Genere des exemples pour r_has_causatif."""
        results = []
        causes = ['pluie', 'neige', 'vent', 'orage', 'tempete', 'greve',
                  'accident', 'incendie', 'inondation', 'seisme', 'guerre',
                  'maladie', 'fatigue', 'stress', 'travail', 'chaleur', 'froid',
                  'secheresse', 'gel', 'verglas', 'brouillard', 'tsunami', 'cyclone',
                  'epidemie', 'pandemie', 'famine', 'pollution', 'deforestation',
                  'chomage', 'inflation', 'recession', 'faillite', 'scandale',
                  'corruption', 'negligence', 'erreur', 'incompetence', 'sabotage',
                  'explosion', 'effondrement', 'collision', 'naufrage', 'crash',
                  'surmenage', 'depression', 'anxiete', 'insomnie', 'addiction']

        # Enrichit les termes d'effet
        effects = self._enrich_terms_jdm(self.CAUSE_EFFECT_TERMS, max_additions=20)
        effects_list = list(effects)
        random.shuffle(effects_list)

        for effect in effects_list:
            for cause in random.sample(causes, min(12, len(causes))):
                if len(results) >= n_samples:
                    break

                variations = self._generate_phrase_variations(
                    effect, cause, 'r_has_causatif', max_variations=2
                )
                results.extend(variations)

            if len(results) >= n_samples:
                break

        return results[:n_samples]

    def _generate_product_class(self, n_samples: int) -> List[Dict]:
        """Genere des exemples pour r_product_of."""
        results = []
        products = ['tableau', 'livre', 'film', 'chanson', 'poeme', 'roman',
                    'sculpture', 'symphonie', 'opera', 'piece', 'oeuvre',
                    'invention', 'decouverte', 'theorie', 'recette',
                    'sonate', 'concerto', 'quatuor', 'requiem', 'messe', 'cantate',
                    'nouvelle', 'conte', 'fable', 'essai', 'memoire', 'these',
                    'brevet', 'formule', 'equation', 'theoreme', 'loi', 'principe',
                    'album', 'single', 'clip', 'spectacle', 'ballet', 'comedie',
                    'tragedie', 'drame', 'farce', 'sketch', 'monologue', 'dialogue',
                    'fresque', 'portrait', 'nature morte', 'paysage', 'autoportrait']
        creators = ['picasso', 'mozart', 'einstein', 'hugo', 'balzac', 'monet',
                    'renoir', 'beethoven', 'bach', 'moliere', 'voltaire',
                    'rousseau', 'descartes', 'pasteur', 'curie',
                    'rembrandt', 'vermeer', 'dali', 'warhol', 'kandinsky', 'manet',
                    'cezanne', 'gauguin', 'van gogh', 'klimt', 'munch', 'caravage',
                    'chopin', 'liszt', 'brahms', 'schubert', 'vivaldi', 'handel',
                    'zola', 'flaubert', 'proust', 'camus', 'sartre', 'baudelaire',
                    'rimbaud', 'verlaine', 'mallarme', 'apollinaire', 'prevert',
                    'edison', 'tesla', 'darwin', 'newton', 'galilee', 'lavoisier',
                    'spielberg', 'hitchcock', 'kubrick', 'godard', 'truffaut']

        for product in products:
            for creator in random.sample(creators, min(15, len(creators))):
                if len(results) >= n_samples:
                    break

                variations = self._generate_phrase_variations(
                    product, creator, 'r_product_of', max_variations=2
                )
                results.extend(variations)

            if len(results) >= n_samples:
                break

        return results[:n_samples]

    def _generate_phrase(
        self,
        nom1: str,
        nom2: str,
        definite: Optional[bool] = None,
        add_article_nom1: bool = True
    ) -> str:
        """
        Genere une phrase "A de B" avec le bon determinant.

        Args:
            nom1: Premier nom (tete)
            nom2: Deuxieme nom (complement)
            definite: Force la definitude (None = aleatoire)
            add_article_nom1: Ajoute un article devant nom1

        Returns:
            Phrase generee
        """
        nom2_lower = nom2.lower()

        # Determine si on utilise un determinant defini
        if definite is None:
            definite = random.choice([True, False])

        # Choisit le determinant pour nom2
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

        # Ajoute optionnellement un article devant nom1 pour plus de diversite
        if add_article_nom1:
            nom1_lower = nom1.lower()
            article_choices = ['le', 'la', 'un', 'une', 'les', 'des']
            # Choix intelligent base sur la terminaison
            if nom1_lower[0] in 'aeiouhàâäéèêëïîôùûü':
                article = random.choice(["l'", "un ", "une "])
            elif nom1_lower.endswith('s') or nom1_lower.endswith('x'):
                article = random.choice(['les ', 'des '])
            elif nom1_lower.endswith('e') or nom1_lower.endswith('ie') or nom1_lower.endswith('ee'):
                article = random.choice(['la ', 'une '])
            else:
                article = random.choice(['le ', 'un '])
            return f"{article}{nom1} {det}{nom2}"

        return f"{nom1} {det}{nom2}"

    def _generate_phrase_variations(
        self,
        nom1: str,
        nom2: str,
        class_name: str,
        max_variations: int = 3
    ) -> List[Dict]:
        """
        Genere plusieurs variations d'une meme paire de termes.

        Returns:
            Liste de dictionnaires avec les variations
        """
        variations = []

        # Variation 1: avec article defini
        phrase1 = self._generate_phrase(nom1, nom2, definite=True)
        if (nom1, nom2, class_name) not in self.generated_pairs:
            self.generated_pairs.add((nom1, nom2, class_name))
            variations.append({
                'phrase': phrase1,
                'nom1': nom1,
                'nom2': nom2,
                'type_jdm': class_name,
                'source': 'generated_variation',
                'weight': 50
            })

        if len(variations) >= max_variations:
            return variations

        # Variation 2: avec article indefini
        phrase2 = self._generate_phrase(nom1, nom2, definite=False)
        key2 = (nom1 + "_indef", nom2, class_name)
        if key2 not in self.generated_pairs and phrase2 != phrase1:
            self.generated_pairs.add(key2)
            variations.append({
                'phrase': phrase2,
                'nom1': nom1,
                'nom2': nom2,
                'type_jdm': class_name,
                'source': 'generated_variation',
                'weight': 45
            })

        if len(variations) >= max_variations:
            return variations

        # Variation 3: pluriel de nom1 si applicable
        if not nom1.endswith('s') and not nom1.endswith('x'):
            nom1_plural = nom1 + 's'
            phrase3 = self._generate_phrase(nom1_plural, nom2, definite=True)
            key3 = (nom1_plural, nom2, class_name)
            if key3 not in self.generated_pairs:
                self.generated_pairs.add(key3)
                variations.append({
                    'phrase': phrase3,
                    'nom1': nom1_plural,
                    'nom2': nom2,
                    'type_jdm': class_name,
                    'source': 'generated_variation',
                    'weight': 40
                })

        return variations

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

    def generate_ambiguous_examples(
        self,
        n_samples: int = 100,
        verbose: bool = True
    ) -> List[Dict]:
        """
        Genere des exemples ambigus pour tester la robustesse des modeles.
        Ces exemples utilisent des termes qui peuvent appartenir a plusieurs classes.

        Args:
            n_samples: Nombre d'exemples a generer
            verbose: Affiche la progression

        Returns:
            Liste de dictionnaires avec les exemples ambigus
        """
        results = []

        if verbose:
            print(f"  Generation de {n_samples} exemples ambigus...")

        ambiguous_items = list(self.AMBIGUOUS_TERMS.items())
        random.shuffle(ambiguous_items)

        # Complements varies pour chaque terme ambigu
        complements = {
            'personne': list(self.PERSONS)[:20],
            'lieu': list(self.LOCATIONS)[:20],
            'objet': list(self.OBJECTS_EXTENDED)[:20],
            'abstrait': ['liberte', 'justice', 'amour', 'guerre', 'paix', 'vie', 'mort',
                        'beaute', 'verite', 'nature', 'temps', 'espace', 'histoire'],
        }

        for term, possible_classes in ambiguous_items:
            if len(results) >= n_samples:
                break

            # Choisit une classe aleatoire parmi les possibles
            chosen_class = random.choice(possible_classes)

            # Choisit un complement approprie
            if chosen_class in ['r_own-1', 'r_product_of', 'r_social_tie']:
                complement_list = complements['personne']
            elif chosen_class in ['r_lieu', 'r_lieu>origine']:
                complement_list = complements['lieu']
            elif chosen_class in ['r_objet>matiere', 'r_holo']:
                complement_list = complements['objet']
            else:
                complement_list = complements['abstrait'] + complements['personne']

            for complement in random.sample(complement_list, min(5, len(complement_list))):
                if len(results) >= n_samples:
                    break

                phrase = self._generate_phrase(term, complement)

                # Marque comme ambigu
                results.append({
                    'phrase': phrase,
                    'nom1': term,
                    'nom2': complement,
                    'type_jdm': chosen_class,
                    'source': 'ambiguous',
                    'weight': 30,
                    'is_ambiguous': True,
                    'possible_classes': possible_classes
                })

        if verbose:
            print(f"    -> {len(results)} exemples ambigus generes")

        return results

    def add_noise_to_data(
        self,
        data: List[Dict],
        noise_ratio: float = 0.1,
        verbose: bool = True
    ) -> List[Dict]:
        """
        Ajoute du bruit aux donnees pour ameliorer la robustesse.

        Types de bruit:
        - Fautes de frappe simulees
        - Variations orthographiques
        - Ajout/suppression d'accents

        Args:
            data: Liste de dictionnaires d'exemples
            noise_ratio: Proportion d'exemples a bruiter (0.0 - 1.0)
            verbose: Affiche la progression

        Returns:
            Donnees avec bruit ajoute
        """
        if verbose:
            print(f"  Ajout de bruit ({noise_ratio*100:.0f}% des donnees)...")

        noisy_data = []
        n_to_noise = int(len(data) * noise_ratio)
        indices_to_noise = set(random.sample(range(len(data)), n_to_noise))

        # Mappings pour le bruit
        accent_variations = {
            'e': ['e', 'é', 'è', 'ê'],
            'a': ['a', 'à', 'â'],
            'u': ['u', 'ù', 'û'],
            'i': ['i', 'î', 'ï'],
            'o': ['o', 'ô'],
            'c': ['c', 'ç'],
        }

        for i, item in enumerate(data):
            if i in indices_to_noise:
                noisy_item = item.copy()
                phrase = item['phrase']

                # Type de bruit aleatoire
                noise_type = random.choice(['accent', 'typo', 'spacing'])

                if noise_type == 'accent':
                    # Change aleatoirement un accent
                    for char, variants in accent_variations.items():
                        if char in phrase.lower():
                            phrase = phrase.replace(char, random.choice(variants), 1)
                            break

                elif noise_type == 'typo':
                    # Supprime ou duplique une lettre
                    if len(phrase) > 10:
                        pos = random.randint(3, len(phrase) - 3)
                        if random.random() < 0.5:
                            phrase = phrase[:pos] + phrase[pos+1:]  # Suppression
                        else:
                            phrase = phrase[:pos] + phrase[pos] + phrase[pos:]  # Duplication

                elif noise_type == 'spacing':
                    # Ajoute ou supprime un espace
                    if '  ' not in phrase and random.random() < 0.5:
                        pos = phrase.find(' ')
                        if pos > 0:
                            phrase = phrase[:pos] + '  ' + phrase[pos+1:]

                noisy_item['phrase'] = phrase
                noisy_item['has_noise'] = True
                noisy_data.append(noisy_item)
            else:
                noisy_data.append(item)

        if verbose:
            print(f"    -> {n_to_noise} exemples bruites")

        return noisy_data

    def generate_paraphrases(
        self,
        data: List[Dict],
        n_paraphrases: int = 2,
        verbose: bool = True
    ) -> List[Dict]:
        """
        Genere des paraphrases syntaxiques pour augmenter la diversite.

        Args:
            data: Liste de dictionnaires d'exemples
            n_paraphrases: Nombre de paraphrases par exemple
            verbose: Affiche la progression

        Returns:
            Donnees originales + paraphrases
        """
        if verbose:
            print(f"  Generation de paraphrases ({n_paraphrases} par exemple)...")

        augmented_data = list(data)  # Garde les originaux
        patterns = self.PARAPHRASE_PATTERNS

        for item in data:
            nom1 = item.get('nom1', '')
            nom2 = item.get('nom2', '')

            if not nom1 or not nom2:
                continue

            # Genere n paraphrases avec des patterns differents
            selected_patterns = random.sample(patterns, min(n_paraphrases, len(patterns)))

            for pattern in selected_patterns:
                try:
                    det = random.choice(['le ', 'la ', 'un ', 'une ', ''])
                    new_phrase = pattern(nom1, nom2, det)

                    # Evite les doublons exacts
                    if new_phrase != item['phrase']:
                        new_item = item.copy()
                        new_item['phrase'] = new_phrase
                        new_item['source'] = 'paraphrase'
                        new_item['original_phrase'] = item['phrase']
                        augmented_data.append(new_item)
                except Exception:
                    continue

        if verbose:
            print(f"    -> {len(augmented_data) - len(data)} paraphrases generees")
            print(f"    -> Total: {len(augmented_data)} exemples")

        return augmented_data

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
