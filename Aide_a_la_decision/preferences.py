"""
preferences.py
Génération de préférences aléatoires pour le problème du mariage stable.
	- Deux ensembles de taille n: étudiants (0..n-1) et établissements (0..n-1).
	- Chaque agent a un ordre strict complet sur l'autre ensemble.

Fonctions utilitaires pour créer les listes de préférences et leurs rangs inverses.
Usage CLI:
    python3 preferences.py --n x --seed y --out /path/to/prefs.json
Le fichier JSON contient:
	{
	  "n": n,
	  "students": {"0": [..], ...},
	  "schools": {"0": [..], ...},
	  "rank_students": {"0": {"0": int, ...}, ...},
	  "rank_schools": {"0": {"0": int, ...}, ...}
	}
"""


"--------------------------------------------------"
import argparse
import json
import random
from typing import Dict, List, Tuple

PrefList = Dict[int, List[int]]
RankMap = Dict[int, Dict[int, int]]

def random_permutations(n: int, rng: random.Random) -> List[List[int]]:
    "Retourne n permutations uniformes de 0..n-1."
    base = list(range(n))
    res = []
    for _ in range(n):
        p = base[:]  # copie
        rng.shuffle(p)
        res.append(p)
    return res
    
"--------------------------------------------------"

def generate_random_preferences(n: int, seed: int | None = None) -> Tuple[PrefList, PrefList]:
    """Génère des préférences aléatoires indépendantes et uniformes pour étudiants et établissements.
    Retourne (students_prefs, schools_prefs) où chaque est un dict id -> liste ordonnée.
    """
    if n <= 0:
        raise ValueError("n doit être > 0")
    rng = random.Random(seed)
    students_lists = random_permutations(n, rng)
    schools_lists = random_permutations(n, rng)
    students = {i: students_lists[i] for i in range(n)}
    schools = {i: schools_lists[i] for i in range(n)}
    return students, schools
    
"--------------------------------------------------"

def build_rank_map(prefs: PrefList) -> RankMap:
    "Pour chaque agent i, construit un dict item -> rang (0 = meilleur)."
    rank: RankMap = {}
    for i, lst in prefs.items():
        rank[i] = {item: r for r, item in enumerate(lst)}
    return rank

"--------------------------------------------------"

def to_json_bundle(n: int, students: PrefList, schools: PrefList) -> dict:
    "Prépare un bundle JSON sérialisable incluant les rangs inverses."
    return {
        "n": n,
        "students": {str(k): v for k, v in students.items()},
        "schools": {str(k): v for k, v in schools.items()},
        "rank_students": {str(k): {str(x): r for x, r in build_rank_map(students)[k].items()} for k in students},
        "rank_schools": {str(k): {str(x): r for x, r in build_rank_map(schools)[k].items()} for k in schools},
    }
    
"--------------------------------------------------"
"--------------------------------------------------"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, required=True, help="taille des ensembles")
    parser.add_argument("--seed", type=int, default=None, help="graine RNG pour reproductibilité")
    parser.add_argument("--out", type=str, required=True, help="chemin du fichier JSON de sortie")
    args = parser.parse_args()

    students, schools = generate_random_preferences(args.n, args.seed)
    bundle = to_json_bundle(args.n, students, schools)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(bundle, f, ensure_ascii=False, indent=2)
    print(f"Écrit: {args.out}")

if __name__ == "__main__":
    main()
    
"--------------------------------------------------"
"--------------------------------------------------"
