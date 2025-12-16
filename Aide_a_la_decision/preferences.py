
import argparse, json, random
from typing import Dict, List

"--------------------------------------------------"

def _random_permutations(n, rng):
    base = list(range(n))
    out = []
    for _ in range(n):
        p = base[:]
        rng.shuffle(p)
        out.append(p)
    return out

"--------------------------------------------------"
"--------------------------------------------------"
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    students = {i: p for i, p in enumerate(_random_permutations(a.n, rng))}
    schools = {i: p for i, p in enumerate(_random_permutations(a.n, rng))}
    def rank_map(prefs):
        return {i: {str(x): r for r, x in enumerate(lst)} for i, lst in prefs.items()}
    bundle = {
        "n": a.n,
        "students": {str(k): v for k,v in students.items()},
        "schools": {str(k): v for k,v in schools.items()},
        "rank_students": {str(k): rank_map(students)[k] for k in students},
        "rank_schools": {str(k): rank_map(schools)[k] for k in schools}
    }
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(bundle, f, ensure_ascii=False, indent=2)
    print(f"Écrit: {a.out}")

if __name__ == "__main__":
    main()
    
"--------------------------------------------------"
"--------------------------------------------------"
