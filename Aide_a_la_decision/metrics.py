
from typing import Dict, List, Tuple
from statistics import mean, median, pstdev

"--------------------------------------------------"

PrefList = Dict[int, List[int]]
RankMap = Dict[int, Dict[int, int]]

def build_rank_map(prefs: PrefList) -> RankMap:
    return {i: {item: r for r, item in enumerate(lst)} for i, lst in prefs.items()}
    
"--------------------------------------------------"

def ranks_students(bundle: dict, matching_students: Dict[int, int]) -> Dict[int, int]:
    rank_students_map: RankMap = {int(k): {int(x): int(r) for x, r in v.items()} 
                                  for k, v in bundle["rank_students"].items()}
    return {e: rank_students_map[e][s] for e, s in matching_students.items()}

"--------------------------------------------------"

def ranks_schools(bundle: dict, matching_schools: Dict[int, int]) -> Dict[int, int]:
    rank_schools_map: RankMap = {int(k): {int(x): int(r) for x, r in v.items()} 
                                 for k, v in bundle["rank_schools"].items()}
    return {s: rank_schools_map[s][e] for s, e in matching_schools.items()}

"--------------------------------------------------"

def score_from_rank(rank: int, n: int) -> float:
    if n <= 1:
        return 1.0
    return 1.0 - (rank / (n - 1))

"--------------------------------------------------"

def gini(values: List[float]) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    total = sum(sorted_vals)
    if total == 0:
        return 0.0
    cum = 0.0
    for i, v in enumerate(sorted_vals, 1):
        cum += i * v
    return (2 * cum) / (n * total) - (n + 1) / n

"--------------------------------------------------"

def topk_rate(ranks: List[int], k: int) -> float:
    if not ranks:
        return 0.0
    return sum(1 for r in ranks if r < k) / len(ranks)

"--------------------------------------------------"

def stats_from_ranks(ranks: List[int], n: int) -> dict:
    if not ranks:
        return {}
    scores = [score_from_rank(r, n) for r in ranks]
    return {
        "n_agents": len(ranks),
        "mean_rank_1based": mean([r + 1 for r in ranks]),
        "median_rank_1based": median([r + 1 for r in ranks]),
        "std_rank": pstdev([float(r) for r in ranks]) if len(ranks) > 1 else 0.0,
        "top1_rate": topk_rate(ranks, 1),
        "top3_rate": topk_rate(ranks, min(3, n)),
        "mean_score": mean(scores),
        "gini_score": gini(scores),
        "hist_1based": {str(i): sum(1 for r in ranks if r + 1 == i) for i in range(1, n + 1)},
    }

"--------------------------------------------------"

def is_stable(bundle: dict, matching_students: Dict[int, int]) -> bool:
    n = int(bundle["n"])
    students = {int(k): v for k, v in bundle["students"].items()}
    rank_schools = {int(k): {int(x): int(r) for x, r in v.items()} for k, v in bundle["rank_schools"].items()}
    matchS = {s: e for e, s in matching_students.items()}
    for e in range(n):
        s_matched = matching_students[e]
        for s in students[e]:
            if s == s_matched:
                break
            e_current = matchS[s]
            if rank_schools[s][e] < rank_schools[s][e_current]:
                return False
    return True
    
"--------------------------------------------------"
