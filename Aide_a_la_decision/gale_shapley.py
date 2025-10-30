import argparse, json
from typing import Dict, List, Tuple

PrefList = Dict[int, List[int]]
RankMap = Dict[int, Dict[int, int]]

def deferred_acceptance(proposer_prefs: PrefList, responder_ranks: RankMap) -> Tuple[Dict[int, int], Dict[int, int]]:
    n = len(proposer_prefs)
    free = list(proposer_prefs.keys())
    next_index = {p: 0 for p in proposer_prefs}
    match_p = {p: None for p in proposer_prefs}
    match_r = {r: None for r in responder_ranks}

    while free:
        p = free.pop()
        if next_index[p] >= n:
            continue
        r = proposer_prefs[p][next_index[p]]
        next_index[p] += 1

        if match_r[r] is None:
            match_r[r] = p
            match_p[p] = r
        else:
            cur = match_r[r]
            if responder_ranks[r][p] < responder_ranks[r][cur]:
                match_r[r] = p
                match_p[p] = r
                match_p[cur] = None
                free.append(cur)
            else:
                free.append(p)

    match_p = {int(k): int(v) for k, v in match_p.items() if v is not None}
    match_r = {int(k): int(v) for k, v in match_r.items() if v is not None}
    return match_p, match_r
    
"--------------------------------------------------"

def da_students_proposing(bundle: dict):
    students = {int(k): v for k, v in bundle["students"].items()}
    rank_schools = {int(k): {int(x): int(r) for x, r in bundle["rank_schools"][str(k)].items()} for k in bundle["schools"].keys()}
    return deferred_acceptance(students, rank_schools)

"--------------------------------------------------"

def da_schools_proposing(bundle: dict):
    schools = {int(k): v for k, v in bundle["schools"].items()}
    rank_students = {int(k): {int(x): int(r) for x, r in bundle["rank_students"][str(k)].items()} for k in bundle["students"].keys()}
    match_s_to_e, match_e_to_s = deferred_acceptance(schools, rank_students)
    return match_e_to_s, match_s_to_e

"--------------------------------------------------"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", type=str, required=True)
    parser.add_argument("--mode", choices=["DA_E", "DA_S"], required=True)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    with open(args.in_path, "r", encoding="utf-8") as f:
        bundle = json.load(f)

    if args.mode == "DA_E":
        matchE, matchS = da_students_proposing(bundle)
    else:
        matchE, matchS = da_schools_proposing(bundle)

    out = {
        "mode": args.mode,
        "matching_students": {str(k): v for k, v in matchE.items()},
        "matching_schools": {str(k): v for k, v in matchS.items()},
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"Écrit: {args.out}")

if __name__ == "__main__":
    main()
"--------------------------------------------------"
