
import argparse, json
from typing import Dict, List

"--------------------------------------------------"

def deferred_acceptance(proposer_prefs, responder_ranks):
    n = len(proposer_prefs)
    free = list(map(int, proposer_prefs.keys()))
    next_idx = {p: 0 for p in free}
    match_p = {p: None for p in free}
    match_r = {int(r): None for r in responder_ranks.keys()}
    while free:
        p = free.pop()
        if next_idx[p] >= n:
            continue
        r = proposer_prefs[str(p)][next_idx[p]]
        next_idx[p] += 1
        if match_r[r] is None:
            match_r[r] = p
            match_p[p] = r
        else:
            cur = match_r[r]
            if responder_ranks[str(r)][str(p)] < responder_ranks[str(r)][str(cur)]:
                match_r[r] = p
                match_p[p] = r
                match_p[cur] = None
                free.append(cur)
            else:
                free.append(p)
    mp = {str(k): v for k, v in match_p.items() if v is not None}
    mr = {str(k): v for k, v in match_r.items() if v is not None}
    return mp, mr

"--------------------------------------------------"
"--------------------------------------------------"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--mode", choices=["DA_E","DA_S"], required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    with open(a.in_path, "r", encoding="utf-8") as f:
        bundle = json.load(f)
    if a.mode == "DA_E":
        proposer = bundle["students"]
        responder_ranks = bundle["rank_schools"]
        ms, mr = deferred_acceptance(proposer, responder_ranks)
    else:
        proposer = bundle["schools"]
        responder_ranks = bundle["rank_students"]
        mr, ms = deferred_acceptance(proposer, responder_ranks)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump({"mode": a.mode, "matching_students": ms, "matching_schools": mr}, f, ensure_ascii=False, indent=2)
    print(f"Écrit: {a.out}")

if __name__ == "__main__":
    main()
    
"--------------------------------------------------"
"--------------------------------------------------"
