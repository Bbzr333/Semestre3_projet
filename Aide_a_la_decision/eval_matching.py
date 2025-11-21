
import argparse, json
from typing import Dict
from metrics import ranks_students, ranks_schools, stats_from_ranks, is_stable

"--------------------------------------------------"
"--------------------------------------------------"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefs", required=True)
    parser.add_argument("--match", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    with open(args.prefs, "r", encoding="utf-8") as f:
        bundle = json.load(f)
    with open(args.match, "r", encoding="utf-8") as f:
        match = json.load(f)

    m_students = {int(k): int(v) for k, v in match["matching_students"].items()}
    m_schools = {int(k): int(v) for k, v in match["matching_schools"].items()}

    rE = ranks_students(bundle, m_students)
    rS = ranks_schools(bundle, m_schools)

    n = int(bundle["n"])
    statsE = stats_from_ranks(list(rE.values()), n)
    statsS = stats_from_ranks(list(rS.values()), n)
    stability = is_stable(bundle, m_students)

    out = {
        "mode": match["mode"],
        "stable": stability,
        "students": {"ranks_0based": rE, **statsE},
        "schools": {"ranks_0based": rS, **statsS}
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"Écrit: {args.out}")

if __name__ == "__main__":
    main()
    
"--------------------------------------------------"
"--------------------------------------------------"
