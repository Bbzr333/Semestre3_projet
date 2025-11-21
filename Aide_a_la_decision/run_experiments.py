import json
import csv
import subprocess
import sys
from pathlib import Path

# Configuration des expériences
SIZES = [10, 20, 50, 100, 200]   # tailles que tu veux tester
SEEDS = range(1, 21)         # 20 seeds par taille
MODES = ["DA_E", "DA_S"]     # étudiants proposants / écoles proposants

# Chemins des scripts (adapte si besoin)
PREF_SCRIPT = "preferences.py"
GS_SCRIPT = "gale_shapley.py"
EVAL_SCRIPT = "eval_matching.py"

# Dossier de sortie
OUT_DIR = Path("experiments")
OUT_DIR.mkdir(exist_ok=True)

CSV_PATH = OUT_DIR / "results.csv"

def run(cmd):
    """Exécute une commande shell et arrête tout en cas d’erreur."""
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        print("Erreur:", p.stderr)
        sys.exit(1)

# Création du CSV
with open(CSV_PATH, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow([
        "n", "seed", "mode",
        "stable",
        "student_mean_rank", "student_median_rank", "student_top1", "student_top3",
        "school_mean_rank", "school_median_rank", "school_top1", "school_top3",
    ])

    for n in SIZES:
        for seed in SEEDS:

            # 1) Générer préférences
            prefs_file = OUT_DIR / f"prefs_n{n}_seed{seed}.json"
            run([sys.executable, PREF_SCRIPT, "--n", str(n), "--seed", str(seed), "--out", str(prefs_file)])

            for mode in MODES:

                # 2) Faire tourner Gale–Shapley
                match_file = OUT_DIR / f"match_{mode}_n{n}_seed{seed}.json"
                run([
                    sys.executable, GS_SCRIPT,
                    "--in", str(prefs_file),
                    "--mode", mode,
                    "--out", str(match_file)
                ])

                # 3) Évaluer le matching
                metrics_file = OUT_DIR / f"metrics_{mode}_n{n}_seed{seed}.json"
                run([
                    sys.executable, EVAL_SCRIPT,
                    "--prefs", str(prefs_file),
                    "--match", str(match_file),
                    "--out", str(metrics_file)
                ])

                # 4) Extraire les résultats dans le CSV
                with open(metrics_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                writer.writerow([
                    n, seed, mode,
                    data["stable"],
                    data["students"]["mean_rank_1based"],
                    data["students"]["median_rank_1based"],
                    data["students"]["top1_rate"],
                    data["students"]["top3_rate"],
                    data["schools"]["mean_rank_1based"],
                    data["schools"]["median_rank_1based"],
                    data["schools"]["top1_rate"],
                    data["schools"]["top3_rate"],
                ])

print("FINI. Résultats dans :", CSV_PATH)
