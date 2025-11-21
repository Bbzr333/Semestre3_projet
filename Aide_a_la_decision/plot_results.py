import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Charger le CSV
df = pd.read_csv("experiments/results.csv")

# Créer un dossier figures/
FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

def save_show(fig_name):
    """Enregistre la figure dans le dossier et l’affiche."""
    out_path = FIG_DIR / f"{fig_name}.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Figure enregistrée : {out_path}")
    plt.show()


# --- 1. RANG MOYEN ÉTUDIANTS ---
plt.figure()
for mode in df["mode"].unique():
    subset = df[df["mode"] == mode]
    means = subset.groupby("n")["student_mean_rank"].mean()
    plt.plot(means.index, means.values, marker="o", label=mode)

plt.xlabel("n")
plt.ylabel("Student Mean Rank")
plt.title("Student Mean Rank vs n")
plt.legend()
save_show("student_mean_rank")


# --- 2. RANG MOYEN ÉTABLISSEMENTS ---
plt.figure()
for mode in df["mode"].unique():
    subset = df[df["mode"] == mode]
    means = subset.groupby("n")["school_mean_rank"].mean()
    plt.plot(means.index, means.values, marker="o", label=mode)

plt.xlabel("n")
plt.ylabel("School Mean Rank")
plt.title("School Mean Rank vs n")
plt.legend()
save_show("school_mean_rank")


# --- 3. TOP-1 ÉTUDIANTS ---
plt.figure()
for mode in df["mode"].unique():
    subset = df[df["mode"] == mode]
    means = subset.groupby("n")["student_top1"].mean()
    plt.plot(means.index, means.values, marker="o", label=mode)

plt.xlabel("n")
plt.ylabel("Top-1 rate (students)")
plt.title("Student Top-1 Rate vs n")
plt.legend()
save_show("student_top1")


# --- 4. TOP-3 ÉTUDIANTS ---
plt.figure()
for mode in df["mode"].unique():
    subset = df[df["mode"] == mode]
    means = subset.groupby("n")["student_top3"].mean()
    plt.plot(means.index, means.values, marker="o", label=mode)

plt.xlabel("n")
plt.ylabel("Top-3 rate (students)")
plt.title("Student Top-3 Rate vs n")
plt.legend()
save_show("student_top3")


# --- 5. TOP-1 ÉTABLISSEMENTS ---
plt.figure()
for mode in df["mode"].unique():
    subset = df[df["mode"] == mode]
    means = subset.groupby("n")["school_top1"].mean()
    plt.plot(means.index, means.values, marker="o", label=mode)

plt.xlabel("n")
plt.ylabel("Top-1 rate (schools)")
plt.title("School Top-1 Rate vs n")
plt.legend()
save_show("school_top1")


# --- 6. TOP-3 ÉTABLISSEMENTS ---
plt.figure()
for mode in df["mode"].unique():
    subset = df[df["mode"] == mode]
    means = subset.groupby("n")["school_top3"].mean()
    plt.plot(means.index, means.values, marker="o", label=mode)

plt.xlabel("n")
plt.ylabel("Top-3 rate (schools)")
plt.title("School Top-3 Rate vs n")
plt.legend()
save_show("school_top3")


# --- 7. BOXPLOT ÉTUDIANTS ---
plt.figure()
data = [
    df[df["mode"] == mode]["student_mean_rank"]
    for mode in df["mode"].unique()
]
plt.boxplot(data, labels=df["mode"].unique())
plt.ylabel("Student Mean Rank")
plt.title("Student Rank Distribution by Mode")
save_show("student_boxplot")


# --- 8. BOXPLOT ÉTABLISSEMENTS ---
plt.figure()
data = [
    df[df["mode"] == mode]["school_mean_rank"]
    for mode in df["mode"].unique()
]
plt.boxplot(data, labels=df["mode"].unique())
plt.ylabel("School Mean Rank")
plt.title("School Rank Distribution by Mode")
save_show("school_boxplot")
