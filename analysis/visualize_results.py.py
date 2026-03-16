import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==============================
# LOAD RESULTS
# ==============================
ppo = pd.read_csv("results_ppo.csv")
ewc = pd.read_csv("results_ppo_ewc.csv")

# Order of tasks
tasks = ["summer", "winter", "rainy", "spring"]

# ==============================
# HELPER: extract performance matrix
# ==============================
def build_matrix(df):
    matrix = {t: [] for t in tasks}
    for trained in tasks:
        sub = df[df["trained_upto"] == trained]
        for eval_task in tasks:
            row = sub[sub["evaluated_on"] == eval_task]
            if len(row) == 0:
                matrix[eval_task].append(np.nan)
            else:
                matrix[eval_task].append(row["power_mean"].values[0])
    return matrix

ppo_mat = build_matrix(ppo)
ewc_mat = build_matrix(ewc)

# ==============================
# FIGURE 1: POWER RETENTION CURVES
# ==============================
plt.figure(figsize=(10, 6))

for task in tasks:
    plt.plot(tasks, ppo_mat[task], marker="o", linestyle="--", label=f"PPO eval on {task}")
    plt.plot(tasks, ewc_mat[task], marker="s", linestyle="-", label=f"PPO+EWC eval on {task}")

plt.xlabel("Training Progression")
plt.ylabel("Mean Power Output")
plt.title("Lifelong Performance Retention Across SST Regimes")
plt.legend(ncol=2, fontsize=9)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("fig1_power_retention.png", dpi=300)
plt.show()

# ==============================
# FIGURE 2: CATASTROPHIC FORGETTING (SUMMER TASK)
# ==============================
summer_ppo = ppo_mat["summer"]
summer_ewc = ewc_mat["summer"]

plt.figure(figsize=(8, 5))
plt.plot(tasks, summer_ppo, "r-o", linewidth=2, label="PPO (baseline)")
plt.plot(tasks, summer_ewc, "g-s", linewidth=2, label="PPO + EWC")

plt.xlabel("Training Progression")
plt.ylabel("Summer Power Output")
plt.title("Catastrophic Forgetting on Initial (Summer) Task")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("fig2_summer_forgetting.png", dpi=300)
plt.show()

# ==============================
# FIGURE 3: FORGETTING PERCENTAGE
# ==============================
def forgetting_curve(values):
    base = values[0]
    return [(base - v) / base * 100 for v in values]

forget_ppo = forgetting_curve(summer_ppo)
forget_ewc = forgetting_curve(summer_ewc)

plt.figure(figsize=(8, 5))
plt.plot(tasks, forget_ppo, "r-o", linewidth=2, label="PPO")
plt.plot(tasks, forget_ewc, "g-s", linewidth=2, label="PPO + EWC")

plt.axhline(0, color="black", linewidth=1)
plt.xlabel("Training Progression")
plt.ylabel("Forgetting (%)")
plt.title("Catastrophic Forgetting Reduction via Lifelong RL")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("fig3_forgetting_percentage.png", dpi=300)
plt.show()

# ==============================
# FIGURE 4: FINAL TASK COMPARISON (BAR)
# ==============================
final_ppo = [ppo_mat[t][-1] for t in tasks]
final_ewc = [ewc_mat[t][-1] for t in tasks]

x = np.arange(len(tasks))
width = 0.35

plt.figure(figsize=(9, 5))
plt.bar(x - width/2, final_ppo, width, label="PPO")
plt.bar(x + width/2, final_ewc, width, label="PPO + EWC")

plt.xticks(x, tasks)
plt.ylabel("Mean Power Output")
plt.title("Final Performance After Lifelong Training")
plt.legend()
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("fig4_final_comparison.png", dpi=300)
plt.show()

print("✅ Visualizations saved:")
print(" - fig1_power_retention.png")
print(" - fig2_summer_forgetting.png")
print(" - fig3_forgetting_percentage.png")
print(" - fig4_final_comparison.png")
