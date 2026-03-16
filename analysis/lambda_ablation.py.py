import subprocess
import pandas as pd
import matplotlib.pyplot as plt

lambdas = [0, 1000, 5000, 10000]
tasks = ["summer", "winter", "rainy", "spring"]

results = []

for lam in lambdas:
    print(f"\nRunning PPO+EWC with lambda = {lam}")

    subprocess.run(
        ["python", "step2_ppo_ewc_lifelong.py", str(lam)],
        check=True
    )

    df = pd.read_csv("results_ppo_ewc.csv")

    # Summer forgetting
    p_initial = df[
        (df.trained_upto == "summer") &
        (df.evaluated_on == "summer")
    ].power_mean.values[0]

    p_final = df[
        (df.trained_upto == "spring") &
        (df.evaluated_on == "summer")
    ].power_mean.values[0]

    forgetting = (p_initial - p_final) / p_initial

    final_avg_power = df[
        df.trained_upto == "spring"
    ].power_mean.mean()

    results.append({
        "lambda": lam,
        "summer_forgetting": forgetting,
        "final_avg_power": final_avg_power
    })

res = pd.DataFrame(results)
res.to_csv("lambda_ablation.csv", index=False)
print(res)

# ==============================
# PLOT: Forgetting vs Lambda
# ==============================
plt.figure(figsize=(7, 5))
plt.plot(res["lambda"], res["summer_forgetting"], "o-", linewidth=2)
plt.xlabel("EWC Lambda")
plt.ylabel("Summer Forgetting Index")
plt.title("Stability–Plasticity Tradeoff")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("fig_lambda_forgetting.png", dpi=300)
plt.show()

# ==============================
# PLOT: Power vs Lambda
# ==============================
plt.figure(figsize=(7, 5))
plt.plot(res["lambda"], res["final_avg_power"], "s-", linewidth=2)
plt.xlabel("EWC Lambda")
plt.ylabel("Final Average Power")
plt.title("Performance vs Memory Strength")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("fig_lambda_power.png", dpi=300)
plt.show()
