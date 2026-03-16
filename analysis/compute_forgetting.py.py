import pandas as pd

tasks = ["summer", "winter", "rainy", "spring"]

ppo = pd.read_csv("results_ppo.csv")
ewc = pd.read_csv("results_ppo_ewc.csv")


def compute_forgetting(df, tasks):
    forgetting = {}
    final_task = tasks[-1]

    for task in tasks[:-1]:
        p_initial = df[
            (df.trained_upto == task) &
            (df.evaluated_on == task)
        ].power_mean.values[0]

        p_final = df[
            (df.trained_upto == final_task) &
            (df.evaluated_on == task)
        ].power_mean.values[0]

        forgetting[task] = (p_initial - p_final) / p_initial

    return forgetting


ppo_forgetting = compute_forgetting(ppo, tasks)
ewc_forgetting = compute_forgetting(ewc, tasks)

print("\n=== Catastrophic Forgetting Index ===")
print("PPO:")
for k, v in ppo_forgetting.items():
    print(f"{k}: {v:.3f}")

print("\nPPO + EWC:")
for k, v in ewc_forgetting.items():
    print(f"{k}: {v:.3f}")
