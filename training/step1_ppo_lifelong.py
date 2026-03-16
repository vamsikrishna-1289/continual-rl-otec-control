from stable_baselines3 import PPO
import pandas as pd
import numpy as np
from env_otec import OTECEnvReal
import torch.nn as nn

SEASONS = ["summer", "winter", "rainy", "spring"]
FILES = {s: f"data/{s}_location.nc" for s in SEASONS}

model = None
results = []

for i, train_season in enumerate(SEASONS):
    env = OTECEnvReal(FILES[train_season])

    if model is None:
        model = PPO(
            "MlpPolicy",
            env,
            policy_kwargs=dict(net_arch=[256, 256], activation_fn=nn.ReLU),
            learning_rate=3e-4,
            verbose=0
        )

    model.learn(total_timesteps=50_000)

    for eval_season in SEASONS[:i+1]:
        eval_env = OTECEnvReal(FILES[eval_season])
        obs, _ = eval_env.reset()
        powers = []

        for _ in range(500):
            act, _ = model.predict(obs, deterministic=True)
            obs, _, done, _, info = eval_env.step(act)
            powers.append(info["power"])
            if done:
                obs, _ = eval_env.reset()

        results.append({
            "trained_upto": train_season,
            "evaluated_on": eval_season,
            "power_mean": np.mean(powers)
        })

df = pd.DataFrame(results)
df.to_csv("results_ppo.csv", index=False)
print(df)
