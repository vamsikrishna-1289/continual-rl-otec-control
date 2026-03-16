import torch
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from env_otec import OTECEnvReal
import torch.nn as nn
import sys

SEASONS = ["summer", "winter", "rainy", "spring"]
FILES = {s: f"data/{s}_location.nc" for s in SEASONS}

LAMBDA_EWC = int(sys.argv[1]) if len(sys.argv) > 1 else 5000



def compute_fisher(model, env, samples=2000):
    fisher = {
        name: torch.zeros_like(param)
        for name, param in model.policy.named_parameters()
    }

    obs, _ = env.reset()

    for _ in range(samples):
        obs_tensor = torch.as_tensor(obs).float().unsqueeze(0)

        dist = model.policy.get_distribution(obs_tensor)
        action = dist.sample()
        log_prob = dist.log_prob(action).mean()

        model.policy.zero_grad()
        (-log_prob).backward()

        for name, param in model.policy.named_parameters():
            if param.grad is not None:
                fisher[name] += param.grad.data.pow(2)

        obs, _, done, _, _ = env.step(action.squeeze(0).detach().numpy())
        if done:
            obs, _ = env.reset()

    for name in fisher:
        fisher[name] /= samples

    return fisher


model = None
fisher = None
old_params = None
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
    else:
        for name, param in model.policy.named_parameters():
            param.data -= LAMBDA_EWC * fisher[name] * (param.data - old_params[name])

    model.learn(total_timesteps=50_000)

    fisher = compute_fisher(model, env)
    old_params = {n: p.clone() for n, p in model.policy.named_parameters()}

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
df.to_csv("results_ppo_ewc.csv", index=False)
print(df)
