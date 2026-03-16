import numpy as np
import xarray as xr
from gymnasium import Env, spaces

class OTECEnvReal(Env):
    def __init__(self, netcdf_path):
        super().__init__()

        ds = xr.open_dataset(netcdf_path)
        sst = ds["analysed_sst"].values - 273.15
        self.sst_mean = float(np.nanmean(sst))
        self.season = ds.attrs.get("season", "unknown")
        ds.close()

        self.delta_t = max(self.sst_mean - 5.0, 18.0)

        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0]),
            high=np.array([40, 30, 2, 300]),
            dtype=np.float32
        )

        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(3,), dtype=np.float32
        )

        self.max_steps = 500
        self.step_count = 0

    def reset(self, seed=None, options=None):
        self.step_count = 0
        self.state = np.array(
            [self.sst_mean, self.delta_t, 1.3, 240.0],
            dtype=np.float32
        )
        return self.state, {}

    def step(self, action):
        # ---- SAFETY CLIP (CRITICAL FIX) ----
        action = np.clip(action, 0.0, 1.0)

        dm, pump, turb = action

        flow = 200 + 100 * dm
        pressure = 1.2 + 0.2 * dm
        pump_eff = 0.6 + 0.4 * pump

        # ---- SAFE DISCRETE MAPPING ----
        turb_idx = int(np.clip(np.floor(turb * 3), 0, 2))
        turb_eff = [0.6, 0.9, 1.2][turb_idx]

        power = (self.delta_t ** 2.1) * flow * pump_eff * turb_eff * 0.0004

        penalty = 20 if power < 30 else 0
        reward = power - penalty - 0.05 * np.sum(action ** 2)

        self.state = np.array(
            [self.sst_mean, self.delta_t, pressure, flow],
            dtype=np.float32
        )

        self.step_count += 1
        done = self.step_count >= self.max_steps

        info = {
            "power": float(power),
            "sst": self.sst_mean,
            "delta_t": self.delta_t,
            "season": self.season,
            "turbine_level": turb_idx
        }

        return self.state, reward, done, False, info

