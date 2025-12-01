import numpy as np
import pandas as pd
try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces
from sklearn.preprocessing import StandardScaler


def prepare_data(df, feature_cols, return_col, normalize=True):
    """
    Prepare data by handling NaN values and normalizing features.

    The NaN issue comes from:
    1. Negative oil prices in April 2020 (log of negative = NaN)
    2. Last row has NaN due to shift in return calculation
    """
    df = df.copy()

    # Drop rows with NaN in the return column
    df = df.dropna(subset=[return_col])
    df = df.reset_index(drop=True)

    # Normalize features if requested (important for RL stability)
    if normalize:
        scaler = StandardScaler()
        df[feature_cols] = scaler.fit_transform(df[feature_cols])

    return df


class WTIEnv(gym.Env):
    """
    OpenAI Gym environment for WTI crude oil futures trading.

    State: window of normalized weather features
    Actions: SHORT (0), FLAT (1), LONG (2)
    Reward: position * return - transaction_cost
    """
    def __init__(self, df, feature_cols, return_col, window=20, cost=0.0005, normalize=True):
        super().__init__()

        # Prepare data (handles NaN, normalizes)
        self.df = prepare_data(df, feature_cols, return_col, normalize=normalize)
        self.feature_cols = feature_cols
        self.return_col = return_col
        self.window = window
        self.cost = cost

        # Actions: short = 0, flat = 1, long = 2
        self.action_space = spaces.Discrete(3)

        # Observation shape: window × features
        n_features = len(feature_cols)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(window, n_features), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self.t = self.window
        self.position = 0  # -1/0/1 mapped to {0,1,2} outside
        return self._get_obs(), {}

    def step(self, action):
        prev_pos = self.position
        self.position = action - 1  # map 0 -> -1, 1 -> 0, 2 -> +1

        # next-day futures return
        ret = self.df.loc[self.t, self.return_col]

        # Safety check for NaN (shouldn't happen after prepare_data, but just in case)
        if np.isnan(ret):
            ret = 0.0

        # Trading P&L
        pnl = self.position * ret

        # transaction cost: penalize changes in position
        cost = self.cost * abs(self.position - prev_pos)

        reward = pnl - cost

        self.t += 1
        terminated = self.t >= len(self.df)
        truncated = False
        return self._get_obs(), reward, terminated, truncated, {}

    def _get_obs(self):
        window_data = self.df.iloc[self.t - self.window : self.t][self.feature_cols]
        return window_data.to_numpy(np.float32)
