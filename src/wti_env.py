import numpy as np
import pandas as pd
import gym
from gym import spaces


class WTIEnv(gym.Env):
    # start with window size 20 but can adjust
    # include some arbitrary transaction cost per position change
    def __init__(self, df, feature_cols, return_col, window=20, cost=0.0005):
        super().__init__()
        self.df = df.reset_index(drop=True)
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

    def reset(self):
        self.t = self.window
        self.position = 0  # -1/0/1 mapped to {0,1,2} outside
        return self._get_obs()

    def step(self, action):
        prev_pos = self.position
        self.position = action - 1  # map 0 -> -1, 1 -> 0, 2 -> +1

        # next-day futures return
        ret = self.df.loc[self.t, self.return_col]

        # Trading P&L
        pnl = self.position * ret

        # transaction cost: penalize changes in position
        cost = self.cost * abs(self.position - prev_pos)

        reward = pnl - cost

        self.t += 1
        done = self.t >= len(self.df)
        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        window_data = self.df.iloc[self.t - self.window : self.t][self.feature_cols]
        return window_data.to_numpy(np.float32)
