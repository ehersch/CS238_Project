# POMDP Oil Futures Trading

This document explains how to set up and run the POMDP (Partially Observable Markov Decision Process) model for WTI crude oil futures trading.

## Overview

The POMDP approach models oil trading as a decision problem where:
- **Hidden States**: Market regimes (bull, bear, volatile, calm) that we can't directly observe
- **Observations**: Weather data (temperature, precipitation, hurricanes) that we CAN observe
- **Actions**: SHORT (-1), FLAT (0), LONG (+1)
- **Belief State**: Probability distribution over hidden states, updated via Bayesian inference

## Files

| File | Description |
|------|-------------|
| `pomdp_model.py` | Core POMDP implementation with Q-learning |
| `pomdp_trading.ipynb` | Interactive notebook for training and evaluation |
| `wti_env.py` | Gymnasium environment for deep RL methods |
| `create_datasets.py` | Creates log returns and simple returns datasets |
| `compare_models.py` | Compares log vs simple returns models |

## Data Files

Two versions of each dataset (contracts 1-4):

| File | Description |
|------|-------------|
| `wti_X_logreturns.csv` | Log returns - drops NaN from negative prices |
| `wti_X_simplereturns.csv` | Simple returns - keeps all rows including April 2020 |
| `wti_X_data_with_hurricanes.csv` | Original merged data with hurricane features |

## Quick Start

### 1. Install Dependencies

```bash
pip install pandas numpy scikit-learn matplotlib
pip install gymnasium  # for deep RL (optional)
```

### 2. Generate Datasets

```bash
cd src
python create_datasets.py
```

This creates both log returns and simple returns versions of the data.

### 3. Run POMDP Model

```python
import pandas as pd
from pomdp_model import WTIPOMDPModel

# Load data
df = pd.read_csv('wti_1_logreturns.csv')

# Define features (all weather + hurricane columns)
feature_cols = [c for c in df.columns if c not in ['Date', 'ret_CL1', 'spot_price', 'CL1']]
return_col = 'ret_CL1'

# Train/test split (sequential - no data leakage)
train_size = int(len(df) * 0.8)
train_df = df.iloc[:train_size]
test_df = df.iloc[train_size:]

# Create and fit POMDP
pomdp = WTIPOMDPModel(
    n_hidden_states=4,   # Number of market regimes
    n_obs_clusters=8,    # Number of weather observation types
    gamma=0.95           # Discount factor
)
pomdp.fit(train_df, feature_cols, return_col)

# Train Q-learning over belief states
pomdp.train_qlearning(train_df, feature_cols, return_col, n_episodes=100)

# Evaluate on test set
results = pomdp.evaluate(test_df, feature_cols, return_col)
print(f"Total Reward: {results['total_reward']:.4f}")
print(f"Sharpe Ratio: {results['sharpe']:.4f}")
```

### 4. Using the Gymnasium Environment

For deep RL methods (DQN, PPO, etc.):

```python
from wti_env import WTIEnv

df = pd.read_csv('wti_1_logreturns.csv')
feature_cols = [c for c in df.columns if c not in ['Date', 'ret_CL1', 'spot_price', 'CL1']]

env = WTIEnv(
    df=df,
    feature_cols=feature_cols,
    return_col='ret_CL1',
    window=20,        # Observation window size
    cost=0.0005,      # Transaction cost
    normalize=True    # Normalize features (recommended)
)

# Standard Gymnasium API
obs, info = env.reset()
action = env.action_space.sample()  # 0=SHORT, 1=FLAT, 2=LONG
obs, reward, terminated, truncated, info = env.step(action)
```

## Key Design Decisions

### Log Returns vs Simple Returns

| Approach | Pros | Cons |
|----------|------|------|
| **Log Returns** | Standard in finance, symmetric | Can't handle negative prices (NaN) |
| **Simple Returns** | Handles negative prices | Extreme values can distort clustering |

April 2020 had negative oil prices (-$37.63). Log returns produce NaN for these days. Simple returns keep them but with extreme values (-306%, +126%).

### Sequential Train/Test Split

We split data **sequentially by date** (not randomly) to prevent data leakage:
- Train: 1986-01-02 → 2016-07-29
- Test: 2016-08-01 → 2024-04-05

### POMDP Components

1. **State Discretization**: K-means clustering on returns to identify market regimes
2. **Observation Discretization**: K-means clustering on weather features
3. **Transition Model**: P(s'|s) learned from historical regime transitions
4. **Observation Model**: P(o|s) learned from weather-regime correlations
5. **Belief Updates**: Bayesian inference after each observation
6. **Policy**: Q-learning over discretized belief states

## Hyperparameter Tuning

Key parameters to experiment with:

```python
WTIPOMDPModel(
    n_hidden_states=4,   # Try 2-6
    n_obs_clusters=8,    # Try 4-16
    gamma=0.95           # Discount factor, try 0.9-0.99
)

pomdp.train_qlearning(
    n_episodes=100,      # More = better but slower
    alpha=0.1,           # Learning rate
    epsilon_start=1.0,   # Initial exploration
    epsilon_end=0.1      # Final exploration
)
```

## Comparing Models

Run the comparison script to evaluate log vs simple returns:

```bash
python compare_models.py
```

## Next Steps

1. **Tune hyperparameters**: Grid search over n_states, n_obs, gamma
2. **Try deep RL**: Install stable-baselines3 for DQN/PPO
3. **Add features**: Integrate OVX volatility index
4. **Improve belief representation**: Use continuous belief states with neural networks
