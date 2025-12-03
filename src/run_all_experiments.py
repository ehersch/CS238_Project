"""
Compare POMDP across:
- 4 futures contracts (CL1-CL4)
- Hurricane vs no hurricane
- Log returns vs simple returns
"""

import pandas as pd
import numpy as np
from pomdp_model import WTIPOMDPModel
import os

os.chdir(
    "/Users/ethanhersch/Documents/Documents/Stanford/CS238/FinalProject/CS238_Project/src"
)


def run_experiment(df, feature_cols, return_col, name):
    """Run POMDP and return results."""
    train_size = int(len(df) * 0.8)
    train_df = df.iloc[:train_size].copy()
    test_df = df.iloc[train_size:].copy()

    pomdp = WTIPOMDPModel(n_hidden_states=4, n_obs_clusters=8, gamma=0.95)
    pomdp.fit(train_df, feature_cols, return_col)
    pomdp.train_qlearning(train_df, feature_cols, return_col, n_episodes=50, alpha=0.1)

    result = pomdp.evaluate(test_df, feature_cols, return_col, use_qlearning=True)

    # Buy & hold baseline
    test_returns = test_df.dropna(subset=[return_col])[return_col].values[:-1]
    bh_total = test_returns.sum()
    bh_sharpe = test_returns.mean() / (test_returns.std() + 1e-8) * np.sqrt(252)

    return {
        "name": name,
        "pomdp_total": result["total_reward"],
        "pomdp_sharpe": result["sharpe"],
        "bh_total": bh_total,
        "bh_sharpe": bh_sharpe,
    }


results = []

for contract in [1, 2, 3, 4]:
    print(f"\n{'='*60}")
    print(f"CONTRACT CL{contract}")
    print("=" * 60)

    # No hurricane (original data)
    df = pd.read_csv(f"wti_{contract}_data.csv")
    feature_cols = [
        c for c in df.columns if c not in ["Date", "ret_CL1", "spot_price", "CL1"]
    ]
    return_col = "ret_CL1"
    print(f"\nNo Hurricane...")
    r = run_experiment(df, feature_cols, return_col, f"CL{contract}_no_hurricane")
    results.append(r)

    # With hurricane
    df = pd.read_csv(f"wti_{contract}_data_with_hurricanes.csv")
    feature_cols = [
        c for c in df.columns if c not in ["Date", "ret_CL1", "spot_price", "CL1"]
    ]
    print(f"With Hurricane...")
    r = run_experiment(df, feature_cols, return_col, f"CL{contract}_hurricane")
    results.append(r)

    # Log returns
    df = pd.read_csv(f"wti_{contract}_logreturns.csv")
    feature_cols = [
        c for c in df.columns if c not in ["Date", "ret_CL1", "spot_price", "CL1"]
    ]
    print(f"Log Returns...")
    r = run_experiment(df, feature_cols, return_col, f"CL{contract}_logreturns")
    results.append(r)

    # Simple returns
    df = pd.read_csv(f"wti_{contract}_simplereturns.csv")
    feature_cols = [
        c for c in df.columns if c not in ["Date", "ret_CL1", "spot_price", "CL1"]
    ]
    print(f"Simple Returns...")
    r = run_experiment(df, feature_cols, return_col, f"CL{contract}_simplereturns")
    results.append(r)

# Print results table
print("\n" + "=" * 80)
print("RESULTS SUMMARY")
print("=" * 80)
print(
    f"{'Experiment':<25} {'POMDP Total':>12} {'POMDP Sharpe':>12} {'B&H Total':>12} {'B&H Sharpe':>12}"
)
print("-" * 80)

for r in results:
    print(
        f"{r['name']:<25} {r['pomdp_total']:>12.4f} {r['pomdp_sharpe']:>12.4f} {r['bh_total']:>12.4f} {r['bh_sharpe']:>12.4f}"
    )

# Save to CSV
df_results = pd.DataFrame(results)
df_results.to_csv("experiment_results.csv", index=False)
print("\nResults saved to experiment_results.csv")
