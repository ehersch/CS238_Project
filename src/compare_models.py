"""
Compare POMDP models trained on:
1. Log returns (drops negative price days)
2. Simple returns (includes negative price days)

This helps answer: Does including the April 2020 extreme event help or hurt?
"""

import pandas as pd
import numpy as np
from pomdp_model import WTIPOMDPModel


def run_comparison(train_pct=0.8):
    print("="*70)
    print(f"COMPARING LOG RETURNS vs SIMPLE RETURNS MODELS ({int(train_pct*100)}/{int((1-train_pct)*100)} split)")
    print("="*70)

    # Load both datasets
    df_log = pd.read_csv('wti_1_logreturns.csv')
    df_simple = pd.read_csv('wti_1_simplereturns.csv')

    feature_cols = [c for c in df_log.columns
                    if c not in ['Date', 'ret_CL1', 'spot_price', 'CL1']]
    return_col = 'ret_CL1'

    # Train/test split
    train_size = int(len(df_log) * train_pct)

    results = {}

    for name, df in [("Log Returns", df_log), ("Simple Returns", df_simple)]:
        print(f"\n{'='*70}")
        print(f"MODEL: {name}")
        print(f"{'='*70}")

        train_df = df.iloc[:train_size].copy()
        test_df = df.iloc[train_size:].copy()

        # Count usable samples
        train_valid = train_df.dropna(subset=[return_col])
        test_valid = test_df.dropna(subset=[return_col])
        print(f"Training samples: {len(train_valid)} (dropped {len(train_df) - len(train_valid)} NaN)")
        print(f"Test samples: {len(test_valid)} (dropped {len(test_df) - len(test_valid)} NaN)")

        # Create and train POMDP
        pomdp = WTIPOMDPModel(n_hidden_states=4, n_obs_clusters=8, gamma=0.95)
        pomdp.fit(train_df, feature_cols, return_col)

        print("\nTraining Q-learning (50 episodes)...")
        training_rewards = pomdp.train_qlearning(
            train_df, feature_cols, return_col,
            n_episodes=50, alpha=0.1
        )

        # Evaluate
        eval_results = pomdp.evaluate(test_df, feature_cols, return_col, use_qlearning=True)

        results[name] = {
            'total_reward': eval_results['total_reward'],
            'sharpe': eval_results['sharpe'],
            'mean_reward': eval_results['mean_reward'],
            'training_final': training_rewards[-1]
        }

        print(f"\nTest Results:")
        print(f"  Total Reward: {eval_results['total_reward']:.4f}")
        print(f"  Sharpe Ratio: {eval_results['sharpe']:.4f}")

    # Compare
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)

    print(f"\n{'Metric':<25} {'Log Returns':>15} {'Simple Returns':>15} {'Winner':>12}")
    print("-"*70)

    for metric in ['total_reward', 'sharpe', 'mean_reward']:
        log_val = results['Log Returns'][metric]
        simple_val = results['Simple Returns'][metric]
        winner = "Log" if log_val > simple_val else "Simple"
        print(f"{metric:<25} {log_val:>15.4f} {simple_val:>15.4f} {winner:>12}")

    # Baselines
    print("\n" + "-"*70)
    print("BASELINES (on test set):")
    print("-"*70)

    for name, df in [("Log Returns", df_log), ("Simple Returns", df_simple)]:
        test_df = df.iloc[train_size:].copy()
        test_returns = test_df.dropna(subset=[return_col])[return_col].values[:-1]

        buy_hold = test_returns.sum()
        sharpe = test_returns.mean() / (test_returns.std() + 1e-8) * np.sqrt(252)

        print(f"{name}:")
        print(f"  Buy & Hold Total: {buy_hold:.4f}, Sharpe: {sharpe:.4f}")

    # Special analysis: What happened around April 2020?
    print("\n" + "="*70)
    print("APRIL 2020 ANALYSIS (in training set)")
    print("="*70)

    apr_mask = (df_simple['Date'] >= '2020-04-15') & (df_simple['Date'] <= '2020-04-25')
    if apr_mask.any() and apr_mask.idxmax() < train_size:
        print("\nApril 2020 is in TRAINING set - model can learn from it!")

        apr_data = df_simple[apr_mask][['Date', 'CL1', 'ret_CL1']]
        print("\nSimple returns during crisis:")
        print(apr_data.to_string(index=False))

        extreme_return = df_simple.loc[df_simple['Date'] == '2020-04-20', 'ret_CL1'].values
        if len(extreme_return) > 0:
            print(f"\nApril 20 return: {extreme_return[0]:.2%}")
            print("A model that learns to go LONG during extreme negative prices")
            print("could capture this +126% single-day return!")
    else:
        print("\nApril 2020 is in TEST set - model hasn't seen it during training")


if __name__ == "__main__":
    import os
    os.chdir('/Users/andrewsung/Developer/CS238_Project/src')

    # Run with 80/20 split (April 2020 in TEST)
    run_comparison(train_pct=0.8)

    print("\n\n" + "#"*70)
    print("# NOW TRYING 91/9 SPLIT (April 2020 in TRAINING)")
    print("#"*70 + "\n")

    # Run with 91/9 split (April 2020 in TRAIN)
    run_comparison(train_pct=0.91)
