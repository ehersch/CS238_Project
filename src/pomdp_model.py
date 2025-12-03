"""
POMDP Model for WTI Crude Oil Futures Trading

This implements a Partially Observable Markov Decision Process where:
- Hidden State: Market regime (bull, bear, volatile, calm)
- Observations: Weather features (temperature, precipitation, hurricanes)
- Actions: SHORT (-1), FLAT (0), LONG (+1)
- Reward: Position * daily return - transaction cost

The key insight is that weather affects oil supply/demand, but we can't directly
observe the market state - we only see weather and must infer likely price movements.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from collections import defaultdict


class WTIPOMDPModel:
    """
    POMDP model for oil futures trading based on weather observations.

    Hidden states are market regimes learned from historical return patterns.
    We maintain a belief state (probability distribution over regimes) and
    use weather observations to update beliefs and select actions.
    """

    def __init__(self, n_hidden_states=4, n_obs_clusters=8, gamma=0.95, discretization='kmeans'):
        """
        Args:
            n_hidden_states: Number of discrete market regimes
            n_obs_clusters: Number of discrete observation clusters for weather
            gamma: Discount factor
            discretization: 'kmeans' or 'uniform'
        """
        self.n_states = n_hidden_states
        self.n_obs = n_obs_clusters
        self.n_actions = 3  # SHORT, FLAT, LONG
        self.gamma = gamma
        self.discretization = discretization

        # Models to be learned
        self.state_kmeans = None  # Clusters returns into market regimes
        self.obs_kmeans = None  # Clusters weather into observation types
        self.scaler = None  # Normalizes weather features

        # For uniform discretization
        self.return_bins = None
        self.obs_bins = None

        # POMDP components (learned from data)
        self.T = None  # Transition matrix: T[s, a, s'] = P(s' | s, a)
        self.O = None  # Observation matrix: O[a, s', o] = P(o | s', a)
        self.R = None  # Reward: R[s, a] = expected reward

        # For belief-based Q-learning
        self.Q = defaultdict(lambda: np.zeros(self.n_actions))

    def _discretize_states(self, returns):
        """
        Discretize continuous returns into market regimes.
        """
        if self.discretization == 'uniform':
            return self._discretize_states_uniform(returns)
        else:
            return self._discretize_states_kmeans(returns)

    def _discretize_states_kmeans(self, returns):
        """
        Discretize using K-means clustering.
        """
        returns_2d = returns.reshape(-1, 1)
        self.state_kmeans = KMeans(n_clusters=self.n_states, random_state=42, n_init=10)
        states = self.state_kmeans.fit_predict(returns_2d)

        # Reorder clusters by return magnitude (0=most bearish, n-1=most bullish)
        cluster_means = [returns[states == i].mean() for i in range(self.n_states)]
        order = np.argsort(cluster_means)
        state_map = {old: new for new, old in enumerate(order)}
        states = np.array([state_map[s] for s in states])

        return states

    def _discretize_states_uniform(self, returns):
        """
        Discretize returns using uniform percentile bins.
        """
        # Create uniform bins based on percentiles
        percentiles = np.linspace(0, 100, self.n_states + 1)
        self.return_bins = np.percentile(returns, percentiles)
        self.return_bins[0] = -np.inf
        self.return_bins[-1] = np.inf

        states = np.digitize(returns, self.return_bins[1:-1])
        return states

    def _discretize_observations(self, weather_data):
        """
        Discretize continuous weather features into observation clusters.
        """
        if self.discretization == 'uniform':
            return self._discretize_observations_uniform(weather_data)
        else:
            return self._discretize_observations_kmeans(weather_data)

    def _discretize_observations_kmeans(self, weather_data):
        """
        Discretize using K-means clustering.
        """
        self.scaler = StandardScaler()
        weather_normalized = self.scaler.fit_transform(weather_data)

        self.obs_kmeans = KMeans(n_clusters=self.n_obs, random_state=42, n_init=10)
        observations = self.obs_kmeans.fit_predict(weather_normalized)

        return observations

    def _discretize_observations_uniform(self, weather_data):
        """
        Discretize using uniform bins per feature, then combine into single observation index.
        """
        self.scaler = StandardScaler()
        weather_normalized = self.scaler.fit_transform(weather_data)

        # Use fewer bins per feature to keep observation space manageable
        n_features = weather_normalized.shape[1]
        bins_per_feature = max(2, int(np.power(self.n_obs, 1/min(n_features, 3))))

        # Only use first 3 features (most important) to keep space small
        n_features_used = min(3, n_features)
        self.obs_bins = []

        discretized = np.zeros((len(weather_normalized), n_features_used), dtype=int)
        for i in range(n_features_used):
            percentiles = np.linspace(0, 100, bins_per_feature + 1)
            bins = np.percentile(weather_normalized[:, i], percentiles)
            bins[0] = -np.inf
            bins[-1] = np.inf
            self.obs_bins.append(bins)
            discretized[:, i] = np.digitize(weather_normalized[:, i], bins[1:-1])

        # Combine into single observation index
        observations = np.zeros(len(weather_normalized), dtype=int)
        for i in range(n_features_used):
            observations = observations * bins_per_feature + discretized[:, i]

        # Map to range [0, n_obs)
        observations = observations % self.n_obs

        return observations

    def _learn_transition_model(self, states):
        """
        Learn transition probabilities T[s, a, s'] = P(s' | s, a).

        Since our actions don't affect the market (we're too small),
        transitions are action-independent: T[s, s'] = P(s' | s).
        """
        self.T = np.zeros((self.n_states, self.n_actions, self.n_states))

        # Count transitions
        transition_counts = np.zeros((self.n_states, self.n_states))
        for t in range(len(states) - 1):
            s, s_next = states[t], states[t + 1]
            transition_counts[s, s_next] += 1

        # Normalize to probabilities (add small smoothing)
        for s in range(self.n_states):
            total = transition_counts[s].sum() + self.n_states * 0.01
            for s_next in range(self.n_states):
                prob = (transition_counts[s, s_next] + 0.01) / total
                # Same transition for all actions (market doesn't care about our trades)
                for a in range(self.n_actions):
                    self.T[s, a, s_next] = prob

    def _learn_observation_model(self, states, observations):
        """
        Learn observation probabilities O[a, s', o] = P(o | s', a).

        Since observations (weather) don't depend on our action, this is
        just O[s, o] = P(o | s).
        """
        self.O = np.zeros((self.n_actions, self.n_states, self.n_obs))

        # Count co-occurrences
        obs_counts = np.zeros((self.n_states, self.n_obs))
        for s, o in zip(states, observations):
            obs_counts[s, o] += 1

        # Normalize to probabilities
        for s in range(self.n_states):
            total = obs_counts[s].sum() + self.n_obs * 0.01
            for o in range(self.n_obs):
                prob = (obs_counts[s, o] + 0.01) / total
                for a in range(self.n_actions):
                    self.O[a, s, o] = prob

    def _learn_reward_model(self, states, returns):
        """
        Learn expected rewards R[s, a].

        Reward = position * return, where position depends on action:
        - SHORT (a=0): position = -1
        - FLAT (a=1): position = 0
        - LONG (a=2): position = +1
        """
        self.R = np.zeros((self.n_states, self.n_actions))

        # Compute mean return for each state
        state_returns = np.zeros(self.n_states)
        state_counts = np.zeros(self.n_states)

        for s, r in zip(states, returns):
            state_returns[s] += r
            state_counts[s] += 1

        for s in range(self.n_states):
            if state_counts[s] > 0:
                mean_return = state_returns[s] / state_counts[s]
            else:
                mean_return = 0

            # Reward for each action
            self.R[s, 0] = -1 * mean_return  # SHORT: profit when price drops
            self.R[s, 1] = 0  # FLAT: no exposure
            self.R[s, 2] = +1 * mean_return  # LONG: profit when price rises

    def fit(self, df, feature_cols, return_col):
        """
        Learn POMDP model from historical data.

        Args:
            df: DataFrame with weather features and returns
            feature_cols: List of weather feature column names
            return_col: Column name for returns
        """
        # Remove NaN
        df = df.dropna(subset=[return_col]).reset_index(drop=True)

        returns = df[return_col].values
        weather_data = df[feature_cols].values

        print(f"Learning POMDP from {len(df)} samples...")

        # Discretize into states and observations
        states = self._discretize_states(returns)
        observations = self._discretize_observations(weather_data)

        print(f"  - {self.n_states} hidden market states")
        print(f"  - {self.n_obs} observation clusters")

        # Learn model components
        self._learn_transition_model(states)
        self._learn_observation_model(states, observations)
        self._learn_reward_model(states, returns)

        # Store for later use
        self.states = states
        self.observations = observations
        self.returns = returns

        print("POMDP model learned!")

        return self

    def update_belief(self, belief, action, observation):
        """
        Bayesian belief update after taking action and receiving observation.

        b'(s') = η * O(o|s',a) * Σ_s T(s'|s,a) * b(s)

        Args:
            belief: Current belief state (probability distribution over states)
            action: Action taken
            observation: Observation received

        Returns:
            Updated belief state
        """
        new_belief = np.zeros(self.n_states)

        for s_next in range(self.n_states):
            # Sum over previous states
            transition_sum = 0
            for s in range(self.n_states):
                transition_sum += self.T[s, action, s_next] * belief[s]

            # Multiply by observation probability
            new_belief[s_next] = self.O[action, s_next, observation] * transition_sum

        # Normalize
        total = new_belief.sum()
        if total > 0:
            new_belief /= total
        else:
            # If all probabilities are 0, fall back to uniform
            new_belief = np.ones(self.n_states) / self.n_states

        return new_belief

    def get_observation(self, weather_features):
        """
        Convert continuous weather features to discrete observation.
        """
        weather_normalized = self.scaler.transform(weather_features.reshape(1, -1))

        if self.discretization == 'uniform':
            n_features_used = len(self.obs_bins)
            bins_per_feature = len(self.obs_bins[0]) - 1

            obs = 0
            for i in range(n_features_used):
                bin_idx = np.digitize(weather_normalized[0, i], self.obs_bins[i][1:-1])
                obs = obs * bins_per_feature + bin_idx

            return obs % self.n_obs
        else:
            return self.obs_kmeans.predict(weather_normalized)[0]

    def expected_reward(self, belief, action):
        """
        Compute expected reward given belief state and action.

        E[R | b, a] = Σ_s b(s) * R(s, a)
        """
        return np.dot(belief, self.R[:, action])

    def greedy_action(self, belief):
        """
        Select action that maximizes expected immediate reward.
        """
        expected_rewards = [
            self.expected_reward(belief, a) for a in range(self.n_actions)
        ]
        return np.argmax(expected_rewards)

    def _belief_to_key(self, belief, precision=2):
        """Convert belief to hashable key for Q-table."""
        return tuple(np.round(belief, precision))

    def qlearning_action(self, belief, epsilon=0.1):
        """
        Select action using epsilon-greedy Q-learning over belief states.
        """
        if np.random.random() < epsilon:
            return np.random.randint(self.n_actions)

        belief_key = self._belief_to_key(belief)
        return np.argmax(self.Q[belief_key])

    def update_q(self, belief, action, reward, next_belief, alpha=0.1):
        """
        Q-learning update for belief-based policy.
        """
        belief_key = self._belief_to_key(belief)
        next_belief_key = self._belief_to_key(next_belief)

        # Q-learning update
        best_next_q = np.max(self.Q[next_belief_key])
        td_target = reward + self.gamma * best_next_q
        td_error = td_target - self.Q[belief_key][action]
        self.Q[belief_key][action] += alpha * td_error

    def train_qlearning(
        self,
        df,
        feature_cols,
        return_col,
        n_episodes=100,
        alpha=0.1,
        epsilon_start=1.0,
        epsilon_end=0.1,
    ):
        """
        Train belief-based Q-learning policy.
        """
        df = df.dropna(subset=[return_col]).reset_index(drop=True)
        weather_data = df[feature_cols].values
        returns = df[return_col].values

        epsilon_decay = (epsilon_start - epsilon_end) / n_episodes

        total_rewards = []

        for episode in range(n_episodes):
            epsilon = max(epsilon_end, epsilon_start - episode * epsilon_decay)

            # Start with uniform belief
            belief = np.ones(self.n_states) / self.n_states
            episode_reward = 0
            position = 0

            for t in range(len(df) - 1):
                # Select action based on current belief
                action = self.qlearning_action(belief, epsilon)
                new_position = action - 1  # Map to -1, 0, +1

                # Calculate reward
                actual_return = returns[t]
                pnl = new_position * actual_return
                transaction_cost = 0.0005 * abs(new_position - position)
                reward = pnl - transaction_cost

                # Get next observation and update belief
                next_obs = self.get_observation(weather_data[t + 1])
                next_belief = self.update_belief(belief, action, next_obs)

                # Q-learning update
                self.update_q(belief, action, reward, next_belief, alpha)

                belief = next_belief
                position = new_position
                episode_reward += reward

            total_rewards.append(episode_reward)

            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(total_rewards[-10:])
                print(
                    f"Episode {episode + 1}/{n_episodes}, Avg Reward: {avg_reward:.4f}, Epsilon: {epsilon:.3f}"
                )

        return total_rewards

    def evaluate(self, df, feature_cols, return_col, use_qlearning=True):
        """
        Evaluate policy on data.
        """
        df = df.dropna(subset=[return_col]).reset_index(drop=True)
        weather_data = df[feature_cols].values
        returns = df[return_col].values

        belief = np.ones(self.n_states) / self.n_states
        position = 0
        total_reward = 0
        positions = []
        rewards = []

        for t in range(len(df) - 1):
            obs = self.get_observation(weather_data[t])

            if use_qlearning:
                action = self.qlearning_action(belief, epsilon=0)  # Greedy
            else:
                action = self.greedy_action(belief)

            new_position = action - 1

            actual_return = returns[t]
            pnl = new_position * actual_return
            transaction_cost = 0.0005 * abs(new_position - position)
            reward = pnl - transaction_cost

            next_obs = (
                self.get_observation(weather_data[t + 1]) if t + 1 < len(df) else obs
            )
            belief = self.update_belief(belief, action, next_obs)

            position = new_position
            total_reward += reward
            positions.append(new_position)
            rewards.append(reward)

        return {
            "total_reward": total_reward,
            "mean_reward": np.mean(rewards),
            "positions": positions,
            "rewards": rewards,
            "sharpe": np.mean(rewards) / (np.std(rewards) + 1e-8) * np.sqrt(252),
        }


def resample_to_weekly(df, feature_cols, return_col='ret_CL1'):
    """
    Resample daily data to weekly frequency.

    - Returns are compounded (sum of log returns or product of simple returns)
    - Weather features are averaged over the week
    """
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')

    # Aggregate returns (sum for log-like returns)
    weekly_returns = df[return_col].resample('W').sum()

    # Average weather features
    weekly_features = df[feature_cols].resample('W').mean()

    # Combine
    weekly_df = weekly_features.copy()
    weekly_df[return_col] = weekly_returns

    # Reset index
    weekly_df = weekly_df.reset_index()
    weekly_df = weekly_df.dropna()

    return weekly_df


def run_pomdp_experiment():
    """
    Run full POMDP experiment on WTI data.
    """
    # Load data with hurricanes
    print("Loading data...")
    df = pd.read_csv("pomdp_data/wti_1_data_with_hurricanes.csv")

    # Define feature columns (all weather + hurricane features)
    feature_cols = [
        c for c in df.columns if c not in ["Date", "ret_CL1", "spot_price", "CL1"]
    ]
    return_col = "ret_CL1"

    print(f"Features: {len(feature_cols)}")
    print(f"Samples: {len(df)}")

    # Split data: 80% train, 20% test
    train_size = int(len(df) * 0.8)
    train_df = df.iloc[:train_size].copy()
    test_df = df.iloc[train_size:].copy()

    print(f"Train: {len(train_df)}, Test: {len(test_df)}")

    # Create and fit POMDP
    pomdp = WTIPOMDPModel(n_hidden_states=4, n_obs_clusters=8, gamma=0.95)
    pomdp.fit(train_df, feature_cols, return_col)

    # Train Q-learning
    print("\nTraining Q-learning...")
    training_rewards = pomdp.train_qlearning(
        train_df, feature_cols, return_col, n_episodes=50
    )
    print(f"Final training reward: {training_rewards[-1]:.4f}")

    # Evaluate on test set
    print("\nEvaluating on test set...")
    results = pomdp.evaluate(test_df, feature_cols, return_col, use_qlearning=True)

    print(f"\n=== Test Results ===")
    print(f"Total Reward: {results['total_reward']:.4f}")
    print(f"Mean Daily Reward: {results['mean_reward']:.6f}")
    print(f"Sharpe Ratio (annualized): {results['sharpe']:.4f}")

    # Compare to baselines
    print("\n=== Baselines ===")

    # Buy and hold
    test_returns = test_df.dropna(subset=[return_col])[return_col].values[:-1]
    buy_hold_reward = test_returns.sum()
    buy_hold_sharpe = test_returns.mean() / (test_returns.std() + 1e-8) * np.sqrt(252)
    print(f"Buy & Hold - Total: {buy_hold_reward:.4f}, Sharpe: {buy_hold_sharpe:.4f}")

    # Random
    np.random.seed(42)
    random_positions = np.random.choice([-1, 0, 1], size=len(test_returns))
    random_reward = (random_positions * test_returns).sum()
    print(f"Random - Total: {random_reward:.4f}")

    return pomdp, results


def run_weekly_experiment():
    """
    Run POMDP experiment with weekly horizon and uniform discretization.
    """
    print("="*60)
    print("WEEKLY HORIZON POMDP EXPERIMENT")
    print("="*60)

    # Load data
    print("\nLoading data...")
    df = pd.read_csv("pomdp_data/wti_1_data_with_hurricanes.csv")

    feature_cols = [
        c for c in df.columns if c not in ["Date", "ret_CL1", "spot_price", "CL1"]
    ]
    return_col = "ret_CL1"

    # Resample to weekly
    print("Resampling to weekly frequency...")
    weekly_df = resample_to_weekly(df, feature_cols, return_col)
    print(f"Weekly samples: {len(weekly_df)}")

    # Split data
    train_size = int(len(weekly_df) * 0.8)
    train_df = weekly_df.iloc[:train_size].copy()
    test_df = weekly_df.iloc[train_size:].copy()
    print(f"Train: {len(train_df)}, Test: {len(test_df)}")

    results_all = {}

    # Test different configurations
    configs = [
        ('kmeans', 4, 8, 0.95),
        ('uniform', 4, 8, 0.95),
        ('uniform', 5, 10, 0.9),
        ('uniform', 3, 6, 0.8),
    ]

    for disc, n_states, n_obs, gamma in configs:
        print(f"\n{'='*50}")
        print(f"Config: {disc}, states={n_states}, obs={n_obs}, gamma={gamma}")
        print("="*50)

        pomdp = WTIPOMDPModel(
            n_hidden_states=n_states,
            n_obs_clusters=n_obs,
            gamma=gamma,
            discretization=disc
        )
        pomdp.fit(train_df, feature_cols, return_col)

        # Train Q-learning
        print("\nTraining Q-learning...")
        pomdp.train_qlearning(train_df, feature_cols, return_col, n_episodes=100)

        # Evaluate
        results = pomdp.evaluate(test_df, feature_cols, return_col, use_qlearning=True)

        config_name = f"{disc}_s{n_states}_o{n_obs}_g{gamma}"
        results_all[config_name] = results

        print(f"\nResults: Total={results['total_reward']:.4f}, Sharpe={results['sharpe']:.4f}")

    # Baselines
    test_returns = test_df.dropna(subset=[return_col])[return_col].values[:-1]
    buy_hold_reward = test_returns.sum()
    buy_hold_sharpe = test_returns.mean() / (test_returns.std() + 1e-8) * np.sqrt(52)  # Weekly

    np.random.seed(42)
    random_positions = np.random.choice([-1, 0, 1], size=len(test_returns))
    random_reward = (random_positions * test_returns).sum()

    # Summary
    print("\n" + "="*60)
    print("SUMMARY - WEEKLY HORIZON")
    print("="*60)
    print(f"{'Config':<35} {'Total Reward':>15} {'Sharpe':>10}")
    print("-"*60)
    for name, res in results_all.items():
        print(f"{name:<35} {res['total_reward']:>15.4f} {res['sharpe']:>10.4f}")
    print("-"*60)
    print(f"{'Buy & Hold':<35} {buy_hold_reward:>15.4f} {buy_hold_sharpe:>10.4f}")
    print(f"{'Random':<35} {random_reward:>15.4f} {'N/A':>10}")

    return results_all


def run_ovx_experiment():
    """
    Run POMDP experiment with OVX (volatility) data on weekly horizon.
    """
    print("="*60)
    print("WEEKLY POMDP WITH OVX DATA")
    print("="*60)

    # Load OVX data
    print("\nLoading OVX data...")
    df = pd.read_csv("ovx_data/wti_weather_hurricanes_ovx_since_2007.csv")
    print(f"Samples: {len(df)} (from 2007)")

    # Include OVX as a feature
    feature_cols = [
        c for c in df.columns if c not in ["Date", "ret_CL1", "spot_price", "CL1"]
    ]
    print(f"Features: {len(feature_cols)} (including OVX)")
    return_col = "ret_CL1"

    # Resample to weekly
    print("Resampling to weekly frequency...")
    weekly_df = resample_to_weekly(df, feature_cols, return_col)
    print(f"Weekly samples: {len(weekly_df)}")

    # Split data
    train_size = int(len(weekly_df) * 0.8)
    train_df = weekly_df.iloc[:train_size].copy()
    test_df = weekly_df.iloc[train_size:].copy()
    print(f"Train: {len(train_df)}, Test: {len(test_df)}")

    results_all = {}

    # Test configurations - based on what worked before
    configs = [
        ('uniform', 5, 10, 0.9),
        ('uniform', 4, 8, 0.9),
        ('uniform', 3, 6, 0.8),
        ('kmeans', 4, 8, 0.9),
    ]

    for disc, n_states, n_obs, gamma in configs:
        print(f"\n{'='*50}")
        print(f"Config: {disc}, states={n_states}, obs={n_obs}, gamma={gamma}")
        print("="*50)

        pomdp = WTIPOMDPModel(
            n_hidden_states=n_states,
            n_obs_clusters=n_obs,
            gamma=gamma,
            discretization=disc
        )
        pomdp.fit(train_df, feature_cols, return_col)

        # Train Q-learning with more episodes
        print("\nTraining Q-learning...")
        pomdp.train_qlearning(train_df, feature_cols, return_col, n_episodes=150)

        # Evaluate
        results = pomdp.evaluate(test_df, feature_cols, return_col, use_qlearning=True)

        config_name = f"{disc}_s{n_states}_o{n_obs}_g{gamma}"
        results_all[config_name] = results

        print(f"\nResults: Total={results['total_reward']:.4f}, Sharpe={results['sharpe']:.4f}")

    # Baselines
    test_returns = test_df.dropna(subset=[return_col])[return_col].values[:-1]
    buy_hold_reward = test_returns.sum()
    buy_hold_sharpe = test_returns.mean() / (test_returns.std() + 1e-8) * np.sqrt(52)

    np.random.seed(42)
    random_positions = np.random.choice([-1, 0, 1], size=len(test_returns))
    random_reward = (random_positions * test_returns).sum()

    # Summary
    print("\n" + "="*60)
    print("SUMMARY - WEEKLY HORIZON WITH OVX")
    print("="*60)
    print(f"{'Config':<35} {'Total Reward':>15} {'Sharpe':>10}")
    print("-"*60)
    for name, res in results_all.items():
        print(f"{name:<35} {res['total_reward']:>15.4f} {res['sharpe']:>10.4f}")
    print("-"*60)
    print(f"{'Buy & Hold':<35} {buy_hold_reward:>15.4f} {buy_hold_sharpe:>10.4f}")
    print(f"{'Random':<35} {random_reward:>15.4f} {'N/A':>10}")

    return results_all


if __name__ == "__main__":
    import os

    os.chdir('/Users/andrewsung/Developer/CS238_Project/src')
    results = run_ovx_experiment()
