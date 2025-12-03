"""
Create two versions of the WTI datasets:
1. Log returns (drops NaN rows from negative prices)
2. Simple returns (keeps all rows, handles negative prices)

This allows comparing model performance on the April 2020 negative price event.
"""

import pandas as pd
import numpy as np
import os

# Paths
DATA_DIR = "/Users/ethanhersch/Documents/Documents/Stanford/CS238/FinalProject/CS238_Project/data"
SRC_DIR = "/Users/ethanhersch/Documents/Documents/Stanford/CS238/FinalProject/CS238_Project/src"


def load_base_data(contract_type):
    """Load and merge spot + futures + weather data."""

    # Oil data
    spot = pd.read_csv(f"{DATA_DIR}/oil_data/Cushing_OK_WTI_Spot_Price_FOB_sorted.csv")
    futures = pd.read_csv(
        f"{DATA_DIR}/oil_data/NYMEX_futures/Cushing_OK_Crude_Oil_Future_Contract_{contract_type}.csv"
    )

    df = pd.merge(spot, futures, on="Date", how="inner")
    df = df.rename(columns={"Price_y": "CL1", "Price_x": "spot_price"})

    # Pipeline weather
    pipeline_locations = [
        "alaska",
        "bakken",
        "western_canada",
        "midwest_corridor",
        "ok_hub",
        "permian",
        "rockies",
        "texas_gulf",
    ]
    for loc in pipeline_locations:
        loc_df = pd.read_csv(f"{DATA_DIR}/processed_data/incoming/{loc}_incoming.csv")
        loc_df = loc_df.rename(
            columns={"mean_temp_K": f"{loc}_temp", "total_precip_mm": f"{loc}_precip"}
        )
        df = pd.merge(df, loc_df, on="Date", how="left")

    # Producing region weather
    producing_locations = [
        "bakken",
        "canada",
        "eagle",
        "central_kansas",
        "n_alaska",
        "nc_ok",
        "ne_ok",
        "nw_ok",
        "permian",
        "s_alaska",
        "saskatchewan",
        "se_kansas",
        "sw_kansas",
    ]
    for loc in producing_locations:
        loc_df = pd.read_csv(
            f"{DATA_DIR}/processed_data/producing_locations/{loc}_producing.csv"
        )
        loc_df = loc_df.rename(
            columns={
                "mean_temp_K": f"{loc}_temp_producing",
                "total_precip_mm": f"{loc}_precip_producing",
            }
        )
        df = pd.merge(df, loc_df, on="Date", how="left")

    # Hurricane data
    hurricane_df = pd.read_csv(
        f"{DATA_DIR}/weather_data/producing_locations/hurdat2_processed_cleaned.csv"
    )
    hurricane_df = hurricane_df.rename(columns={"date": "Date"})
    df = pd.merge(df, hurricane_df, on="Date", how="left")

    # Fill NaN hurricane values with defaults
    hurricane_cols = [c for c in hurricane_df.columns if c != "Date"]
    for col in hurricane_cols:
        if col == "maximum_sustained_wind_knots":
            df[col] = df[col].fillna(0)
        elif col == "central_pressure_mb":
            df[col] = df[col].fillna(1013.25)  # standard pressure
        else:
            df[col] = df[col].fillna(0)

    return df


def create_log_returns_dataset(df):
    """
    Create dataset with log returns.
    NaN rows (from negative prices) will need to be dropped during training.
    """
    df = df.copy()
    df["ret_CL1"] = np.log(df["CL1"].shift(-1) / df["CL1"])
    return df


def create_simple_returns_dataset(df):
    """
    Create dataset with simple returns.
    Handles negative prices by using absolute value in denominator.

    Simple return = (P_{t+1} - P_t) / |P_t|

    This preserves the sign of the return while handling negative prices.
    """
    df = df.copy()

    # Simple return: (next_price - current_price) / abs(current_price)
    df["ret_CL1"] = (df["CL1"].shift(-1) - df["CL1"]) / df["CL1"].abs()

    return df


def analyze_negative_price_period(df_log, df_simple):
    """Analyze the April 2020 negative price period."""

    print("\n" + "=" * 60)
    print("APRIL 2020 NEGATIVE PRICE ANALYSIS")
    print("=" * 60)

    # Find the negative price period
    mask = (df_log["Date"] >= "2020-04-15") & (df_log["Date"] <= "2020-04-25")

    print("\nLog Returns (original):")
    print(df_log.loc[mask, ["Date", "spot_price", "CL1", "ret_CL1"]].to_string())

    print("\nSimple Returns (new):")
    print(df_simple.loc[mask, ["Date", "spot_price", "CL1", "ret_CL1"]].to_string())

    # Count NaN
    log_nan = df_log["ret_CL1"].isna().sum()
    simple_nan = df_simple["ret_CL1"].isna().sum()

    print(f"\nNaN count - Log returns: {log_nan}, Simple returns: {simple_nan}")

    # The key insight: what would trading on April 20, 2020 have done?
    print("\n" + "-" * 60)
    print("KEY TRADING INSIGHT:")
    print("-" * 60)

    apr20_idx = df_simple[df_simple["Date"] == "2020-04-20"].index[0]
    apr20_return = df_simple.loc[apr20_idx, "ret_CL1"]

    print(f"April 20, 2020: Price went from -$37.63 to +$10.01")
    print(f"Simple return: {apr20_return:.2%}")
    print(
        f"If you went LONG on April 20: reward = +1 * {apr20_return:.4f} = {apr20_return:.4f}"
    )
    print(
        f"If you went SHORT on April 20: reward = -1 * {apr20_return:.4f} = {-apr20_return:.4f}"
    )
    print("\nThis is a MASSIVE move that the simple returns model can learn from!")


def main():
    print("Creating datasets with both return types...")
    print("=" * 60)

    for contract in range(1, 5):
        print(f"\nProcessing Contract {contract}...")

        # Load base data
        df = load_base_data(contract)

        # Create both versions
        df_log = create_log_returns_dataset(df)
        df_simple = create_simple_returns_dataset(df)

        # Save log returns version
        log_path = f"{SRC_DIR}/wti_{contract}_logreturns.csv"
        df_log.to_csv(log_path, index=False)
        print(f"  Saved: {log_path}")
        print(
            f"    Shape: {df_log.shape}, NaN in ret_CL1: {df_log['ret_CL1'].isna().sum()}"
        )

        # Save simple returns version
        simple_path = f"{SRC_DIR}/wti_{contract}_simplereturns.csv"
        df_simple.to_csv(simple_path, index=False)
        print(f"  Saved: {simple_path}")
        print(
            f"    Shape: {df_simple.shape}, NaN in ret_CL1: {df_simple['ret_CL1'].isna().sum()}"
        )

    # Detailed analysis of the negative price period
    df = load_base_data(1)
    df_log = create_log_returns_dataset(df)
    df_simple = create_simple_returns_dataset(df)
    analyze_negative_price_period(df_log, df_simple)

    print("\n" + "=" * 60)
    print("USAGE:")
    print("=" * 60)
    print(
        """
# For log returns (drops NaN automatically):
df = pd.read_csv('wti_1_logreturns.csv')
env = WTIEnv(df, feature_cols, 'ret_CL1', normalize=True)

# For simple returns (keeps all rows):
df = pd.read_csv('wti_1_simplereturns.csv')
env = WTIEnv(df, feature_cols, 'ret_CL1', normalize=True)

# For POMDP:
pomdp = WTIPOMDPModel()
pomdp.fit(df, feature_cols, 'ret_CL1')
"""
    )


if __name__ == "__main__":
    os.chdir(SRC_DIR)
    main()
