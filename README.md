# Oil Futures Trading with Weather Data  
**CS238: Decision Making Under Uncertainty — Stanford University**  
**Authors:** Andrew Sung · Ethan Hersch · Jerry Yin  

This repository contains the code, data pipeline, and reinforcement learning models for our CS238 project on **trading WTI crude oil futures using weather, volatility, and price data**. We evaluate both **POMDP-based belief-state Q-learning** and **Deep Q-Network (DQN)** methods, and study how decision frequency (daily vs weekly vs monthly) affects strategy performance in this highly noisy commodity market.

📄 **Project Paper:** *Oil Futures Trading with Weather Data: an Evaluation of POMDP and Deep Q-Learning Approaches*  


## Overview

Crude oil prices are influenced by global supply/demand conditions, transportation constraints, and weather-driven disruptions (pipeline freezing, hurricanes, etc.).  
We investigate whether **weather (ERA5), OVX volatility, and futures structure** provide useful signals for sequential decision-making agents.

We compare three approaches:

- **POMDP with belief-state Q-learning**  
- **Deep Q-Network (DQN)** with 58-dimensional feature inputs  
- **Baselines:** Buy-and-Hold, 10-year WTI long-only Sharpe

Our key finding:

> **Daily trading is dominated by noise and models underperform.  
> Weekly/monthly horizons yield strong Sharpe ratios, showing weather effects operate on slower timescales.**

---

## Data Structure

### Weather Data (`data/weather_data/`)
- `incoming_pipeline/` — Temperature & precipitation along major pipeline corridors (1986–2025)  
- `producing_locations/` — Same coverage for major oil-producing regions  

These regions correspond to production and transportation corridors relevant for WTI (Cushing, OK).

### Oil & Volatility Data (`data/oil_data/`)
- `Cushing_OK_WTI_Spot_Price_FOB_sorted.csv` — WTI spot price (1986–2024)  
- `NYMEX_futures/` — CL1–CL4 front-month futures (start years 1983–1985)  
- OVX volatility index — Available starting May 2007  

Processed merged datasets live in `data/processed_data/`.

**Note:** Weather is currently filtered to *trading days only*; including non-trading days is a potential extension.

---

## Model Performance

Daily models evaluated 2016–2024; weekly/monthly models evaluated 2021–2024.

| Strategy                     | Horizon              | Total Return (%) | Sharpe Ratio |
|-----------------------------|----------------------|------------------|--------------|
| DQN Agent                   | Daily                | 44.9             | 0.128        |
| POMDP (K-means)             | Daily                | 31.2             | 0.089        |
| Buy & Hold                  | Daily                | 137.3            | 0.386        |
| **POMDP + OVX (Uniform)**   | **Weekly**           | **32**           | **1.343**    |
| Buy & Hold                  | Weekly               | 70.6             | 0.561        |
| **POMDP + OVX (Uniform)**   | **Monthly**          | **43.2**         | **2.231**    |
| POMDP + OVX (K-means)       | Monthly              | 5.4              | 0.370        |
| Buy & Hold                  | Monthly              | 63.1             | 0.608        |
| General WTI Baseline        | 10-year Performance  | 39               | 0.08         |

### Key Takeaways
- Daily trading = **too noisy** to extract meaningful weather signal.  
- Weekly & monthly horizons = **4–10× higher Sharpe ratio** than buy-and-hold.  
- OVX significantly improves regime inference.  

---

### File Organization
```CS238_oil_exploration.ipynb``` contains code from a Colab notebook used to gather weather data from Google Earth Engine.

```\src``` contains the POMDP and DQN models.
- ```\src\pomdp_data``` contains the merged datasets for all CL1-CL4 contracts with and without weather data, using simple and log returns. These are the csv files used for evaluation of the POMDP model.
- ```\src\ovx_data``` contains the similar data to ```\src\pomdp_data``` but with the addition of OVX information and this is the data used for the performant POMDP model.

```\data``` contains folders with the data we collected. 
- ```\data\processed_data``` is the processed weather weather data for hurricanes, producing, and transportation regions.
- ```data\oil_data``` contains the CL1-CL4 and oil spot price datasets.
- ```data\ovx_data.csv``` is the volatility data since 2007.