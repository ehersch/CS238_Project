# CS238_Project

See CS238_oil_exploration.ipynb for the code to retrieve 2m temp and daily total precipitation from the ERA5 daily aggregate dataset

Inside the data folder, there are two directories
- data/weather_data/incoming_pipeline includes data from 01-01-1986 to 11-01-2025 along the major pipelines
- data/weather_data/producing_locations includes data from the same time period in major oil producing regions

These are all major pipelines and major oil-producing regions for WTI crude oil. This is what the WTI (Cushing, OK) spot price is based on.

Inside data/oil_data is the information for oil spot prices and futures prices (WTI Cushing OK)
- Cushing_OK_WTI_Spot_Price_FOB_sorted.csv contains the spot prices of Cushing OK crude (daily) sorted from 1986 to 2024.
- NYMEX_futures contains the futures prices for the 1,2,3,4 month futures. They are also sorted like the spot prices.

Note, the 1 and 3 month options start 1983 and the 2 and 4 month options start 01-02-1985
- All go until 2024-04-05