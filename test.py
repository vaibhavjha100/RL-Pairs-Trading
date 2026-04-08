import pandas as pd
import pickle

# Load cointegrated pairs list from pickle file
c_pairs_path = 'data/pickle/cointegrated_pairs.pkl'
with open(c_pairs_path, 'rb') as f:
    cointegrated_pairs = pickle.load(f)

# Print the cointegrated pairs
print(cointegrated_pairs)

# Load spread raw data from csv file
spread_raw_path = 'data/spread/raw.csv'
spread_raw_df = pd.read_csv(spread_raw_path)

# Print the spread raw data information
print(spread_raw_df.info())

# Print the spread raw data
print(spread_raw_df.head())

# Load the hedge ratios from pickle file
hedge_ratios_path = 'data/pickle/hedge_ratios.pkl'
with open(hedge_ratios_path, 'rb') as f:
    hedge_ratios = pickle.load(f)

# Print the hedge ratios
print(hedge_ratios)