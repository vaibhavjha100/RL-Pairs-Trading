"""
Goal: Build spread time series and hedge-ratio artifacts from selected pairs.

Inputs: Cointegrated pairs, trading datasets, and required historical price windows.

Processing: Calculates hedge ratios, computes pair spreads, and structures spread tables.

Outputs: Spread raw dataset and hedge-ratio pickles under data/spread and data/pickle.
"""

import os
import pickle
import numpy as np
import pandas as pd


PAIR_KEY_SEPARATOR = "|"
SPREAD_FIELDS = ["Open", "High", "Low", "Close", "Volume"]


def load_inputs():
    pairs_path = os.path.join("data", "pickle", "cointegrated_pairs.pkl")
    raw_path = os.path.join("data", "trading", "raw.csv")
    train_path = os.path.join("data", "trading", "train.csv")

    with open(pairs_path, "rb") as f:
        cointegrated_pairs = pickle.load(f)

    raw_df = pd.read_csv(raw_path, parse_dates=["Date"]).sort_values(["Date", "Ticker"]).reset_index(drop=True)
    train_df = pd.read_csv(train_path, parse_dates=["Date"]).sort_values(["Date", "Ticker"]).reset_index(drop=True)
    return cointegrated_pairs, raw_df, train_df


def compute_beta_with_intercept(log_a, log_b):
    x = np.asarray(log_b, dtype=float)
    y = np.asarray(log_a, dtype=float)
    design = np.column_stack([np.ones(len(x)), x])
    coeffs, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    # coeffs[0] is intercept(alpha), coeffs[1] is hedge ratio(beta)
    return float(coeffs[1])


def compute_hedge_ratios(cointegrated_pairs, train_df):
    close_wide_train = train_df.pivot(index="Date", columns="Ticker", values="Close")
    hedge_ratios = {}

    for pair in cointegrated_pairs:
        ticker_a, ticker_b = pair
        pair_close = close_wide_train[[ticker_a, ticker_b]].dropna()

        log_a = np.log(pair_close[ticker_a].astype(float))
        log_b = np.log(pair_close[ticker_b].astype(float))
        beta = compute_beta_with_intercept(log_a, log_b)
        hedge_ratios[(ticker_a, ticker_b)] = beta

    return hedge_ratios


def build_spread_raw_df(cointegrated_pairs, hedge_ratios, raw_df):
    field_wide = {
        field: raw_df.pivot(index="Date", columns="Ticker", values=field).sort_index()
        for field in SPREAD_FIELDS
    }

    spread_frames = []
    for pair in cointegrated_pairs:
        ticker_a, ticker_b = pair
        pair_key = f"{ticker_a}{PAIR_KEY_SEPARATOR}{ticker_b}"
        beta = hedge_ratios[(ticker_a, ticker_b)]
        pair_df = pd.DataFrame(index=field_wide["Close"].index)
        pair_df.index.name = "Date"

        for field in SPREAD_FIELDS:
            series_a = field_wide[field][ticker_a].astype(float)
            series_b = field_wide[field][ticker_b].astype(float)
            spread_series = np.log(series_a) - (beta * np.log(series_b))

            if field == "Close":
                out_col = "spread"
            else:
                out_col = f"{field.lower()}_spread"
            pair_df[out_col] = spread_series

        pair_df = pair_df.reset_index()
        pair_df["Pair"] = pair_key
        spread_frames.append(pair_df)

    spread_raw_df = pd.concat(spread_frames, ignore_index=True)
    spread_raw_df = spread_raw_df.sort_values(["Pair", "Date"]).reset_index(drop=True)
    ordered_cols = ["Date", "Pair", "spread", "open_spread", "high_spread", "low_spread", "volume_spread"]
    spread_raw_df = spread_raw_df[ordered_cols]
    return spread_raw_df


def main():
    cointegrated_pairs, raw_df, train_df = load_inputs()
    hedge_ratios = compute_hedge_ratios(cointegrated_pairs, train_df)
    spread_raw_df = build_spread_raw_df(cointegrated_pairs, hedge_ratios, raw_df)

    os.makedirs(os.path.join("data", "pickle"), exist_ok=True)
    os.makedirs(os.path.join("data", "spread"), exist_ok=True)

    hedge_path = os.path.join("data", "pickle", "hedge_ratios.pkl")
    with open(hedge_path, "wb") as f:
        pickle.dump(hedge_ratios, f)

    spread_raw_path = os.path.join("data", "spread", "raw.csv")
    spread_raw_df.to_csv(spread_raw_path)

    print("\n" + "=" * 50)
    print("SPREAD MODULE SUMMARY")
    print("=" * 50)
    print(f"Input cointegrated pairs : {len(cointegrated_pairs)}")
    print(f"Hedge ratios computed    : {len(hedge_ratios)}")
    print(f"spread_raw shape         : {spread_raw_df.shape}")
    print(f"Saved hedge ratios       : {hedge_path}")
    print(f"Saved spread raw data    : {spread_raw_path}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
