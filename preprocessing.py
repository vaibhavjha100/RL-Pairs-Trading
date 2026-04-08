import os
import sys
import pickle
import pandas as pd
import numpy as np

try:
    import ta
except ImportError:
    import subprocess
    print("ta library not found. Installing now...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ta"])
    import ta

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
except ImportError:
    import subprocess
    print("scikit-learn not found. Installing now...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn"])
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

try:
    import skfuzzy as fuzz
except ImportError:
    import subprocess
    print("scikit-fuzzy not found. Installing now...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-fuzzy"])
    import skfuzzy as fuzz

# ============================================================================
# Part 1: Trading data preprocessing
# ============================================================================

def calc_trading_indicators(trading_sub_df):
    """
    Safely calculates technical indicators for a given trading ticker dataframe.
    Wrapped in try-except blocks to ensure failures don't crash the script.
    Uses the 'ta' package.
    """
    # Create copies of series to feed into indicators to avoid warnings
    trading_close = trading_sub_df['Close']
    trading_high = trading_sub_df['High']
    trading_low = trading_sub_df['Low']
    trading_volume = trading_sub_df['Volume'] if 'Volume' in trading_sub_df.columns else pd.Series(0, index=trading_sub_df.index)

    # Helper format to isolate failures
    def safe_assign(col_name, indicator_class, method_name, **kwargs):
        try:
            ind = indicator_class(**kwargs)
            trading_sub_df[col_name] = getattr(ind, method_name)()
        except Exception:
            trading_sub_df[col_name] = np.nan

    # Trend Indicators
    try:
        from ta.trend import SMAIndicator, EMAIndicator, MACD, ADXIndicator, CCIIndicator
        safe_assign('SMA_20', SMAIndicator, 'sma_indicator', close=trading_close, window=20)
        safe_assign('EMA_20', EMAIndicator, 'ema_indicator', close=trading_close, window=20)
        
        # MACD
        try:
            macd = MACD(close=trading_close)
            trading_sub_df['MACD'] = macd.macd()
            trading_sub_df['MACD_signal'] = macd.macd_signal()
            trading_sub_df['MACD_diff'] = macd.macd_diff()
        except Exception:
            trading_sub_df['MACD'] = np.nan
            trading_sub_df['MACD_signal'] = np.nan
            trading_sub_df['MACD_diff'] = np.nan
            
        # ADX
        try:
            adx = ADXIndicator(high=trading_high, low=trading_low, close=trading_close)
            trading_sub_df['ADX'] = adx.adx()
            trading_sub_df['ADX_pos'] = adx.adx_pos()
            trading_sub_df['ADX_neg'] = adx.adx_neg()
        except Exception:
            trading_sub_df['ADX'] = np.nan
            trading_sub_df['ADX_pos'] = np.nan
            trading_sub_df['ADX_neg'] = np.nan
            
        safe_assign('CCI_14', CCIIndicator, 'cci', high=trading_high, low=trading_low, close=trading_close, window=14)
    except Exception:
        pass

    # Momentum Indicators
    try:
        from ta.momentum import RSIIndicator, ROCIndicator, StochasticOscillator, WilliamsRIndicator
        safe_assign('RSI_14', RSIIndicator, 'rsi', close=trading_close, window=14)
        safe_assign('ROC_10', ROCIndicator, 'roc', close=trading_close, window=10)
        
        # Stochastic
        try:
            stoch = StochasticOscillator(high=trading_high, low=trading_low, close=trading_close)
            trading_sub_df['STOCH_k'] = stoch.stoch()
            trading_sub_df['STOCH_d'] = stoch.stoch_signal()
        except Exception:
            trading_sub_df['STOCH_k'] = np.nan
            trading_sub_df['STOCH_d'] = np.nan
            
        safe_assign('WILLR_14', WilliamsRIndicator, 'williams_r', high=trading_high, low=trading_low, close=trading_close, lbp=14)
    except Exception:
        pass

    # Volatility Indicators
    try:
        from ta.volatility import BollingerBands, AverageTrueRange
        try:
            bb = BollingerBands(close=trading_close, window=20, window_dev=2)
            trading_sub_df['BBL_20_2.0'] = bb.bollinger_lband()
            trading_sub_df['BBM_20_2.0'] = bb.bollinger_mavg()
            trading_sub_df['BBU_20_2.0'] = bb.bollinger_hband()
            trading_sub_df['BBB_20_2.0'] = bb.bollinger_wband()
        except Exception:
            pass
            
        safe_assign('ATR_14', AverageTrueRange, 'average_true_range', high=trading_high, low=trading_low, close=trading_close, window=14)
    except Exception:
        pass

    # Volume Indicators
    try:
        from ta.volume import OnBalanceVolumeIndicator
        safe_assign('OBV', OnBalanceVolumeIndicator, 'on_balance_volume', close=trading_close, volume=trading_volume)
    except Exception:
        pass

    return trading_sub_df


# ============================================================================
# Part 2: Spread data preprocessing
# ============================================================================

def calc_spread_candlestick_features(spread_sub_df):
    spread_sub_df = spread_sub_df.copy()
    spread_sub_df["upper_shadow"] = spread_sub_df["high_spread"] - spread_sub_df[["open_spread", "spread"]].max(axis=1)
    spread_sub_df["lower_shadow"] = spread_sub_df[["open_spread", "spread"]].min(axis=1) - spread_sub_df["low_spread"]
    spread_sub_df["body_length"] = (spread_sub_df["spread"] - spread_sub_df["open_spread"]).abs()
    spread_sub_df["body_color"] = np.where(
        spread_sub_df["spread"] > spread_sub_df["open_spread"],
        1,
        np.where(spread_sub_df["spread"] < spread_sub_df["open_spread"], -1, 0)
    )
    return spread_sub_df


def _safe_talib_call(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception:
        return None


def calc_spread_talib_indicators(spread_sub_df):
    # Lazy import so Part 1 can still run when TA-Lib is unavailable.
    try:
        import talib
    except ImportError:
        import subprocess
        print("TA-Lib not found. Installing now...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "TA-Lib"])
            import talib
        except Exception as e:
            raise ImportError(
                "TA-Lib is required for spread preprocessing and could not be installed automatically. "
                "Please install it manually and re-run."
            ) from e

    df = spread_sub_df.copy()
    o = df["open_spread"].astype(float).values
    h = df["high_spread"].astype(float).values
    l = df["low_spread"].astype(float).values
    c = df["spread"].astype(float).values
    v = df["volume_spread"].astype(float).values

    # Overlap studies
    out = _safe_talib_call(talib.BBANDS, c)
    if out is not None:
        df["BBANDS_upper"], df["BBANDS_middle"], df["BBANDS_lower"] = out
    for name, fn, args in [
        ("DEMA", talib.DEMA, (c,)),
        ("EMA", talib.EMA, (c,)),
        ("HT_TRENDLINE", talib.HT_TRENDLINE, (c,)),
        ("KAMA", talib.KAMA, (c,)),
        ("MA", talib.MA, (c,)),
        ("MIDPOINT", talib.MIDPOINT, (c,)),
        ("MIDPRICE", talib.MIDPRICE, (h, l)),
        ("SAR", talib.SAR, (h, l)),
        ("SAREXT", talib.SAREXT, (h, l)),
        ("SMA", talib.SMA, (c,)),
        ("T3", talib.T3, (c,)),
        ("TEMA", talib.TEMA, (c,)),
        ("TRIMA", talib.TRIMA, (c,)),
        ("WMA", talib.WMA, (c,)),
    ]:
        res = _safe_talib_call(fn, *args)
        if res is not None:
            df[name] = res
    out = _safe_talib_call(talib.MAMA, c)
    if out is not None:
        df["MAMA"], df["FAMA"] = out

    # Momentum indicators
    for name, fn, args in [
        ("ADX", talib.ADX, (h, l, c)),
        ("ADXR", talib.ADXR, (h, l, c)),
        ("APO", talib.APO, (c,)),
        ("AROONOSC", talib.AROONOSC, (h, l)),
        ("BOP", talib.BOP, (o, h, l, c)),
        ("CCI", talib.CCI, (h, l, c)),
        ("CMO", talib.CMO, (c,)),
        ("DX", talib.DX, (h, l, c)),
        ("MFI", talib.MFI, (h, l, c, v)),
        ("MINUS_DI", talib.MINUS_DI, (h, l, c)),
        ("MINUS_DM", talib.MINUS_DM, (h, l)),
        ("MOM", talib.MOM, (c,)),
        ("PLUS_DI", talib.PLUS_DI, (h, l, c)),
        ("PLUS_DM", talib.PLUS_DM, (h, l)),
        ("PPO", talib.PPO, (c,)),
        ("ROC", talib.ROC, (c,)),
        ("ROCP", talib.ROCP, (c,)),
        ("ROCR", talib.ROCR, (c,)),
        ("RSI", talib.RSI, (c,)),
        ("TRIX", talib.TRIX, (c,)),
        ("ULTOSC", talib.ULTOSC, (h, l, c)),
        ("WILLR", talib.WILLR, (h, l, c)),
    ]:
        res = _safe_talib_call(fn, *args)
        if res is not None:
            df[name] = res
    out = _safe_talib_call(talib.AROON, h, l)
    if out is not None:
        df["AROON_down"], df["AROON_up"] = out
    out = _safe_talib_call(talib.MACD, c)
    if out is not None:
        df["MACD"], df["MACD_signal"], df["MACD_hist"] = out
    out = _safe_talib_call(talib.MACDEXT, c)
    if out is not None:
        df["MACDEXT"], df["MACDEXT_signal"], df["MACDEXT_hist"] = out
    out = _safe_talib_call(talib.MACDFIX, c)
    if out is not None:
        df["MACDFIX"], df["MACDFIX_signal"], df["MACDFIX_hist"] = out
    out = _safe_talib_call(talib.STOCH, h, l, c)
    if out is not None:
        df["STOCH_k"], df["STOCH_d"] = out
    out = _safe_talib_call(talib.STOCHF, h, l, c)
    if out is not None:
        df["STOCHF_k"], df["STOCHF_d"] = out
    out = _safe_talib_call(talib.STOCHRSI, c)
    if out is not None:
        df["STOCHRSI_k"], df["STOCHRSI_d"] = out

    # Volume indicators
    for name, fn, args in [
        ("AD", talib.AD, (h, l, c, v)),
        ("ADOSC", talib.ADOSC, (h, l, c, v)),
        ("OBV", talib.OBV, (c, v)),
    ]:
        res = _safe_talib_call(fn, *args)
        if res is not None:
            df[name] = res

    # Volatility indicators
    for name, fn, args in [
        ("ATR", talib.ATR, (h, l, c)),
        ("NATR", talib.NATR, (h, l, c)),
        ("TRANGE", talib.TRANGE, (h, l, c)),
    ]:
        res = _safe_talib_call(fn, *args)
        if res is not None:
            df[name] = res

    return df


def build_spread_sequences(feature_df, feature_cols, lookback=60):
    x_list = []
    y_list = []
    meta = []

    for pair, pair_df in feature_df.groupby("Pair"):
        pair_df = pair_df.sort_values("Date").reset_index(drop=True)
        x_raw = pair_df[feature_cols].values
        y_raw = pair_df["spread"].values
        d_raw = pair_df["Date"].values

        for i in range(lookback, len(pair_df)):
            x_list.append(x_raw[i - lookback:i, :])
            y_list.append(y_raw[i])
            meta.append((pair, pd.Timestamp(d_raw[i]), i))

    if x_list:
        x = np.asarray(x_list, dtype=float)
    else:
        x = np.empty((0, lookback, len(feature_cols)), dtype=float)
    y = np.asarray(y_list, dtype=float)
    meta_df = pd.DataFrame(meta, columns=["Pair", "target_date", "pair_row_idx"])
    return x, y, meta_df


def run_part2_spread_pipeline():
    print("\nPart 2: Loading spread raw data...")
    spread_dir = os.path.join("data", "spread")
    pickle_dir = os.path.join("data", "pickle")
    os.makedirs(spread_dir, exist_ok=True)
    os.makedirs(pickle_dir, exist_ok=True)

    spread_raw_path = os.path.join(spread_dir, "raw.csv")
    if not os.path.exists(spread_raw_path):
        print(f"Error: {spread_raw_path} not found. Please run spread.py first.")
        sys.exit(1)

    spread_df = pd.read_csv(spread_raw_path, parse_dates=["Date"])
    if "Unnamed: 0" in spread_df.columns:
        spread_df = spread_df.drop(columns=["Unnamed: 0"])

    required_cols = ["Date", "Pair", "spread", "open_spread", "high_spread", "low_spread", "volume_spread"]
    missing_cols = [c for c in required_cols if c not in spread_df.columns]
    if missing_cols:
        print(f"Error: Missing required spread columns: {missing_cols}")
        sys.exit(1)

    for col in ["spread", "open_spread", "high_spread", "low_spread", "volume_spread"]:
        spread_df[col] = pd.to_numeric(spread_df[col], errors="coerce")

    spread_df = spread_df.sort_values(["Pair", "Date"]).reset_index(drop=True)

    print("Part 2: Engineering candlestick + TA-Lib indicators...")
    spread_frames = []
    for _, pair_df in spread_df.groupby("Pair", sort=False):
        pair_df = calc_spread_candlestick_features(pair_df)
        pair_df = calc_spread_talib_indicators(pair_df)
        spread_frames.append(pair_df)
    spread_features = pd.concat(spread_frames, ignore_index=True)

    numeric_cols = spread_features.select_dtypes(include=[np.number]).columns.tolist()
    fill_cols = [c for c in numeric_cols if c != "body_color"]
    spread_features[fill_cols] = (
        spread_features.groupby("Pair")[fill_cols].transform(lambda x: x.ffill().bfill())
    )
    spread_features[fill_cols] = spread_features[fill_cols].replace([np.inf, -np.inf], np.nan)
    spread_features[fill_cols] = spread_features[fill_cols].fillna(spread_features[fill_cols].median())

    spread_preprocessed_path = os.path.join(spread_dir, "preprocessed.csv")
    spread_features.to_csv(spread_preprocessed_path, index=False)

    print("Part 2: Z-score scaling + fixed PCA(8)...")
    id_cols = ["Date", "Pair"]
    target_col = "spread"
    feature_cols = [c for c in spread_features.columns if c not in id_cols]
    continuous_cols = [c for c in feature_cols if c != "body_color"]

    scaler = StandardScaler()
    if not continuous_cols:
        print("Error: No continuous spread features available for scaling/PCA.")
        return

    x_scaled = scaler.fit_transform(spread_features[continuous_cols].values)

    fixed_pca_components = 8
    fixed_fcm_clusters = 10

    if len(spread_features) < 2:
        print("Error: Not enough spread rows for PCA/fuzzy c-means.")
        return

    if x_scaled.shape[1] < fixed_pca_components:
        print(
            f"Error: Need at least {fixed_pca_components} continuous features for fixed PCA; "
            f"found {x_scaled.shape[1]}."
        )
        return

    pca = PCA(n_components=fixed_pca_components)
    x_pca = pca.fit_transform(x_scaled)

    pca_cols = [f"PC{i + 1}" for i in range(fixed_pca_components)]
    pca_df = pd.DataFrame(x_pca, columns=pca_cols)
    pca_df["Date"] = spread_features["Date"].values
    pca_df["Pair"] = spread_features["Pair"].values
    pca_df["body_color"] = spread_features["body_color"].values
    pca_df[target_col] = spread_features[target_col].values
    pca_df = pca_df[["Date", "Pair"] + pca_cols + ["body_color", target_col]]

    pca_features_path = os.path.join(spread_dir, "pca_features.csv")
    pca_df.to_csv(pca_features_path, index=False)

    print("Part 2: Fixed fuzzy c-means clustering (single 8D PCA space, c=10)...")
    if len(pca_df) < fixed_fcm_clusters:
        print(
            f"Error: Need at least {fixed_fcm_clusters} rows for fixed fuzzy c-means; "
            f"found {len(pca_df)}."
        )
        return

    pca_clustered = pca_df.copy()
    x_fcm = pca_df[pca_cols].values.T  # scikit-fuzzy requires shape (features, samples)
    centers, memberships, _, _, _, _, _ = fuzz.cluster.cmeans(
        x_fcm, c=fixed_fcm_clusters, m=2.0, error=0.005, maxiter=1000, init=None
    )
    assigned = np.argmax(memberships, axis=0)
    replaced = centers[assigned, :]
    for i, col in enumerate(pca_cols):
        pca_clustered[col] = replaced[:, i]
    pca_clustered["body_color"] = pca_df["body_color"].values
    pca_clustered[target_col] = pca_df[target_col].values

    pca_clustered_path = os.path.join(spread_dir, "pca_clustered.csv")
    pca_clustered.to_csv(pca_clustered_path, index=False)

    print("Part 2: Building sequence datasets...")
    sequence_feature_cols = [c for c in pca_clustered.columns if c not in ["Date", "Pair", target_col]]
    # Include lagged spread as an input feature by design.
    sequence_feature_cols.append(target_col)

    x, y, meta_df = build_spread_sequences(
        pca_clustered[["Date", "Pair"] + sequence_feature_cols].copy(),
        feature_cols=sequence_feature_cols,
        lookback=60
    )

    split_date = pd.to_datetime("2024-01-01")
    train_mask = meta_df["target_date"] < split_date
    test_mask = ~train_mask

    x_train, y_train = x[train_mask.values], y[train_mask.values]
    x_test, y_test = x[test_mask.values], y[test_mask.values]

    meta_df["split"] = np.where(train_mask, "train", "test")
    sequence_meta_path = os.path.join(spread_dir, "sequence_xy.csv")
    meta_df.to_csv(sequence_meta_path, index=False)

    with open(os.path.join(pickle_dir, "spread_scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(pickle_dir, "spread_pca.pkl"), "wb") as f:
        pickle.dump(pca, f)
    with open(os.path.join(pickle_dir, "spread_fcm.pkl"), "wb") as f:
        pickle.dump(
            {
                "clusters": fixed_fcm_clusters,
                "centers": centers,
                "memberships": memberships
            },
            f
        )
    with open(os.path.join(pickle_dir, "spread_X_train.pkl"), "wb") as f:
        pickle.dump(x_train, f)
    with open(os.path.join(pickle_dir, "spread_y_train.pkl"), "wb") as f:
        pickle.dump(y_train, f)
    with open(os.path.join(pickle_dir, "spread_X_test.pkl"), "wb") as f:
        pickle.dump(x_test, f)
    with open(os.path.join(pickle_dir, "spread_y_test.pkl"), "wb") as f:
        pickle.dump(y_test, f)

    print("\nPart 2 (Spread) Summary:")
    print(f"Spread rows input                : {len(spread_df)}")
    print(f"Engineered feature columns       : {len(spread_features.columns)}")
    print(f"PCA components retained          : {len(pca_cols)}")
    print(f"Fuzzy c-means clusters used      : {fixed_fcm_clusters}")
    print(f"Sequence samples total           : {len(meta_df)}")
    print(f"Sequence train samples           : {len(x_train)}")
    print(f"Sequence test samples            : {len(x_test)}")
    print(f"Saved {spread_preprocessed_path}")
    print(f"Saved {pca_features_path}")
    print(f"Saved {pca_clustered_path}")
    print(f"Saved {sequence_meta_path}")

def main():
    print("Part 1: Loading trading raw data...")
    trading_dir = os.path.join('data', 'trading')
    os.makedirs(trading_dir, exist_ok=True)

    trading_raw_path = os.path.join(trading_dir, 'raw.csv')
    if not os.path.exists(trading_raw_path):
        print(f"Error: {trading_raw_path} not found. Please run collection.py first.")
        sys.exit(1)
        
    trading_df = pd.read_csv(trading_raw_path, parse_dates=['Date'])
    
    trading_dropped_tickers = []
    
    print("Part 1: Processing trading tickers and computing indicators...")
    trading_grouped = trading_df.groupby('Ticker')
    
    trading_price_cols = [c for c in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'] if c in trading_df.columns]
    
    trading_preprocessed_path = os.path.join(trading_dir, 'preprocessed.csv')
    trading_train_path = os.path.join(trading_dir, 'train.csv')
    trading_test_path = os.path.join(trading_dir, 'test.csv')

    trading_prep_file = open(trading_preprocessed_path, 'w')
    trading_train_file = open(trading_train_path, 'w')
    trading_test_file = open(trading_test_path, 'w')
    
    trading_is_first = True
    trading_surviving_tickers = 0
    trading_train_count = 0
    trading_test_count = 0
    
    for trading_ticker, trading_group in trading_grouped:
        trading_group = trading_group.sort_values('Date').reset_index(drop=True)
        
        trading_first_valid_idx = trading_group['Close'].first_valid_index()
        if trading_first_valid_idx is None:
            trading_dropped_tickers.append({"Ticker": trading_ticker, "Reason": "No valid Close prices found."})
            continue
            
        trading_real_start_date = trading_group.loc[trading_first_valid_idx, 'Date']
        
        trading_group = trading_group.iloc[trading_first_valid_idx:].copy()
        trading_group[trading_price_cols] = trading_group[trading_price_cols].ffill()
        trading_group = calc_trading_indicators(trading_group)
        
        # Split logic
        trading_split_date = pd.to_datetime('2024-01-01')
        trading_train_group = trading_group[trading_group['Date'] < trading_split_date]
        trading_test_group = trading_group[trading_group['Date'] >= trading_split_date]
        
        # Append to files
        trading_group.to_csv(trading_prep_file, header=trading_is_first, index=False)
        if not trading_train_group.empty:
            trading_train_group.to_csv(trading_train_file, header=trading_is_first, index=False)
            trading_train_count += len(trading_train_group)
        if not trading_test_group.empty:
            trading_test_group.to_csv(trading_test_file, header=trading_is_first, index=False)
            trading_test_count += len(trading_test_group)
            
        trading_surviving_tickers += 1
        trading_is_first = False
        
    trading_prep_file.close()
    trading_train_file.close()
    trading_test_file.close()
        
    print("\n" + "="*50)
    print(" PART 1 (TRADING) - DROPPED TICKERS")
    print("="*50)
    for d in trading_dropped_tickers:
        print(f"{d['Ticker']:<15} | {d['Reason']}")
    if not trading_dropped_tickers:
        print("No tickers were dropped.")
    print("="*50)
    
    if trading_surviving_tickers == 0:
        print("\nError: No trading tickers survived preprocessing!")
        sys.exit(1)
        
    print(f"\nSaved {trading_preprocessed_path}")
    print(f"Saved {trading_train_path} (Rows: {trading_train_count})")
    print(f"Saved {trading_test_path} (Rows: {trading_test_count})")
    
    print("\nPart 1 (Trading) Summary:")
    print(f"Total initial tickers      : {len(trading_grouped)}")
    print(f"Total dropped tickers      : {len(trading_dropped_tickers)}")
    print(f"Total surviving tickers    : {trading_surviving_tickers}")
    print(f"Train samples (<2024-01)   : {trading_train_count}")
    print(f"Test samples (>=2024-01)   : {trading_test_count}")

    spread_raw_path = os.path.join("data", "spread", "raw.csv")
    if os.path.exists(spread_raw_path):
        run_part2_spread_pipeline()
    else:
        print(f"\nPart 2 (Spread) skipped: '{spread_raw_path}' not found.")

if __name__ == "__main__":
    main()
