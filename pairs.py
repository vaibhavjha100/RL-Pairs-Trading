import os
import sys
import pickle
import pandas as pd
import numpy as np

# Ensure dependencies are installed
try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score
except ImportError:
    print("scikit-learn not found. Installing now...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn"])
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score

try:
    from statsmodels.tsa.stattools import coint
except ImportError:
    print("statsmodels not found. Installing now...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "statsmodels"])
    from statsmodels.tsa.stattools import coint

# ==============================================================================
# PART 1: CLUSTERING (KMeans Pre-Selection)
# ==============================================================================

def engineer_features(df, sector_dict):
    """
    Transforms row-level train data into stock-level feature vectors.
    """
    print("Engineering stock-level features...")
    
    # 1. Determine daily returns if missing
    if 'Return' not in df.columns:
        price_col = 'Adj Close' if 'Adj Close' in df.columns and df['Adj Close'].notna().any() else 'Close'
        # Sort values to ensure chronological pct_change
        df = df.sort_values(['Ticker', 'Date'])
        df['Return'] = df.groupby('Ticker')[price_col].pct_change()
    
    grouped = df.groupby('Ticker')
    
    # Feature 1 & 2: Annualized Return & Volatility
    mean_ret = grouped['Return'].mean() * 252
    volatility = grouped['Return'].std() * np.sqrt(252)
    
    # Feature 3 & 4: Skewness & Kurtosis
    skewness = grouped['Return'].skew()
    kurtosis = grouped['Return'].apply(lambda x: x.kurt())
    
    # Feature 5: Volume (Log-transformed to handle extreme scale)
    if 'Volume' in df.columns:
        avg_volume = np.log1p(grouped['Volume'].mean())
    else:
        avg_volume = pd.Series(0, index=mean_ret.index)
        
    # Feature 6: Autocorrelation (Lag 1)
    autocorr = grouped['Return'].apply(lambda x: x.autocorr(lag=1))
    
    # Feature 7: Completeness Ratio
    global_dates = df['Date'].nunique()
    completeness = grouped['Date'].count() / global_dates
    
    # Consolidate into a feature matrix
    features = pd.DataFrame({
        'Mean_Return_Ann': mean_ret,
        'Volatility_Ann': volatility,
        'Skewness': skewness,
        'Kurtosis': kurtosis,
        'Log_Avg_Volume': avg_volume,
        'Autocorr_Lag1': autocorr,
        'Completeness_Ratio': completeness
    })
    
    # Impute missing generated metrics (e.g. kurtosis on flat periods)
    # Fill with median across all stocks to prevent distortion
    features = features.fillna(features.median()).fillna(0)
    
    # Integrate sector metadata
    features['Sector'] = features.index.map(sector_dict)
    features['Sector'] = features['Sector'].fillna('Unknown')
    
    # One-hot encode the categorical variables (dtype=int stops StandardScaler Warnings)
    features = pd.get_dummies(features, columns=['Sector'], drop_first=False, dtype=int)
    
    return features

def select_best_k(scaled_features, k_range=range(5, 31), random_state=42):
    """
    Evaluates KMeans over a range and scores based on Silhouette and Balance constraints.
    """
    print(f"Evaluating candidate cluster numbers from {min(k_range)} to {max(k_range)}...")
    best_k = None
    best_score = -np.inf
    best_labels = None
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(scaled_features)
        
        # Base metric: Silhouette Score measures cohesion & separation
        sil_score = silhouette_score(scaled_features, labels)
        
        # Secondary objective: Penalize fragmented, tiny clusters (size < 4)
        # Pairs trading needs pools of candidates. Clusters of 1 or 2 are risky/useless.
        unique, counts = np.unique(labels, return_counts=True)
        tiny_clusters = np.sum(counts < 4)
        
        # Composite objective: -0.05 penalty arbitrarily anchors tiny cluster impact 
        # relative to common silhouette score fluctuations (~0.01 - 0.20 span)
        composite_score = sil_score - (0.05 * tiny_clusters)
        
        if composite_score > best_score:
            best_score = composite_score
            best_k = k
            best_labels = labels
            
    print(f"Optimal k selected: {best_k} (Composite Score: {best_score:.4f})")
    u, c = np.unique(best_labels, return_counts=True)
    
    # Create an ordered string report of sizes
    size_report = ", ".join([f"C{cluster_id}:{count}" for cluster_id, count in zip(u, c)])
    print(f"Cluster distributions -> {size_report}")
    
    return best_k, best_labels

def preselect_clusters(train_path="data/trading/train.csv", sector_path="data/pickle/sector.pkl", out_dir="data/pickle"):
    """
    Core execution routine for Part 1. Generates and saves cluster mappings.
    """
    print(f"\n--- PART 1: K-MEANS PAIR PRESELECTION ---")
    
    print(f"Loading '{train_path}' and metadata...")
    df = pd.read_csv(train_path, parse_dates=['Date'])
    
    with open(sector_path, 'rb') as f:
        sector_dict = pickle.load(f)
        
    features = engineer_features(df, sector_dict)
    print(f"Generated feature matrix with shape {features.shape}.")
    
    print("Scaling numeric distributions...")
    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(features)
    scaled_features = pd.DataFrame(scaled_array, index=features.index, columns=features.columns)
    
    best_k, best_labels = select_best_k(scaled_features, k_range=range(5, 31))
    
    # Compile the final dictionary { cluster_label : tuple_of_tickers }
    scaled_features['Cluster'] = best_labels
    cluster_dict = {}
    for cluster_id, group in scaled_features.groupby('Cluster'):
        cluster_dict[cluster_id] = tuple(group.index.tolist())
        
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'kmeans_clusters.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump(cluster_dict, f)
        
    print(f"K-Means complete! Cluster mappings successfully saved to '{out_path}'.\n")
    return cluster_dict

# ==============================================================================
# PART 2: INTRA-CLUSTER CORRELATION EVALUATION
# ==============================================================================

def find_intra_cluster_pairs(cluster_dict, train_df, corr_threshold=0.7):
    """
    Computes pairwise intra-cluster correlations on daily returns.
    Returns and saves a list of unique, highly correlated stock pairs.
    """
    print(f"\n--- PART 2: INTRA-CLUSTER CORRELATION EVALUATION ---")
    
    # Ensure Return is present
    if 'Return' not in train_df.columns:
        price_col = 'Adj Close' if 'Adj Close' in train_df.columns and train_df['Adj Close'].notna().any() else 'Close'
        train_df = train_df.sort_values(['Ticker', 'Date'])
        train_df['Return'] = train_df.groupby('Ticker')[price_col].pct_change()
        
    print("Pivoting returns to wide format for correlation analysis...")
    wide_returns = train_df.pivot(index='Date', columns='Ticker', values='Return')
    
    selected_pairs = []
    total_candidates_checked = 0
    
    for cluster_id, tickers in cluster_dict.items():
        tickers = list(tickers)
        if len(tickers) < 2:
            continue
            
        cluster_rets = wide_returns[tickers]
        
        # Correlation matrix
        corr_matrix = cluster_rets.corr()
        
        # Calculate pairwise overlapping observation counts dynamically
        notna_matrix = cluster_rets.notna().astype(int)
        overlap_counts = notna_matrix.T.dot(notna_matrix)
        
        # Extract native numpy arrays to drastically speed up element search
        corr_np = corr_matrix.values
        overlap_np = overlap_counts.values
        ticker_list = corr_matrix.columns.tolist()
        
        # Retrieve upper triangle indices to strictly avoid duplicates & self-pairs
        r, c = np.triu_indices_from(corr_np, k=1)
        
        for idx in range(len(r)):
            total_candidates_checked += 1
            i, j = r[idx], c[idx]
            
            overlap = overlap_np[i, j]
            corr = corr_np[i, j]
            
            # Avoid nan correlation instances resulting from 0 variance windows
            if pd.notna(corr) and pd.notna(overlap):
                if corr > corr_threshold:
                    # Enforce alphabetically sorted order (A, B)
                    t1, t2 = sorted([ticker_list[i], ticker_list[j]])
                    selected_pairs.append({
                        'pair': (t1, t2),
                        'correlation': corr,
                        'overlap': overlap,
                        'cluster': cluster_id
                    })
                    
    # Primary sort by descending correlation coefficient
    selected_pairs.sort(key=lambda x: x['correlation'], reverse=True)
    
    print("\nAnalysis Complete!")
    print(f"Clusters evaluated       : {len(cluster_dict)}")
    print(f"Raw candidate pairs      : {total_candidates_checked}")
    print(f"Pairs > {corr_threshold} corr    : {len(selected_pairs)}")
    
    if selected_pairs:
        print(f"\nTop High-Correlation Pairs:")
        # Print top 5 securely
        for idx, item in enumerate(selected_pairs[:5]):
            p1, p2 = item['pair']
            print(f"  {idx+1}. {p1} & {p2} (Corr: {item['correlation']:.4f}, Co-Obs: {item['overlap']})")
    else:
        print("\nNo pairs met the correlation threshold.")
        
    out_dir = 'data/pickle'
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'correlated_pairs.pkl')
    
    with open(out_path, 'wb') as f:
        pickle.dump(selected_pairs, f)
        
    print(f"\nSaved {len(selected_pairs)} correlated pairs to '{out_path}'.")
    return selected_pairs

# ==============================================================================
# PART 3: ROLLING COINTEGRATION TESTING
# ==============================================================================

def test_cointegration(candidate_pairs, train_df, pvalue_threshold=0.05, out_dir='data/pickle'):
    """
    Runs full-train-period Engle-Granger cointegration testing on candidate pairs.
    Tests both directions and keeps the direction with lower p-value.
    Returns a list of tuple pairs in directional order: (lhs_ticker, rhs_ticker).
    """
    print(f"\n--- PART 3: FULL-PERIOD COINTEGRATION EVALUATION ---")

    price_col = 'Adj Close' if 'Adj Close' in train_df.columns and train_df['Adj Close'].notna().any() else 'Close'
    print(f"Using '{price_col}' for full-period price-level extraction...")

    wide_prices = train_df.pivot(index='Date', columns='Ticker', values=price_col)

    cointegrated_pairs = []
    total_candidates = len(candidate_pairs)
    tested_pairs = 0
    skipped_pairs = 0

    for pair_info in candidate_pairs:
        t1, t2 = pair_info['pair']

        # Pairwise overlap across the full training period
        pair_data = wide_prices[[t1, t2]].dropna()
        if pair_data.empty:
            skipped_pairs += 1
            continue

        y1 = pair_data[t1].values
        y2 = pair_data[t2].values

        # Avoid degenerate tests on flat series
        if np.std(y1) == 0 or np.std(y2) == 0:
            skipped_pairs += 1
            continue

        tested_pairs += 1

        # Directional Engle-Granger: keep lower p-value direction
        _, pval_t1_t2, _ = coint(y1, y2, trend='ct')
        _, pval_t2_t1, _ = coint(y2, y1, trend='ct')

        if pval_t1_t2 <= pval_t2_t1:
            best_pair = (t1, t2)
            best_pval = pval_t1_t2
        else:
            best_pair = (t2, t1)
            best_pval = pval_t2_t1

        if pd.notna(best_pval) and best_pval < pvalue_threshold:
            cointegrated_pairs.append(best_pair)

    # Remove accidental duplicates while preserving order
    cointegrated_pairs = list(dict.fromkeys(cointegrated_pairs))

    print("\nCointegration Analysis Complete!")
    print(f"Candidate pairs received   : {total_candidates}")
    print(f"Pairs tested              : {tested_pairs}")
    print(f"Pairs skipped             : {skipped_pairs}")
    print(f"Cointegrated pairs found  : {len(cointegrated_pairs)}")

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'cointegrated_pairs.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump(cointegrated_pairs, f)

    print(f"Saved cointegrated pairs to '{out_path}'.")
    return cointegrated_pairs

def main():
    # PART 1: KMeans Preselection
    clusters = preselect_clusters(
        train_path='data/trading/train.csv',
        sector_path='data/pickle/sector.pkl',
        out_dir='data/pickle'
    )
    
    # PART 2: Intra-Cluster Correlation Screening
    train_df = pd.read_csv('data/trading/train.csv', parse_dates=['Date'])
    correlated_pairs = find_intra_cluster_pairs(clusters, train_df, corr_threshold=0.7)
    
    # PART 3: Rolling Cointegration
    test_cointegration(correlated_pairs, train_df)

if __name__ == "__main__":
    main()
