import os
import sys
import pickle
import logging
import pandas as pd
import yfinance as yf

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_downloaded_tickers(data):
    """Return tickers with at least one non-NaN close value."""
    if data is None or data.empty:
        return set()

    if isinstance(data.columns, pd.MultiIndex):
        if 'Close' in data.columns.get_level_values(0):
            close_cols = data['Close']
        else:
            return set()
        return set(close_cols.dropna(how='all', axis=1).columns.tolist())

    if 'Close' in data.columns and not data['Close'].dropna().empty:
        return {None}
    return set()

def main():
    trading_dir = os.path.join('data', 'trading')
    pickle_dir = os.path.join('data', 'pickle')
    os.makedirs(trading_dir, exist_ok=True)
    os.makedirs(pickle_dir, exist_ok=True)

    csv_file = os.path.join(trading_dir, 'ind_nifty500list.csv')
    
    # 1. Read NSE 500 CSV
    if not os.path.exists(csv_file):
        logging.error(f"'{csv_file}' not found in the current directory.")
        sys.exit(1)
        
    try:
        df_nifty = pd.read_csv(csv_file)
        logging.info(f"Successfully read {len(df_nifty)} rows from {csv_file}")
    except Exception as e:
        logging.error(f"Error reading {csv_file}: {e}")
        sys.exit(1)
        
    # 2. Convert symbols to .NS format
    if 'Symbol' not in df_nifty.columns:
        logging.error("Column 'Symbol' not found in the CSV.")
        sys.exit(1)
        
    df_nifty['Yahoo_Symbol'] = df_nifty['Symbol'].astype(str).str.strip() + '.NS'
    symbols = [s for s in df_nifty['Yahoo_Symbol'].tolist() if s and s != '.NS']
    symbols = list(dict.fromkeys(symbols))  # preserve order + deduplicate
    
    # 3. Extract sector/industry to dict
    # Using the 'Industry' column as per NSE 500 structure
    if 'Industry' in df_nifty.columns:
        sector_dict = df_nifty.set_index('Yahoo_Symbol')['Industry'].to_dict()
    else:
        sector_dict = {sym: 'Unknown' for sym in symbols}
        logging.warning("Column 'Industry' not found. Using 'Unknown' for all symbols.")
        
    # Save dict to data/pickle/sector.pkl
    pkl_path = os.path.join(pickle_dir, 'sector.pkl')
    try:
        with open(pkl_path, 'wb') as f:
            pickle.dump(sector_dict, f)
        logging.info(f"Saved sector dictionary to {pkl_path}")
    except Exception as e:
        logging.error(f"Failed to save dictionary to {pkl_path}: {e}")
        
    # 4. Use yfinance to download daily history from 2015 through 2026
    # yfinance `end` is exclusive, so use 2027-01-01 to include 2026-12-31.
    start_date = "2015-01-01"
    end_date = "2027-01-01"
    logging.info(
        f"Downloading historical daily data for {len(symbols)} tickers "
        f"(start='{start_date}', end='{end_date}', interval='1d')..."
    )
    try:
        data = yf.download(
            symbols,
            start=start_date,
            end=end_date,
            interval="1d",
            auto_adjust=False,
            group_by="column",
            threads=True
        )
    except Exception as e:
        logging.error(f"Error during yfinance download: {e}")
        sys.exit(1)
        
    # 5. Handle errors gracefully and log failures
    # yfinance shares errors via shared._ERRORS for versions < 0.2.33 or properties.
    downloaded_tickers = extract_downloaded_tickers(data)

    # If data came back in single-ticker format, map success to the requested symbol.
    if downloaded_tickers == {None} and len(symbols) == 1:
        downloaded_tickers = {symbols[0]}
    elif downloaded_tickers == {None}:
        downloaded_tickers = set()

    failures = [sym for sym in symbols if sym not in downloaded_tickers]

    # Retry failed tickers once, serially, to reduce transient API/network misses.
    if failures:
        logging.info(f"Retrying {len(failures)} failed tickers one-by-one...")
        recovered = []
        still_failed = []
        for sym in failures:
            try:
                retry_data = yf.download(
                    sym,
                    start=start_date,
                    end=end_date,
                    interval="1d",
                    auto_adjust=False,
                    group_by="column",
                    threads=False
                )
                retry_ok = False
                if not retry_data.empty and 'Close' in retry_data.columns:
                    retry_ok = not retry_data['Close'].dropna().empty
                if retry_ok:
                    recovered.append(sym)
                else:
                    still_failed.append(sym)
            except Exception:
                still_failed.append(sym)

        if recovered:
            logging.info(f"Recovered {len(recovered)} tickers on retry.")
            try:
                recovered_data = yf.download(
                    recovered,
                    start=start_date,
                    end=end_date,
                    interval="1d",
                    auto_adjust=False,
                    group_by="column",
                    threads=True
                )
                if not recovered_data.empty:
                    data = data.combine_first(recovered_data)
            except Exception as e:
                logging.warning(f"Failed to merge recovered ticker data: {e}")

        failures = still_failed
        
    if failures:
        logging.warning(f"Failed to download data for {len(failures)} tickers.")
        # log failure list internally
        logging.info(f"Failed tickers: {failures}")
        
    # 6. Convert to long format
    logging.info("Converting data to long format...")
    try:
        if isinstance(data.columns, pd.MultiIndex):
            # yfinance default creates columns (Price, Ticker)
            data.columns.names = ['Price', 'Ticker']
            # Stack the Ticker level to create a long format DataFrame
            # rename_axis to ensure index names before reset
            import inspect
            if 'future_stack' in inspect.signature(data.stack).parameters:
                long_data = data.stack(level='Ticker', future_stack=True)
            else:
                long_data = data.stack(level='Ticker')
            long_data = long_data.reset_index()
        else:
            # Handles edge case if there is only 1 ticker successfully downloaded
            long_data = data.reset_index()
            # If there's barely anything, columns might just be Date, Open, Close...
            long_data['Ticker'] = symbols[0]
            
        # Ensure standard column names
        if 'index' in long_data.columns:
            long_data = long_data.rename(columns={'index': 'Date'})
            
        long_data = long_data.sort_values(by=['Ticker', 'Date']).reset_index(drop=True)
        
    except Exception as e:
        logging.error(f"Error reshaping data: {e}")
        sys.exit(1)
        
    # Save to data/trading/raw.csv
    csv_out_path = os.path.join(trading_dir, 'raw.csv')
    try:
        long_data.to_csv(csv_out_path, index=False)
        logging.info(f"Saved long format data to {csv_out_path}")
    except Exception as e:
        logging.error(f"Failed to save CSV {csv_out_path}: {e}")

    # 7. Print summary stats, date range, failures list, and dict sample
    print("\n" + "="*50)
    print(" SUMMARY STATISTICS")
    print("="*50)
    
    if 'Date' in long_data.columns and not long_data.empty:
        min_date = long_data['Date'].min()
        max_date = long_data['Date'].max()
        # Convert to string if it's a timestamp
        print(f"Date Range              : {str(min_date).split(' ')[0]} to {str(max_date).split(' ')[0]}")
    else:
        print("Date Range              : Not available (empty data)")
        
    print(f"Total Symbols Given     : {len(symbols)}")
    print(f"Total Symbols Downloaded: {len(long_data['Ticker'].unique()) if 'Ticker' in long_data.columns else (len(symbols) - len(failures))}")
    print(f"Data Points Extracted   : {len(long_data)}")
    print(f"Total Download Failures : {len(failures)}")
    
    if failures:
        print("\nFailures List:")
        # Print up to 10 failures so output isn't massive, or print all depending on preference
        for fail in failures:
            print(f" - {fail}")
            
    print("\nSector Dictionary Sample (first 5):")
    for k, v in list(sector_dict.items())[:5]:
        print(f"  {k}: {v}")
        
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
