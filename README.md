# MPHDRL

MPHDRL is a reinforcement learning project for pairs trading research.  
The repository provides an end-to-end pipeline for data collection, preprocessing, pair/spread construction, model training, backtesting, formal comparison, and market-neutral analysis.

## Installation and Setup

1. Clone the repository and enter the project directory.
2. Create and activate a virtual environment.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

3. Install the project and dependencies from `pyproject.toml`.

```powershell
pip install --upgrade pip
pip install -e .
```

4. Download the Nifty 500 stock list from NSE and place it in `data/trading` as `ind_nifty500list.csv` (required for collection).

- Source: `https://www.nseindia.com/static/products-services/indices-nifty500-index`
- Required path: `data/trading/ind_nifty500list.csv`

## Usage

Run the full pipeline:

```powershell
rl-pairs-pipeline
```

Helpful options:

```powershell
rl-pairs-pipeline --list-steps
rl-pairs-pipeline --dry-run
rl-pairs-pipeline --skip-eda
```

You can also run key stages directly:

```powershell
rl-pairs-train
rl-pairs-backtest
```

## Requirements and Dependencies

- Python `>=3.10`
- Core dependencies are managed in `pyproject.toml` and include:
  - `captum`
  - `matplotlib`
  - `numpy`
  - `pandas`
  - `scikit-fuzzy`
  - `scikit-learn`
  - `scipy`
  - `seaborn`
  - `statsmodels`
  - `ta`
  - `TA-Lib`
  - `torch`
  - `yfinance`

