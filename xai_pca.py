from __future__ import annotations

import argparse
import os
import pickle
import sys
from typing import List

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Interpret saved PCA components using original spread features."
    )
    p.add_argument(
        "--pca-path",
        type=str,
        default=os.path.join("data", "pickle", "spread_pca.pkl"),
        help="Path to fitted PCA pickle.",
    )
    p.add_argument(
        "--preprocessed-path",
        type=str,
        default=os.path.join("data", "spread", "preprocessed.csv"),
        help="Path to spread preprocessed CSV used before PCA fitting.",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Top absolute loading features to print per component.",
    )
    p.add_argument(
        "--save-csv",
        type=str,
        default=os.path.join("data", "backtest", "results", "xai_pca_loadings.csv"),
        help="Optional path to save long-form loadings summary CSV.",
    )
    return p.parse_args()


def _require_file(path: str, label: str) -> None:
    if not os.path.isfile(path):
        print(f"Missing {label}: {path}", file=sys.stderr)
        sys.exit(1)


def get_continuous_feature_names(preprocessed_path: str) -> List[str]:
    df = pd.read_csv(preprocessed_path, nrows=1)
    id_cols = {"Date", "Pair"}
    feature_cols = [c for c in df.columns if c not in id_cols]
    continuous_cols = [c for c in feature_cols if c != "body_color"]
    if not continuous_cols:
        print("No continuous spread features found in preprocessed CSV.", file=sys.stderr)
        sys.exit(1)
    return continuous_cols


def main() -> None:
    args = parse_args()
    _require_file(args.pca_path, "PCA pickle")
    _require_file(args.preprocessed_path, "spread preprocessed CSV")

    with open(args.pca_path, "rb") as f:
        pca = pickle.load(f)

    if not hasattr(pca, "components_"):
        print(f"PCA object at {args.pca_path} has no components_.", file=sys.stderr)
        sys.exit(1)

    feature_names = get_continuous_feature_names(args.preprocessed_path)
    components = np.asarray(pca.components_, dtype=np.float64)
    n_components, n_features = components.shape
    if n_features != len(feature_names):
        print(
            f"Feature mismatch: PCA has {n_features} columns, preprocessed has {len(feature_names)} continuous columns.",
            file=sys.stderr,
        )
        sys.exit(1)

    evr = np.asarray(getattr(pca, "explained_variance_ratio_", np.full(n_components, np.nan)), dtype=np.float64)
    top_k = max(1, int(args.top_k))

    print("=" * 88)
    print("PCA COMPONENT INTERPRETATION (top absolute loadings)")
    print("=" * 88)
    print(f"PCA path:          {os.path.abspath(args.pca_path)}")
    print(f"Preprocessed path: {os.path.abspath(args.preprocessed_path)}")
    print(f"n_components:      {n_components}")
    print(f"n_features:        {n_features}")
    print()

    rows = []
    for i in range(n_components):
        pc_name = f"PC{i + 1}"
        vec = components[i]
        idx_sorted = np.argsort(np.abs(vec))[::-1][:top_k]
        evr_i = float(evr[i]) if i < len(evr) else float("nan")
        print(f"{pc_name}  (explained_variance_ratio={evr_i:.6f})")
        print("-" * 88)
        print(f"{'rank':>4}  {'feature':<38}  {'loading':>12}  {'abs_loading':>12}")
        for rank, j in enumerate(idx_sorted, start=1):
            feat = feature_names[j]
            loading = float(vec[j])
            abs_loading = abs(loading)
            print(f"{rank:>4}  {feat:<38}  {loading:>+12.6f}  {abs_loading:>12.6f}")
            rows.append(
                {
                    "component": pc_name,
                    "explained_variance_ratio": evr_i,
                    "rank_by_abs_loading": rank,
                    "feature": feat,
                    "loading": loading,
                    "abs_loading": abs_loading,
                }
            )
        print()

    out_csv = args.save_csv.strip()
    if out_csv:
        out_dir = os.path.dirname(out_csv)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        print(f"Saved loadings summary: {os.path.abspath(out_csv)}")


if __name__ == "__main__":
    main()
