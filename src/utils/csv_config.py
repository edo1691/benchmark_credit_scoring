from typing import Dict, Any, Tuple
import pandas as pd
from pathlib import Path
import re

DEFAULT_CSV_CONFIG: Dict[str, Dict[str, Any]] = {
    "german": {
        "sep": r"\s+",
        "engine": "python",
        "header": None,
        "encoding": "utf-8",
        "auto_header": True,
    },
    "australian": {"sep": ",", "header": 0, "encoding": "utf-8"},
    "japan": {"sep": ",", "header": 0, "encoding": "utf-8"},
    "kg_ccdea": {"sep": ",", "header": 0, "encoding": "utf-8"},
    "kg_gmsc": {"sep": ",", "header": 0, "encoding": "utf-8"},
    "taiwan": {"sep": ",", "header": 0, "encoding": "utf-8"},
    "th02": {"sep": ",", "header": 0, "encoding": "utf-8"},

    # ✅ generic format for imbalanced / derived datasets you generate
    "_derived": {"sep": ",", "header": 0, "encoding": "utf-8"},
}


def get_csv_params(name: str) -> Dict[str, Any]:
    key = name.lower()
    if key in DEFAULT_CSV_CONFIG:
        return DEFAULT_CSV_CONFIG[key].copy()
    raise KeyError(f"No CSV config found for dataset '{name}'")


def _resolve_dataset_name_and_kind(name: str) -> Tuple[str, bool]:
    """
    Resolves 'german_3' -> ('german', True) meaning derived.
    Resolves 'german'   -> ('german', False) meaning raw.

    Derived rule: <base>_<integer> where <base> exists in DEFAULT_CSV_CONFIG.
    """
    key = name.lower()

    # exact match => raw dataset
    if key in DEFAULT_CSV_CONFIG and key != "_derived":
        return key, False

    # check suffix pattern
    m = re.match(r"^(.*)_(\d+)$", key)
    if m:
        base = m.group(1)
        if base in DEFAULT_CSV_CONFIG and base != "_derived":
            return base, True

    # not known: treat as raw name (will raise KeyError later)
    return key, False


def read_csv_by_name(path: str, name: str, **kwargs) -> pd.DataFrame:
    """
    Read raw datasets using dataset-specific config & post-processing.
    Read derived datasets (e.g. german_3) using '_derived' config and
    skip raw post-processing.
    """
    base_name, is_derived = _resolve_dataset_name_and_kind(name)

    if is_derived:
        params = get_csv_params("_derived")
    else:
        params = get_csv_params(base_name)

    auto_header = params.pop("auto_header", False)
    params.update(kwargs)

    df = pd.read_csv(path, **params)

    if auto_header:
        df.columns = [f"ft_{i+1}" for i in range(len(df.columns))]

    # ✅ dataset-specific post-processing only for RAW files
    if (not is_derived) and base_name == "german":
        if not df.empty:
            last_col = df.columns[-1]
            df["target"] = df[last_col].map({1: 0, 2: 1})
            if last_col != "target":
                df = df.drop(columns=[last_col])

    return df


def create_imbalanced_datasets(
    dataset_name: str,
    data_dir: Path,
    imbalance_levels: list = None,
    target_col: str = "target",
    random_state: int = 123,
):
    if imbalance_levels is None:
        imbalance_levels = [1, 2, 3, 5, 10, 15, 20]

    dataset_path = data_dir / "input" / f"{dataset_name}.csv"
    if not dataset_path.exists():
        raise FileNotFoundError(f"File not found: {dataset_path}")

    df = read_csv_by_name(str(dataset_path), dataset_name)

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset")

    df = df.copy()
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df = df[df[target_col].isin([0, 1])].copy()
    df[target_col] = df[target_col].astype(int)

    df_pos = df[df[target_col] == 1]
    df_neg = df[df[target_col] == 0]

    n_total_original = len(df)
    n_pos_original = len(df_pos)
    n_neg_original = len(df_neg)
    original_rate = float(df[target_col].mean())

    print("==========================================")
    print(f"Original dataset: {dataset_name}")
    print(f"Total samples: {n_total_original}")
    print(f"Positive samples: {n_pos_original}")
    print(f"Negative samples: {n_neg_original}")
    print(f"Positive rate: {original_rate:.4f}")
    print("==========================================\n")

    for pct in imbalance_levels:
        desired_ratio = pct / 100.0
        n_pos_required = int((desired_ratio * n_neg_original) / (1 - desired_ratio))

        if n_pos_required <= 0:
            print(f"Skipping {pct}% — would require 0 positives.")
            continue
        if n_pos_required > n_pos_original:
            print(f"Skipping {pct}% — not enough positive samples.")
            continue

        df_pos_sampled = df_pos.sample(n=n_pos_required, random_state=random_state, replace=False)

        df_new = pd.concat([df_neg, df_pos_sampled]).sample(
            frac=1, random_state=random_state
        ).reset_index(drop=True)

        n_total_new = len(df_new)
        n_pos_new = int(df_new[target_col].sum())
        new_rate = float(df_new[target_col].mean())

        save_path = data_dir / "intermediate" / f"{dataset_name}" / f"{dataset_name}_{pct}.csv"

        # ✅ save as derived format that read_csv_by_name will parse with "_derived"
        df_new.to_csv(save_path, index=False, sep=",")

        print(f"{dataset_name}_{pct}.csv | Total={n_total_new} | Positives={n_pos_new} | Rate={new_rate:.4f}")

    print("\nFinished generating imbalanced datasets.")