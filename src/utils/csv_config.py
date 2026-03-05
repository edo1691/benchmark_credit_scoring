from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
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
    "australian": {
        "sep": r"\s+",
        "engine": "python",
        "header": None,
        "auto_header": True,
    },
    "japan": {"sep": ",",
        "engine": "python",
        "header": None,
        "encoding": "utf-8",
        "auto_header": True,
    },
    "kgccdea": {"sep": ",", "header": 0, "encoding": "utf-8"},
    "kggmsc": {"sep": ",", "header": 0, "encoding": "utf-8"},
    "taiwan": {"sep": ";", "header": 0, "encoding": "utf-8"},
    "th02": {"sep": ";", "header": 0, "encoding": "utf-8"},

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


def _get_base_dataset_name(name: str) -> str:
    """Return dataset base name (before '_')."""
    return name.split("_")[0].lower()


def read_csv_by_name(path: str, name: str, **kwargs) -> pd.DataFrame:
    """
    Read datasets using dataset-specific config & post-processing.
    Names like german_1, taiwan_99, etc. use the same config as german, taiwan.
    """
    
    base_name = _get_base_dataset_name(name)

    params = get_csv_params(base_name)

    auto_header = params.pop("auto_header", False)
    params.update(kwargs)

    df = pd.read_csv(path, **params)

    if auto_header:
        df.columns = [f"ft_{i+1}" for i in range(len(df.columns))]

    # dataset-specific post-processing
    if base_name == "german":
        if not df.empty:
            last_col = df.columns[-1]
            df["target"] = df[last_col].map({1: 0, 2: 1})
            if last_col != "target":
                df = df.drop(columns=[last_col])

    elif base_name == "australian":
        if not df.empty:
            last_col = df.columns[-1]
            df["target"] = df[last_col].map({0: 0, 1: 1})
            if last_col != "target":
                df = df.drop(columns=[last_col])

    elif base_name == "japan":
        if not df.empty:
            last_col = df.columns[-1]
            df["target"] = df[last_col].map({"+": 1, "-": 0})
            if last_col != "target":
                df = df.drop(columns=[last_col])

    elif base_name == "kgccdea":
        if not df.empty:
            last_col = "card"
            df["target"] = df[last_col].map({"yes": 0, "no": 1})
            if last_col != "target":
                df = df.drop(columns=[last_col])

    elif base_name == "kggmsc":
        if not df.empty:
            df = df.iloc[:, 1:].copy()
            last_col = "SeriousDlqin2yrs"
            df["target"] = df[last_col].map({0: 0, 1: 1})
            if last_col != "target":
                df = df.drop(columns=[last_col])

    elif base_name == "taiwan":
        if not df.empty:
            df = df.iloc[:, 1:].copy()
            last_col = "default payment next month"
            df["target"] = df[last_col].map({0: 0, 1: 1})
            if last_col != "target":
                df = df.drop(columns=[last_col])

    elif base_name == "th02":
        if not df.empty:
            last_col = "BAD"
            df["target"] = df[last_col].map({0: 0, 1: 1})
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
    """
    Create derived datasets with a desired positive rate (pct/100).

    Preference:
      1) Keep as many positives as possible -> downsample negatives first.
      2) If not achievable, keep all negatives -> downsample positives.
    """

    if imbalance_levels is None:
        imbalance_levels = [1, 2, 3, 5, 10, 15, 20]

    dataset_path = data_dir / "input" / f"{dataset_name}.csv"
    if not dataset_path.exists():
        raise FileNotFoundError(f"File not found: {dataset_path}")

    df = read_csv_by_name(str(dataset_path), dataset_name)

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset")

    # Clean + ensure binary {0,1}
    df = df.copy()
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df = df[df[target_col].isin([0, 1])].copy()
    df[target_col] = df[target_col].astype(int)

    df_pos = df[df[target_col] == 1]
    df_neg = df[df[target_col] == 0]

    n_pos = len(df_pos)
    n_neg = len(df_neg)
    n_total = len(df)
    original_rate = float(df[target_col].mean()) if n_total else 0.0

    print("==========================================")
    print(f"Original dataset: {dataset_name}")
    print(f"Total samples: {n_total}")
    print(f"Positive samples: {n_pos}")
    print(f"Negative samples: {n_neg}")
    print(f"Positive rate: {original_rate:.4f}")
    print("==========================================\n")

    save_dir = data_dir / "intermediate" / dataset_name
    save_dir.mkdir(parents=True, exist_ok=True)

    for pct in imbalance_levels:
        r = pct / 100.0

        # guardrails
        if r <= 0 or r >= 1:
            print(f"[SKIP] {pct}% — invalid target rate.")
            continue
        if n_pos == 0 or n_neg == 0:
            print(f"[SKIP] {pct}% — dataset has only one class.")
            continue

        # Use a per-level seed to avoid identical samples across pcts if desired
        seed = random_state + int(pct)

        # ---- Option A: keep all positives, downsample negatives
        n_neg_needed = int(np.floor(n_pos * (1 - r) / r))

        if n_neg_needed <= n_neg:
            df_pos_keep = df_pos
            df_neg_keep = df_neg.sample(n=n_neg_needed, random_state=seed, replace=False)

            strategy = f"keep all positives ({n_pos}), downsample negatives to {n_neg_needed}"

        else:
            # ---- Option B: keep all negatives, downsample positives
            n_pos_needed = int(np.floor(r * n_neg / (1 - r)))

            if n_pos_needed <= 0:
                print(f"[SKIP] {dataset_name}_{pct}: would require 0 positives.")
                continue
            if n_pos_needed > n_pos:
                # This should be rare, but handle anyway
                print(f"[SKIP] {dataset_name}_{pct}: not achievable with available data.")
                continue

            df_neg_keep = df_neg
            df_pos_keep = df_pos.sample(n=n_pos_needed, random_state=seed, replace=False)

            strategy = f"keep all negatives ({n_neg}), downsample positives to {n_pos_needed}"

        df_new = (
            pd.concat([df_neg_keep, df_pos_keep], axis=0)
              .sample(frac=1, random_state=seed)
              .reset_index(drop=True)
        )

        n_total_new = len(df_new)
        n_pos_new = int(df_new[target_col].sum())
        new_rate = float(df_new[target_col].mean())

        save_path = save_dir / f"{dataset_name}_{pct}.csv"
        df_new.to_csv(save_path, index=False, sep=",")

        print(
            f"{dataset_name}_{pct}.csv | Total={n_total_new} | Pos={n_pos_new} | "
            f"Rate={new_rate:.4f} | {strategy}"
        )

    print("\nFinished generating imbalanced datasets.")


def merge_xai_results(
    data_dir: Path,
    datasets_to_run: list,
    imbalance_levels: list,
    model_choices: list,
):
    
    for dataset in datasets_to_run:

        print(f"\nProcessing dataset: {dataset}")

        dfs = []

        out_dir = data_dir / "output" / dataset

        for imbalance in imbalance_levels:

            dataset_variant = f"{dataset}_{imbalance}"

            for model in model_choices:

                file_path = out_dir / f"{dataset_variant}_{model}_xai_iterations.csv"

                if not file_path.exists():
                    print(f"Missing file: {file_path}")
                    continue

                df = pd.read_csv(file_path)

                # add metadata
                df["dataset"] = dataset
                df["imbalance"] = imbalance
                df["model"] = model

                dfs.append(df)

        if len(dfs) == 0:
            print(f"No files found for {dataset}")
            continue

        df_final = pd.concat(dfs, ignore_index=True)

        save_path = out_dir / f"{dataset}_xai_all_models.csv"
        df_final.to_csv(save_path, index=False)

        print(f"Saved merged dataset: {save_path}")