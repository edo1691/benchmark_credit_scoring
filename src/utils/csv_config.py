from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
from pathlib import Path
import re
from pathlib import Path
from typing import Iterable
import pandas as pd
from fractions import Fraction


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
    Create derived datasets with target positive rates.

    For each target rate, the function maximizes the number of usable
    observations while preserving the requested imbalance ratio.

    Example:
        1% positive rate means:
            1 positive / 99 negatives
            2 positives / 198 negatives
            ...
        The function chooses the largest feasible multiple.
    """

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

    n_pos = len(df_pos)
    n_neg = len(df_neg)
    n_total = len(df)

    print("==========================================")
    print(f"Original dataset: {dataset_name}")
    print(f"Total samples: {n_total}")
    print(f"Positive samples: {n_pos}")
    print(f"Negative samples: {n_neg}")
    print(f"Positive rate: {df[target_col].mean():.4f}")
    print("==========================================\n")

    save_dir = data_dir / "intermediate" / dataset_name
    save_dir.mkdir(parents=True, exist_ok=True)

    for pct in imbalance_levels:
        r = pct / 100

        if r <= 0 or r >= 1:
            print(f"[SKIP] {pct}% — invalid target rate.")
            continue

        if n_pos == 0 or n_neg == 0:
            print(f"[SKIP] {pct}% — dataset has only one class.")
            continue

        seed = random_state + int(float(pct) * 1000)

        # Convert percentage into an exact fraction
        # Example: 1% -> 1/100, 5% -> 1/20
        rate_fraction = Fraction(str(r)).limit_denominator(10_000)

        pos_unit = rate_fraction.numerator
        total_unit = rate_fraction.denominator
        neg_unit = total_unit - pos_unit

        # Largest feasible multiplier preserving the exact ratio
        max_k_pos = n_pos // pos_unit
        max_k_neg = n_neg // neg_unit
        k = min(max_k_pos, max_k_neg)

        if k <= 0:
            print(
                f"[SKIP] {dataset_name}_{pct}: exact {pct}% rate not achievable "
                f"with available positives/negatives."
            )
            continue

        n_pos_keep = pos_unit * k
        n_neg_keep = neg_unit * k

        df_pos_keep = df_pos.sample(
            n=n_pos_keep,
            random_state=seed,
            replace=False,
        )

        df_neg_keep = df_neg.sample(
            n=n_neg_keep,
            random_state=seed,
            replace=False,
        )

        df_new = (
            pd.concat([df_pos_keep, df_neg_keep], axis=0)
            .sample(frac=1, random_state=seed)
            .reset_index(drop=True)
        )

        n_total_new = len(df_new)
        n_pos_new = int(df_new[target_col].sum())
        n_neg_new = n_total_new - n_pos_new
        new_rate = df_new[target_col].mean()

        save_path = save_dir / f"{dataset_name}_{pct}.csv"
        df_new.to_csv(save_path, index=False, sep=",")

        print(
            f"{dataset_name}_{pct}.csv | "
            f"Total={n_total_new} | "
            f"Pos={n_pos_new} | "
            f"Neg={n_neg_new} | "
            f"Rate={new_rate:.4f} | "
            f"Target={r:.4f}"
        )

    print("\nFinished generating imbalanced datasets.")


def merge_xai_results(
    data_dir: Path,
    datasets_to_run: list[str],
    imbalance_levels: list,
    model_choices: list[str],
    file_types: list[str] = None,
    normalize: bool = True,
    normalization_suffix: str = "_normalized",
):
    """
    Merge XAI result files across imbalance levels and models.

    For each dataset and each file_type, this function:
    1. Reads all available XAI result files
    2. Merges them into one raw dataframe
    3. Optionally creates a normalized version
    4. Saves both raw and normalized outputs

    Parameters
    ----------
    data_dir : Path
        Base data directory.
    datasets_to_run : list[str]
        Dataset names to process.
    imbalance_levels : list
        Imbalance values to include.
    model_choices : list[str]
        Model names to include.
    file_types : list[str] | None
        File types to merge, e.g.:
        ["resume_local", "resume_global", "summary"]

        If None, defaults to ["resume_local"].
    normalize : bool, default=True
        Whether to create and save a normalized output.
    normalization_suffix : str, default="_normalized"
        Suffix added to normalized saved filename.

    Returns
    -------
    dict
        Nested dictionary keyed by dataset and file_type:
        {
            dataset_name: {
                file_type: {
                    "raw": raw_dataframe,
                    "normalized": normalized_dataframe or None
                }
            }
        }
    """
    if file_types is None:
        file_types = ["resume_local"]

    results = {}

    for dataset in datasets_to_run:
        print(f"\nProcessing dataset: {dataset}")
        out_dir = data_dir / "output" / dataset / "xai"
        results[dataset] = {}

        for file_type in file_types:
            print(f"  Merging file type: {file_type}")
            dfs = []

            for imbalance in imbalance_levels:
                dataset_variant = f"{dataset}_{imbalance}"

                for model in model_choices:
                    file_path = (
                        out_dir
                        / model
                        / dataset_variant
                        / "metrics"
                        / f"{dataset_variant}_{model}_{file_type}.csv"
                    )

                    if not file_path.exists():
                        print(f"Missing file: {file_path}")
                        continue

                    df = pd.read_csv(file_path)

                    # Add metadata
                    df["dataset"] = dataset
                    df["imbalance"] = imbalance
                    df["model"] = model

                    dfs.append(df)

            if not dfs:
                print(f"No files found for dataset={dataset}, file_type={file_type}")
                results[dataset][file_type] = {
                    "raw": None,
                    "normalized": None,
                }
                continue

            # -----------------------------
            # Raw merged output
            # -----------------------------
            df_raw = pd.concat(dfs, ignore_index=True)

            raw_save_path = out_dir / f"{dataset}_xai_all_models_{file_type}.csv"
            df_raw.to_csv(raw_save_path, index=False)
            print(f"Saved raw merged dataset: {raw_save_path}")

            df_normalized = None

            # -----------------------------
            # Normalized output
            # -----------------------------
            if normalize:
                df_normalized = df_raw.copy()

                # Select numeric columns
                numeric_cols = df_normalized.select_dtypes(include="number").columns.tolist()

                # Exclude metadata / identifiers / columns you do not want to normalize
                exclude_cols = {
                    "imbalance",
                    "feature",
                    "index",
                    "feature_importance",
                }

                numeric_cols = [c for c in numeric_cols if c not in exclude_cols]

                for col in numeric_cols:
                    min_val = df_normalized[col].min()
                    max_val = df_normalized[col].max()

                    # Avoid division by zero for constant columns
                    if pd.notna(min_val) and pd.notna(max_val):
                        if max_val > min_val:
                            df_normalized[col] = (
                                (df_normalized[col] - min_val) / (max_val - min_val)
                            )
                        else:
                            df_normalized[col] = 0.0

                normalized_save_path = (
                    out_dir
                    / f"{dataset}_xai_all_models_{file_type}{normalization_suffix}.csv"
                )
                df_normalized.to_csv(normalized_save_path, index=False)
                print(f"Saved normalized merged dataset: {normalized_save_path}")

            results[dataset][file_type] = {
                "raw": df_raw,
                "normalized": df_normalized,
            }

    return