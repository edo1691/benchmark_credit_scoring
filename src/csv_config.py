"""CSV configuration helpers for datasets in this project.

Provide per-dataset default parameters (delimiter, header, encoding)
and a small helper to load a dataset by name using pandas.

Example:
    from src.csv_config import read_csv_by_name
    df = read_csv_by_name('data/german.csv', 'german')
"""
from typing import Dict, Any

import pandas as pd

# Default CSV reading parameters per dataset name (lower-cased keys).
# Update or extend this dict when adding new datasets or special formats.
DEFAULT_CSV_CONFIG: Dict[str, Dict[str, Any]] = {
    "german": {
        # the german dataset in this workspace uses whitespace to separate
        # fields rather than a single tab character.  We use a regex
        # separator and the Python engine to cope with variable spacing.
        # There is no header row, so we'll generate placeholder column names
        # (ft_1, ft_2, ...).
        "sep": r"\s+",           # regex for one or more whitespace chars
        "engine": "python",      # required for regex separators
        "header": None,
        "encoding": "utf-8",
        "auto_header": True,  # custom flag interpreted by helper
    },
    "australian": {"sep": ",", "header": 0, "encoding": "utf-8"},
    "japan": {"sep": ",", "header": 0, "encoding": "utf-8"},
    "kg_ccdea": {"sep": ",", "header": 0, "encoding": "utf-8"},
    "kg_gmsc": {"sep": ",", "header": 0, "encoding": "utf-8"},
    "taiwan": {"sep": ",", "header": 0, "encoding": "utf-8"},
    "th02": {"sep": ",", "header": 0, "encoding": "utf-8"},
}


def get_csv_params(name: str) -> Dict[str, Any]:
    """Return a copy of the CSV read parameters for `name`.

    Args:
        name: Dataset name (case-insensitive). Commonly the CSV filename
              without extension (e.g. "german").

    Raises:
        KeyError: if no configuration exists for the given name.
    """
    key = name.lower()
    if key in DEFAULT_CSV_CONFIG:
        return DEFAULT_CSV_CONFIG[key].copy()
    raise KeyError(f"No CSV config found for dataset '{name}'")


def read_csv_by_name(path: str, name: str, **kwargs) -> pd.DataFrame:
    """Read a CSV file using the dataset-specific defaults.

    Parameters in `DEFAULT_CSV_CONFIG` are used first; any `kwargs` passed
    will override those defaults (for example `sep` or `encoding`).

    A dataset may declare the custom key ``auto_header`` in its
    configuration. When set to ``True`` the loader will generate placeholder
    column names of the form ``ft_1``, ``ft_2`` ... after reading.  This is
    useful for sources like the German file, which has no header row.

    Args:
        path: Path to the CSV file.
        name: Dataset name to determine default parsing parameters.

    Returns:
        A pandas DataFrame loaded with the resolved parameters.
    """
    params = get_csv_params(name)
    # pop the custom key so pandas doesn't choke on it
    auto_header = params.pop("auto_header", False)
    params.update(kwargs)
    df = pd.read_csv(path, **params)
    if auto_header:
        # assign placeholder column names
        df.columns = [f"ft_{i+1}" for i in range(len(df.columns))]
    return df
