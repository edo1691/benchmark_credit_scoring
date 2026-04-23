import os

import shap
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC
from sklearn.base import clone


def wrap_booster_if_needed(model):
    """Wrap raw XGBoost Booster object into an XGBClassifier."""
    if isinstance(model, xgb.Booster):
        clf = xgb.XGBClassifier()
        clf._Booster = model
        clf._le = None  # Disable label encoder (important for compatibility)
        return clf
    return model


class SHAPTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, model):
        self.model = model
        self.explainer = None
        self.feature_names = None

    def fit(self, X, y=None):
        self.feature_names = X.columns.tolist() if hasattr(X, "columns") else None
        tree_models = (RandomForestClassifier, RandomForestRegressor, lgb.LGBMModel)
        try:
            if isinstance(self.model, tree_models):
                # Use TreeExplainer for RF and LightGBM
                self.explainer = shap.TreeExplainer(self.model)
            elif isinstance(self.model, xgb.XGBModel):
                # Try TreeExplainer first, fallback if it fails (e.g. XGBoost 3.0.0)
                try:
                    self.explainer = shap.TreeExplainer(self.model)
                except Exception as e:
                    print(f"TreeExplainer failed for XGBoost, falling back: {e}")
                    masker = shap.maskers.Independent(X)
                    self.explainer = shap.Explainer(self.model.predict, masker)
            elif isinstance(self.model, SVC):
                masker = shap.maskers.Independent(X)
                self.explainer = shap.Explainer(self.model.predict, masker)
            else:
                # Generic fallback
                masker = shap.maskers.Independent(X)
                self.explainer = shap.Explainer(self.model.predict, masker)
        except Exception as e:
            raise ValueError(f"SHAP Explainer initialization failed: {str(e)}")
        return self

    def transform(self, X, col_names=None):
        if self.feature_names is not None:
            if col_names is None:
                X = X[self.feature_names]
            else:
                X = pd.DataFrame(X, columns=col_names)[self.feature_names]
        return self.explainer(X)


def get_train_xai(pipeline, data):
    # Compute SHAP transformer
    x_train_preprocessed = pipeline.named_steps["preprocessing"].transform(data)
    model = pipeline.named_steps["model"]
    print("DEBUG - Model from pipeline:", type(model))
    model = wrap_booster_if_needed(model)
    print("DEBUG - Model after wrapping:", type(model))
    shap_transformer = SHAPTransformer(model)
    shap_transformer.fit(x_train_preprocessed)
    return shap_transformer


def get_prediction_xai(
    pipeline, shap_transformer, data, col_names=None, preprocessed_data=False
):

    # Pre-processing data values
    if preprocessed_data:
        shap_values = shap_transformer.transform(data)
    else:
        x_test_preprocessed = pipeline.named_steps["preprocessing"].transform(data)
        shap_values = shap_transformer.transform(x_test_preprocessed)

    # Check the shape of the SHAP values
    if shap_values.values.ndim == 3:
        # For 3-dimensional SHAP values (e.g., binary classification with SHAP values for both classes)
        shap_values_class_1 = shap_values.values[:, :, 1]
        base_values_class_1 = shap_values.base_values[:, 1]
    elif shap_values.values.ndim == 2:
        # For 2-dimensional SHAP values (e.g., regression or single-class SHAP values)
        shap_values_class_1 = shap_values.values
        base_values_class_1 = shap_values.base_values
    else:
        raise ValueError(
            "Unexpected SHAP values shape: {}".format(shap_values.values.shape)
        )

    # Check if shap_values_class_1 is a DataFrame or numpy array and use appropriate columns
    if isinstance(data, pd.DataFrame):
        shap_df = pd.DataFrame(shap_values_class_1, columns=data.columns)
    else:
        shap_df = pd.DataFrame(shap_values_class_1, columns=col_names)
    shap_df["base_value"] = base_values_class_1

    return shap_df


import pandas as pd


def compute_xai_metrics_from_shap(df_shap: pd.DataFrame):
    """
    Compute global and local XAI stability metrics from stacked SHAP values.

    Parameters
    ----------
    df_shap : pd.DataFrame
        Stacked SHAP values across iterations.
        Expected to contain:
        - SHAP feature columns
        - n_iteration
        - index
        - optionally base_value

    Returns
    -------
    df_ranking_local : pd.DataFrame
        Local rank stability per (index, feature)

    extra : pd.DataFrame
        Extra output returned by compute_rankstab_local_from_shap

    df_ranking_global : pd.DataFrame
        Global ranking summary per feature

    df_resume_local : pd.DataFrame
        Local stability merged with global feature importance

    df_resume_global : pd.DataFrame
        Global summary including rankstab_global, value_stab, feature_importance
        and weighted measures

    df_resume : pd.DataFrame
        Final feature-level summary merging global and aggregated local stability
    """

    # ---------------------------------------------------------------------
    # 1) Validate required metadata columns
    # ---------------------------------------------------------------------
    required_cols = {"n_iteration", "index"}
    missing_cols = required_cols - set(df_shap.columns)
    if missing_cols:
        raise ValueError(
            f"df_shap is missing required columns: {sorted(missing_cols)}"
        )

    # ---------------------------------------------------------------------
    # 2) Prepare absolute SHAP values
    # ---------------------------------------------------------------------
    df_abs = df_shap.copy()

    meta_cols = {"n_iteration", "index", "base_value"}
    feature_cols = [col for col in df_abs.columns if col not in meta_cols]

    if not feature_cols:
        raise ValueError("No feature columns found in df_shap.")

    df_abs[feature_cols] = df_abs[feature_cols].abs()

    if "base_value" in df_abs.columns:
        df_abs = df_abs.drop(columns=["base_value"])

    # ---------------------------------------------------------------------
    # 3) GLOBAL stability
    # ---------------------------------------------------------------------
    df_values_global = compute_df_values_global(df_abs)
    value_stab_global = compute_value_stab(df_values_global)

    df_resume_cv_global = value_stab_global.copy()

    metric_cols = [
        col for col in df_resume_cv_global.columns
        if col not in ["feature", "value_stab"]
    ]

    if not metric_cols:
        raise ValueError(
            "compute_value_stab(df_values_global) did not return iteration/value columns."
        )

    df_resume_cv_global["feature_importance"] = (
        df_resume_cv_global[metric_cols].mean(axis=1)
    )

    df_filtered = df_resume_cv_global[
        df_resume_cv_global["feature_importance"] > 0
    ].copy()

    df_resume_cv_global = df_resume_cv_global[
        ["feature", "feature_importance", "value_stab"]
    ].copy()

    df_ranking_global = compute_df_ranking_global(df_filtered)
    df_resume_stab_global = df_ranking_global[
        ["feature", "rankstab_global"]
    ].copy()

    # Make dtype consistent
    df_resume_cv_global["feature"] = df_resume_cv_global["feature"].astype(str)
    df_ranking_global["feature"] = df_ranking_global["feature"].astype(str)
    df_resume_stab_global["feature"] = df_resume_stab_global["feature"].astype(str)

    df_resume_global = df_resume_stab_global.merge(
        df_resume_cv_global,
        on="feature",
        how="left"
    )

    total_importance = df_resume_global["feature_importance"].sum()
    if total_importance != 0:
        df_resume_global["feature_importance"] = (
            df_resume_global["feature_importance"] / total_importance * 100
        )
    else:
        df_resume_global["feature_importance"] = 0.0

    df_resume_global["rankstab_global_weight"] = (
        df_resume_global["rankstab_global"] * df_resume_global["feature_importance"]
    )
    df_resume_global["value_stab_weight"] = (
        df_resume_global["value_stab"] * df_resume_global["feature_importance"]
    )

    # ---------------------------------------------------------------------
    # 4) LOCAL stability
    # ---------------------------------------------------------------------
    meta_cols = ["n_iteration", "index"]

    selected_features = df_resume_global["feature"].astype(str).tolist()

    cols_to_keep = meta_cols + [
        col for col in selected_features if col in df_abs.columns
    ]

    df_abs_filtered = df_abs[cols_to_keep]
    rankstab_records, extra = compute_rankstab_local_from_shap(df_abs_filtered)

    df_ranking_local = pd.DataFrame(rankstab_records)
    extra = pd.DataFrame(extra)

    if not df_ranking_local.empty:
        if "feature" not in df_ranking_local.columns:
            raise ValueError("df_ranking_local does not contain a 'feature' column.")
        if "index" not in df_ranking_local.columns:
            raise ValueError("df_ranking_local does not contain an 'index' column.")
        if "rankstab_local" not in df_ranking_local.columns:
            raise ValueError("df_ranking_local does not contain a 'rankstab_local' column.")

        df_ranking_local["feature"] = pd.to_numeric(
            df_ranking_local["feature"], errors="coerce"
        ).astype("Int64")

    if not extra.empty and "feature" in extra.columns:
        extra["feature"] = pd.to_numeric(
            extra["feature"], errors="coerce"
        ).astype("Int64")

    df_resume_global["feature"] = pd.to_numeric(
        df_resume_global["feature"], errors="coerce"
    ).astype("Int64")

    df_resume_local = df_ranking_local.merge(
        df_resume_global,
        on="feature",
        how="left"
    )

    df_resume_local = df_resume_local[
        ["feature", "index", "feature_importance", "rankstab_local"]
    ].copy()

    df_resume_local["rankstab_local_weight"] = (
        df_resume_local["rankstab_local"] * df_resume_local["feature_importance"]
    )

    # Aggregate local stability by feature
    df_resume_local_agg = (
        df_resume_local.groupby("feature", as_index=False)
        .agg({
            "rankstab_local": "mean",
            "rankstab_local_weight": "mean"
        })
    )

    # ---------------------------------------------------------------------
    # 5) Final summary
    # ---------------------------------------------------------------------
    df_resume = df_resume_global.merge(
        df_resume_local_agg,
        on="feature",
        how="left"
    )

    return (
        df_ranking_local,
        extra,
        df_ranking_global,
        df_resume_local,
        df_resume_global,
        df_resume,
    )


def get_train_iterations_shap(
    pipeline,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    n_iterations: int = 5,
    percentage: float = 90,
    min_positive_per_iteration: int = 2,
):
    """
    Train the same model multiple times on different random subsamples
    of the TRAIN set and compute SHAP values on the TEST set.

    Sampling strategy
    -----------------
    - Guarantees at least `min_positive_per_iteration` positives in each iteration.
    - Tries to preserve the original class proportion when possible.
    - Allows positives to be repeated if needed.
    - Tries to vary negatives without replacement as much as possible.

    Assumptions
    -----------
    - Binary target with labels {0, 1}.
    """

    # ---------------------------------------------------------------------
    # 1) Preprocess train/test while preserving original indices
    # ---------------------------------------------------------------------
    x_train_pre = pipeline.named_steps["preprocessing"].transform(x_train)
    x_test_pre = pipeline.named_steps["preprocessing"].transform(x_test)

    if not isinstance(x_train_pre, pd.DataFrame):
        x_train_pre = pd.DataFrame(x_train_pre, index=x_train.index)
    else:
        x_train_pre = x_train_pre.copy()
        x_train_pre.index = x_train.index

    if not isinstance(x_test_pre, pd.DataFrame):
        x_test_pre = pd.DataFrame(x_test_pre, index=x_test.index)
    else:
        x_test_pre = x_test_pre.copy()
        x_test_pre.index = x_test.index

    # Align y_train to the preprocessed train index
    y_train = y_train.loc[x_train_pre.index].copy()

    feature_names = (
        x_train_pre.columns.tolist()
        if hasattr(x_train_pre, "columns")
        else None
    )

    # ---------------------------------------------------------------------
    # 2) Basic checks
    # ---------------------------------------------------------------------
    unique_classes = set(pd.Series(y_train).dropna().unique())
    if not unique_classes.issubset({0, 1}):
        raise ValueError(
            f"Expected binary target with labels {{0, 1}}, got {sorted(unique_classes)}"
        )

    pos_idx = y_train[y_train == 1].index.to_numpy()
    neg_idx = y_train[y_train == 0].index.to_numpy()

    n_pos_total = len(pos_idx)
    n_neg_total = len(neg_idx)

    if n_pos_total == 0:
        raise ValueError("y_train contains no positive class (1).")
    if n_neg_total == 0:
        raise ValueError("y_train contains no negative class (0).")

    sample_size = int(len(x_train_pre) * (percentage / 100))
    expected_test_rows = len(x_test_pre)

    if sample_size < 2:
        raise ValueError(
            f"sample_size={sample_size} is too small. Increase `percentage`."
        )

    if min_positive_per_iteration < 1:
        raise ValueError("`min_positive_per_iteration` must be >= 1.")

    if min_positive_per_iteration >= sample_size:
        raise ValueError(
            "`min_positive_per_iteration` must be smaller than the total sample size."
        )

    positive_rate = n_pos_total / len(y_train)

    # ---------------------------------------------------------------------
    # 3) Helper: sample one iteration with guaranteed positives
    # ---------------------------------------------------------------------
    def _sample_iteration_indices(random_state: int) -> np.ndarray:
        rng = np.random.RandomState(random_state)

        # Target number of positives based on original prevalence,
        # but never below the requested minimum.
        n_pos_target = max(
            min_positive_per_iteration,
            int(round(sample_size * positive_rate))
        )

        # Keep at least one slot for negatives.
        n_pos_target = min(n_pos_target, sample_size - 1)
        n_neg_target = sample_size - n_pos_target

        # Positives:
        # - without replacement if enough are available
        # - with replacement if more are needed than available
        if n_pos_target <= n_pos_total:
            sampled_pos = rng.choice(pos_idx, size=n_pos_target, replace=False)
        else:
            sampled_pos_unique = pos_idx.copy()
            extra_needed = n_pos_target - n_pos_total
            sampled_pos_extra = rng.choice(pos_idx, size=extra_needed, replace=True)
            sampled_pos = np.concatenate([sampled_pos_unique, sampled_pos_extra])

        # Negatives:
        # - prefer without replacement so zeros change as much as possible
        # - fallback to replacement only if absolutely necessary
        if n_neg_target <= n_neg_total:
            sampled_neg = rng.choice(neg_idx, size=n_neg_target, replace=False)
        else:
            sampled_neg = rng.choice(neg_idx, size=n_neg_target, replace=True)

        sampled_idx = np.concatenate([sampled_pos, sampled_neg])
        rng.shuffle(sampled_idx)

        return sampled_idx

    # ---------------------------------------------------------------------
    # 4) SHAP across iterations
    # ---------------------------------------------------------------------
    shap_frames = []

    for iteration in range(n_iterations):
        sampled_idx = _sample_iteration_indices(random_state=iteration)

        sampled_x = x_train_pre.loc[sampled_idx]
        sampled_y = y_train.loc[sampled_idx]

        # Safety check: never fit on a single-class sample
        if sampled_y.nunique() < 2:
            raise RuntimeError(
                f"Iteration {iteration} produced a single-class sample. "
                "This should not happen with the custom sampler."
            )

        model_clone = clone(pipeline.named_steps["model"])
        model_clone.fit(sampled_x, sampled_y)

        shap_transformer = SHAPTransformer(model_clone)
        shap_transformer.fit(sampled_x)

        shap_df = get_prediction_xai(
            pipeline,
            shap_transformer,
            x_test_pre,
            feature_names,
            preprocessed_data=True,
        )

        if len(shap_df) == expected_test_rows:
            shap_df = shap_df.copy()
            shap_df.index = x_test_pre.index
        else:
            print(
                f"[WARN] iter={iteration}: get_prediction_xai returned {len(shap_df)} rows "
                f"(expected {expected_test_rows})."
            )

        shap_df["n_iteration"] = iteration
        shap_df["index"] = shap_df.index
        shap_df["n_pos_sampled"] = int((sampled_y == 1).sum())
        shap_df["n_neg_sampled"] = int((sampled_y == 0).sum())

        shap_frames.append(shap_df)

    df_shap = pd.concat(shap_frames, ignore_index=True)

    return df_shap


import pandas as pd


def compute_rankstab_local_from_shap(
    df_abs: pd.DataFrame,
    index_col: str = "index",
    iter_col: str = "n_iteration",
    exclude_columns: list = None,
    output_col: str = "rankstab_local",
    higher_is_better: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute local rank stability from SHAP absolute values.

    Parameters
    ----------
    df_abs : pd.DataFrame
        Input dataframe with columns:
        [index_col, iter_col, <feature1>, <feature2>, ...]
        Each row corresponds to one observation in one iteration.
    index_col : str
        Observation identifier column.
    iter_col : str
        Iteration identifier column.
    exclude_columns : list | None
        Columns to exclude from feature ranking. If None, uses [index_col, iter_col].
    output_col : str
        Name of the output stability column.
    higher_is_better : bool
        If True, convert rank variance to a stability score:
            stability = 1 / (1 + variance)
        so that higher values indicate greater stability.
        If False, returns the raw rank variance, where lower values indicate greater stability.

    Returns
    -------
    out : pd.DataFrame
        DataFrame with columns [feature, index_col, output_col]
    long : pd.DataFrame
        Long-format dataframe with one row per (index, iteration, feature),
        including the computed rank.
    """
    if exclude_columns is None:
        exclude_columns = [index_col, iter_col]

    df = df_abs.copy()

    # Feature columns = everything except id/iteration columns
    feature_cols = [c for c in df.columns if c not in exclude_columns]

    if not feature_cols:
        raise ValueError("No feature columns found after excluding index/iteration columns.")

    # Ensure numeric
    for c in feature_cols:
        df[c] = (
            df[c]
            .astype(str)
            .str.replace(",", ".", regex=False)
        )
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Rank per row: higher SHAP => rank 1
    df[feature_cols] = df[feature_cols].rank(axis=1, ascending=False, method="average")

    # Long format: one row per (index, iteration, feature)
    long = df.melt(
        id_vars=[index_col, iter_col],
        value_vars=feature_cols,
        var_name="feature",
        value_name="rank",
    )

    # Variance of ranks across iterations for each (index, feature)
    out = (
        long.groupby([index_col, "feature"], as_index=False)["rank"]
        .var(ddof=1)
        .rename(columns={"rank": "_rank_variance"})
    )

    if higher_is_better:
        # Convert variance into stability score in (0, 1], where 1 = perfectly stable
        out[output_col] = 1 / (1 + out["_rank_variance"])
        out = out.drop(columns="_rank_variance")
    else:
        out = out.rename(columns={"_rank_variance": output_col})

    return out, long



def compute_df_values_global(
    df_shap_or_abs: pd.DataFrame,
    iteration_col: str = "n_iteration",
    index_col: str = "index",
    drop_cols: list = None,
) -> pd.DataFrame:
    """
    Build df_values_global (Global SRA values table).

    Logic (same as your code):
      1) (Optionally) drop columns like 'base_value'
      2) Take absolute values for feature columns
      3) Group by iteration and SUM feature values across all test rows (different indices)
      4) Transpose so rows = features, columns = iterations

    Output
    ------
    df_values_global:
      columns: ['feature', <iter0>, <iter1>, ...]
      each cell = summed |SHAP| for that feature in that iteration
    """
    df = df_shap_or_abs.copy()

    # drop optional columns safely
    if drop_cols:
        df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    # feature columns = everything except metadata
    meta_cols = {iteration_col, index_col}
    feature_cols = [c for c in df.columns if c not in meta_cols]

    # abs only on feature columns
    df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce").abs()

    # group by iteration and sum across rows (index values)
    df_sum = (
        df.drop(columns=[index_col], errors="ignore")
          .groupby(iteration_col, as_index=True)
          .sum(numeric_only=True)
    )

    # transpose -> feature x iteration
    df_values_global = (
        df_sum.T
              .reset_index()
              .rename(columns={"index": "feature"})
    )

    return df_values_global


def compute_df_ranking_global(
    df_values_global: pd.DataFrame,
    feature_col: str = "feature",
) -> pd.DataFrame:
    """
    Build df_ranking_global (Global ranking per iteration + rankstab_global).

    Logic (same as your code):
      1) Drop feature column -> matrix [n_features x n_iterations]
      2) Rank features per iteration (descending)
      3) For each feature row, compute rankstab_global across iterations:
           (1/(K-1)) * sum_k (rank_k - mean_rank)^2

    Output
    ------
    df_ranking_global:
      columns: ['feature', <iter0>, <iter1>, ..., 'rankstab_global']
      ranks are 1..n_features (1 = most important)
    """
    if feature_col not in df_values_global.columns:
        raise ValueError(f"'{feature_col}' not found in df_values_global")

    # rank per iteration column (descending)
    rank_mat = (
        df_values_global
        .drop(columns=[feature_col])
        .rank(axis=0, ascending=False, method="average")
    )

    num_iterations = rank_mat.shape[1]
    if num_iterations < 2:
        raise ValueError("Need at least 2 iterations to compute rankstab_global.")

    # vectorized version of your rankstab formula
    mean_ranks = rank_mat.mean(axis=1)
    rankstab = ((rank_mat.sub(mean_ranks, axis=0)) ** 2).sum(axis=1) / (num_iterations - 1)

    df_ranking_global = rank_mat.copy()
    df_ranking_global.insert(0, feature_col, df_values_global[feature_col].values)
    df_ranking_global["rankstab_global"] = rankstab.values

    return df_ranking_global


def compute_value_stab(df: pd.DataFrame, min_presence: int = 2) -> pd.DataFrame:
    """
    Compute ValueStab** for each feature (row) and return the input DataFrame
    with an additional 'value_stab' column.
    Parameters:
    df : pd.DataFrame
        Input DataFrame with columns [n_iteration, 0, 1, ..., rankstab]
    min_presence : int
        Minimum non-zero or non-NaN values required to compute the value stab
    Returns:
    pd.DataFrame
        Same as input with a new column 'value_stab'
    """
    # Drop rankstab if exists, keep a copy of it to re-attach later
    rankstab = df.get("rankstab", pd.Series(index=df.index))
    df_values = df.drop(columns=["rankstab"], errors="ignore").set_index("feature")

    # Calculate value stab for each row (feature)
    def calc_stab(row):
        valid_vals = row.dropna()
        if (valid_vals != 0).sum() >= min_presence:
            mean_val = valid_vals.mean()
            if mean_val != 0:
                std_val = valid_vals.std(ddof=1)
                return std_val / mean_val
        return np.nan

    df_values["value_stab"] = df_values.apply(calc_stab, axis=1)
    # Reconstruct final DataFrame (including rankstab if it existed)
    result_df = df_values.reset_index()
    if "rankstab" in df.columns:
        result_df["rankstab"] = rankstab.values
    return result_df


def compute_rankstab(matrix):
    """
    Compute rankstab values for a given matrix.
    Each row represents a sample and each column a feature.
    """
    num_columns = matrix.shape[1]
    sum_rows = matrix.sum(axis=1).values.reshape(-1, 1)
    expected = sum_rows * (1 / num_columns)
    deviations = matrix.values - expected
    squared_deviations = deviations**2
    return (1 / (num_columns - 1)) * squared_deviations.sum(axis=1)


def get_feature_importance(pipeline, feature_names):
    # Extract the model from the pipeline
    model = pipeline.named_steps["model"]

    # Check the type of model and get feature importance accordingly
    if hasattr(model, "feature_importances_"):
        ft_importances = model.feature_importances_
    elif hasattr(model, "booster_"):
        ft_importances = model.booster_.feature_importance(importance_type="gain")
    elif hasattr(model, "feature_importance"):
        ft_importances = model.feature_importance(importance_type="gain")
    elif model.__class__.__name__ in [
        "MLPClassifier",
        "MLPRegressor",
        "SVM",
        "SVM-Rbf",
        "ANN",
        "BagDT",
        "BagNN",
        "AdaBoost-DT",
    ]:
        ft_importances = np.zeros(len(feature_names))
    else:
        ft_importances = np.zeros(len(feature_names))
        # raise ValueError("Model type not supported for feature importance extraction.")

    # Normalize to percentages
    if not np.all(ft_importances == 0):
        ft_importances = ft_importances / ft_importances.sum() * 100

    # Create a DataFrame with feature names and their importance
    feature_importance_df = pd.DataFrame(
        {"Feature": feature_names, "Importance": ft_importances}
    ).sort_values(by="Importance", ascending=False)

    return feature_importance_df


def compose_shap_df_preds_file_name(algo, config, product, project_name):
    return os.path.join(config['repo'].get_path(), f"{project_name}_shap_values_{algo}_{product}.pq")


def compose_shap_model_output_file_name(repo, shap_df_name, project_name, product, algo):
    return os.path.join(repo.get_path(), f"{project_name}_{shap_df_name}_{algo}_{product}.pq")


def compose_shap_file_name(repo, algo, product):
    path_shap_transform = repo.path_shap_transform_test() if repo.flag_test() else repo.path_shap_transform()
    return f"{path_shap_transform}_{algo}_{product}.pkl"
