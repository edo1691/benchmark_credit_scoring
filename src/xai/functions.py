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
from sklearn.base import clone


import numpy as np
import pandas as pd
from sklearn.base import clone


def get_train_iterations_xai(
    pipeline,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    n_iterations: int = 5,
    percentage: float = 90,
):
    """
    Train the same model multiple times on different random subsamples of the TRAIN set,
    compute SHAP values on the TEST set for each iteration, and then compute global + local
    stability summaries.

    Returns
    -------
    df_shap : pd.DataFrame
        SHAP values for the test set across all iterations.
        Shape ~ (n_iterations * len(x_test), n_features + metadata)

    df_ranking_local : pd.DataFrame
        Local rank stability per (test_row_index, feature).

    extra : pd.DataFrame
        Extra output returned by compute_rankstab_local_from_shap (kept as in your logic).

    df_ranking_global : pd.DataFrame
        Global ranking per iteration + rankstab_global per feature.

    df_resume_local : pd.DataFrame
        Local ranking summary merged with global feature_importance.

    df_resume_global : pd.DataFrame
        Global summary including rankstab_global, value_stab, feature_importance and weights.

    df_resume : pd.DataFrame
        Final summary merging global and aggregated local stability per feature.

    Notes
    -----
    - get_prediction_xai(...) is called exactly as you already have it.
    - The main robustness change is preserving indices through preprocessing and sampling.
    """

    # ---------------------------------------------------------------------
    # 1) Preprocess train/test while preserving original indices
    # ---------------------------------------------------------------------
    x_train_pre = pipeline.named_steps["preprocessing"].transform(x_train)
    x_test_pre = pipeline.named_steps["preprocessing"].transform(x_test)

    # Force DataFrames with the original indices (prevents alignment/row-loss issues)
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

    feature_names = x_train_pre.columns.tolist() if hasattr(x_train_pre, "columns") else None

    # ---------------------------------------------------------------------
    # 2) SHAP across iterations (train on subsamples, explain full test set)
    # ---------------------------------------------------------------------
    sample_size = int(len(x_train_pre) * (percentage / 100))
    expected_test_rows = len(x_test_pre)

    shap_frames = []
    seen_samples = []  # keep your "ensure unique sample" logic

    for iteration in range(n_iterations):

        # --- 2.1 sample a unique training subset (by iteration seed) ---
        while True:
            sampled_x = x_train_pre.sample(n=sample_size, random_state=iteration, replace=False)

            # keep your uniqueness check
            if not any(sampled_x.equals(prev) for prev in seen_samples):
                seen_samples.append(sampled_x)
                break

        # match y using index-safe .loc
        sampled_y = y_train.loc[sampled_x.index]

        # --- 2.2 train a cloned model on the sampled data ---
        model_clone = clone(pipeline.named_steps["model"])
        model_clone.fit(sampled_x, sampled_y)

        # --- 2.3 build SHAP transformer for this specific fitted model ---
        shap_transformer = SHAPTransformer(model_clone)
        shap_transformer.fit(sampled_x)

        # --- 2.4 compute SHAP for the full test set (your prediction function unchanged) ---
        shap_df = get_prediction_xai(
            pipeline,
            shap_transformer,
            x_test_pre,
            feature_names,
            preprocessed_data=True,
        )

        # If get_prediction_xai returns same number of rows but reset index, restore it.
        if len(shap_df) == expected_test_rows:
            shap_df = shap_df.copy()
            shap_df.index = x_test_pre.index
        else:
            print(
                f"[WARN] iter={iteration}: get_prediction_xai returned {len(shap_df)} rows "
                f"(expected {expected_test_rows}). Row loss is happening inside get_prediction_xai/SHAP."
            )

        # add metadata columns
        shap_df["n_iteration"] = iteration
        shap_df["index"] = shap_df.index

        shap_frames.append(shap_df)

    # stack all iterations
    df_shap = pd.concat(shap_frames, ignore_index=True)

    # ---------------------------------------------------------------------
    # 3) Prepare absolute SHAP values for stability computations
    # ---------------------------------------------------------------------
    df_abs = df_shap.abs()

    # Drop base_value once here (and we will not drop it again later)
    if "base_value" in df_abs.columns:
        df_abs = df_abs.drop(columns=["base_value"])

    # ---------------------------------------------------------------------
    # 4) GLOBAL stability: SRA ranking + rankstab_global + value_stab
    # ---------------------------------------------------------------------
    # Your compute_global_sra_rankstab expects df_shap-like input with n_iteration/index
    # It will ignore identifier columns and compute per-feature global stats.
    df_values_global = compute_df_values_global(df_abs)
   
    # value stability based on df_values_global
    value_stab_global = compute_value_stab(df_values_global)

    # feature importance = mean importance across iterations
    df_resume_cv_global = value_stab_global.copy()

    df_resume_cv_global["feature_importance"] = (
        df_resume_cv_global.drop(columns=["feature", "value_stab"]).mean(axis=1)
    )
    df_filtered = df_resume_cv_global[(df_resume_cv_global["feature_importance"] > 0)]

    df_resume_cv_global = df_resume_cv_global[["feature", "feature_importance", "value_stab"]]

    # rankstab_global summary: feature + rankstab_global
    df_ranking_global = compute_df_ranking_global(df_filtered)
    # NOTE: compute_global_sra_rankstab returns "feature" column (not "index")
    df_resume_stab_global = df_ranking_global[["feature", "rankstab_global"]].copy()

    # Merge global rank stability with value stability + importance
    df_resume_global = df_resume_stab_global.merge(df_resume_cv_global, on="feature", how="left")

    # Convert feature_importance to percentage
    total_importance = df_resume_global["feature_importance"].sum()
    if total_importance and total_importance != 0:
        df_resume_global["feature_importance"] = df_resume_global["feature_importance"] / total_importance * 100
    else:
        df_resume_global["feature_importance"] = 0.0

    # Weighted measures (your exact logic)
    df_resume_global["rankstab_global_weight"] = (
        df_resume_global["rankstab_global"] * df_resume_global["feature_importance"]
    )
    df_resume_global["value_stab_weight"] = (
        df_resume_global["value_stab"] * df_resume_global["feature_importance"]
    )

    # ---------------------------------------------------------------------
    # 5) LOCAL stability: SMLE rank stability per (index, feature)
    # ---------------------------------------------------------------------
    # Your compute_rankstab_local_from_shap returns:
    #   - rankstab_records: list[dict] with columns: feature, index, rankstab_local
    #   - extra: something else you want to keep
    rankstab_records, extra = compute_rankstab_local_from_shap(df_abs)

    df_ranking_local = pd.DataFrame(rankstab_records)
    extra = pd.DataFrame(extra)

    # Your downstream expects int feature ids (kept)
    if not df_ranking_local.empty and "feature" in df_ranking_local.columns:
        df_ranking_local["feature"] = df_ranking_local["feature"].astype("int64")
    if not extra.empty and "feature" in extra.columns:
        extra["feature"] = extra["feature"].astype("int64")

    # Merge local rank stability with global feature importance
    df_resume_local = df_ranking_local.merge(df_resume_global, on="feature", how="left")

    # Keep only what you used later
    df_resume_local = df_resume_local[["feature", "index", "feature_importance", "rankstab_local"]].copy()

    # Local weighted stability
    df_resume_local["rankstab_local_weight"] = (
        df_resume_local["rankstab_local"] * df_resume_local["feature_importance"]
    )

    # Aggregate local stability per "index" (your logic)
    df_resume_local_agg = df_resume_local.groupby(["index"], as_index=True).sum(numeric_only=True)

    # In your original code you drop these after sum
    for col in ["feature", "feature_importance"]:
        if col in df_resume_local_agg.columns:
            df_resume_local_agg = df_resume_local_agg.drop(columns=[col])

    # Rename aggregated index to "feature" so it can merge with df_resume_global
    df_resume_local_agg = df_resume_local_agg.reset_index().rename(columns={"index": "feature"})

    # ---------------------------------------------------------------------
    # 6) Final resume = global resume + aggregated local stability
    # ---------------------------------------------------------------------
    df_resume = df_resume_global.merge(df_resume_local_agg, on="feature", how="left")

    return (
        df_shap,
        df_ranking_local,
        extra,
        df_ranking_global,
        df_resume_local,
        df_resume_global,
        df_resume,
    )


def compute_rankstab_local_from_shap(
    df_abs: pd.DataFrame,
    index_col: str = "index",
    iter_col: str = "n_iteration",
    exclude_columns: list = None,
) -> pd.DataFrame:
    """
    Input: df_abs with columns [index, n_iteration, <feature1>, <feature2>, ...]
           Each row = one observation (index) in one iteration, with SHAP abs values.

    Output: DataFrame with columns [feature, index, rankstab_local]
            rankstab_local = sample variance of feature rank across iterations.
    """
    if exclude_columns is None:
        exclude_columns = [index_col, iter_col]

    df = df_abs.copy()

    # Feature columns = everything except id/iteration columns
    feature_cols = [c for c in df.columns if c not in exclude_columns]

    # Ensure numeric (handles commas etc.)
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

    # Rank stability = sample variance of ranks across iterations for each (index, feature)
    out = (
        long.groupby([index_col, "feature"], as_index=False)["rank"]
        .var(ddof=1)  # matches (1/(K-1))*sum(...)
        .rename(columns={"rank": "rankstab_local"})
    )

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
