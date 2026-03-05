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


def get_train_iterations_xai(
    pipeline, x_train, x_test, y_train, n_iterations=5, percentage=90
):
    """
    Same logic as your original function, but with index-safe handling to avoid
    accidental row-count / alignment issues.

    NOTE: get_prediction_xai(...) is unchanged (called exactly as before).
    """

    # -----------------------------
    # Preprocess (keep indices!)
    # -----------------------------
    x_train_preprocessed = pipeline.named_steps["preprocessing"].transform(x_train)
    x_test_preprocessed = pipeline.named_steps["preprocessing"].transform(x_test)

    # Ensure DataFrame with original indices (critical for stable sampling + alignment)
    if not isinstance(x_train_preprocessed, pd.DataFrame):
        x_train_preprocessed = pd.DataFrame(x_train_preprocessed, index=x_train.index)
    else:
        # make sure it uses x_train index (in case transform changed it)
        x_train_preprocessed = x_train_preprocessed.copy()
        x_train_preprocessed.index = x_train.index

    if not isinstance(x_test_preprocessed, pd.DataFrame):
        x_test_preprocessed = pd.DataFrame(x_test_preprocessed, index=x_test.index)
    else:
        x_test_preprocessed = x_test_preprocessed.copy()
        x_test_preprocessed.index = x_test.index

    feature_names = (
        x_train_preprocessed.columns.tolist()
        if hasattr(x_train_preprocessed, "columns")
        else None
    )

    shap_predicted_df_full = []

    sample_size = int(len(x_train_preprocessed) * (percentage / 100))
    unique_samples = []

    # Optional: keep expected test size for sanity checks
    expected_test_rows = len(x_test_preprocessed)

    for iteration in range(n_iterations):

        # -----------------------------
        # Sample rows from train (preserve index)
        # -----------------------------
        while True:
            sampled_data = x_train_preprocessed.sample(
                n=sample_size, random_state=iteration, replace=False
            )

            # Your uniqueness check (kept)
            if not any(
                sampled_data.equals(existing_sample)
                for existing_sample in unique_samples
            ):
                unique_samples.append(sampled_data)
                break

        # Match target values USING .loc (index-safe)
        sampled_y = y_train.loc[sampled_data.index]

        # -----------------------------
        # Train cloned model
        # -----------------------------
        model_clone = clone(pipeline.named_steps["model"])
        model_clone.fit(sampled_data, sampled_y)

        # Fresh SHAP transformer with the new model
        shap_transformer = SHAPTransformer(model_clone)
        shap_transformer.fit(sampled_data)

        # -----------------------------
        # Predict SHAP on the FULL test set
        # (your function unchanged)
        # -----------------------------
        shap_predicted_df = get_prediction_xai(
            pipeline,
            shap_transformer,
            x_test_preprocessed,
            feature_names,
            preprocessed_data=True,
        )

        # -----------------------------
        # Defensive alignment (does NOT change your prediction function)
        # If get_prediction_xai returns same number of rows but resets index,
        # we restore the test index to keep consistency downstream.
        # If it returns fewer rows, we do NOT force it (we warn).
        # -----------------------------
        if len(shap_predicted_df) == expected_test_rows:
            shap_predicted_df = shap_predicted_df.copy()
            shap_predicted_df.index = x_test_preprocessed.index
        else:
            print(
                f"[WARN] iteration={iteration}: shap_predicted_df rows={len(shap_predicted_df)} "
                f"!= expected_test_rows={expected_test_rows}. "
                f"Row reduction is happening inside get_prediction_xai / SHAP pipeline."
            )

        # Add iteration and index columns
        shap_predicted_df["n_iteration"] = iteration
        shap_predicted_df["index"] = shap_predicted_df.index

        shap_predicted_df_full.append(shap_predicted_df)

    # Concatenate all iterations
    df_shap = pd.concat(shap_predicted_df_full, ignore_index=True)

    # All the shap values change to positive
    df_abs = df_shap.abs()
    df_abs = df_abs.drop(columns=["base_value"])

    df_values_global, df_ranking_global = compute_global_sra_rankstab(
        df_abs,
        drop_cols=["base_value"]  # optional
        )

    # Start SMLE
    # Rank the columns for each row with the highest value first, excluding specified columns
    rankstab_records, extra = compute_rankstab_local_from_shap(df_abs)

    df_resume_stab_global = df_ranking_global.reset_index()
    df_resume_stab_global = df_resume_stab_global[["index", "rankstab_global"]]

    value_stab_global = compute_value_stab(df_values_global)
    # Compute the mean of the feature columns (excluding 'index' and 'value_stab')
    df_resume_cv_global = value_stab_global.copy()
    df_resume_cv_global["feature_importance"] = df_resume_cv_global.drop(
        columns=["feature", "value_stab"]
    ).mean(axis=1)
    # Select only the desired columns
    df_resume_cv_global = df_resume_cv_global[
        ["feature", "feature_importance", "value_stab"]
    ]

    df_resume_global = df_resume_stab_global.merge(
        df_resume_cv_global, left_on="index", right_on="feature", how="left"
    )

    # Drop the 'index' column
    df_resume_global = df_resume_global.drop(columns="index")
    # Convert feature_importance to relative percentage
    total_importance = df_resume_global["feature_importance"].sum()
    df_resume_global["feature_importance"] = (
        df_resume_global["feature_importance"] / total_importance * 100
    )
    # Create rankstab_weight column
    df_resume_global["rankstab_global_weight"] = (
        df_resume_global["rankstab_global"] * df_resume_global["feature_importance"]
    )
    # Create rankstab_weight column
    df_resume_global["value_stab_weight"] = (
        df_resume_global["value_stab"] * df_resume_global["feature_importance"]
    )

    # df_resume_global = df_resume_global.reset_index()
    # df_resume_global = df_resume_global.rename(columns={'index': 'feature'})
    print(df_resume_global)
    print(list(df_resume_global.head(0)))
    df_ranking_local = pd.DataFrame(rankstab_records)
    extra = pd.DataFrame(extra)
    df_ranking_local["feature"] = df_ranking_local["feature"].astype("int64")
    extra["feature"] = extra["feature"].astype("int64")
    print(df_ranking_local.head())

    df_resume_local = df_ranking_local.merge(df_resume_global, on="feature", how="left")
    print(df_resume_local.head())
    print(list(df_resume_local.head(0)))
    df_resume_local = df_resume_local[
        ["feature", "index", "feature_importance", "rankstab_local"]
    ]

    # Create rankstab_weight column
    df_resume_local["rankstab_local_weight"] = (
        df_resume_local["rankstab_local"] * df_resume_local["feature_importance"]
    )

    df_resume_local_agg = df_resume_local.groupby(["index"]).sum()
    df_resume_local_agg = df_resume_local_agg.drop(
        columns=["feature", "feature_importance"]
    )
    df_resume_local_agg = df_resume_local_agg.reset_index().rename(
        columns={"index": "feature"}
    )

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


def compute_global_sra_rankstab(
    df_shap: pd.DataFrame,
    iteration_col: str = "n_iteration",
    index_col: str = "index",
    drop_cols: list = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute global SRA ranking + rank stability from a SHAP-per-iteration dataframe.

    Parameters
    ----------
    df_shap : pd.DataFrame
        Input dataframe containing SHAP values for test rows, repeated across iterations.
        Expected columns:
          - iteration_col (e.g. 'n_iteration')
          - index_col (e.g. 'index')  -> identifies the test row/observation
          - feature columns           -> SHAP values per feature
        May also include extra columns (e.g. 'base_value').

    iteration_col : str
        Name of the iteration column.

    index_col : str
        Name of the observation identifier column.

    drop_cols : list[str] | None
        Optional extra columns to drop before processing (e.g. ['base_value']).

    Returns
    -------
    df_values_global : pd.DataFrame
        Table with one row per feature and one column per iteration.
        Values = summed |SHAP| across all test rows (indexes) for that iteration.

        Columns: ['feature', <iter0>, <iter1>, ...].

    df_ranking_global : pd.DataFrame
        Same shape as df_values_global, but entries are ranks (1 = most important),
        plus an extra column 'rankstab_global' = stability of ranks across iterations.

        Columns: ['feature', <iter0>, <iter1>, ..., 'rankstab_global'].
    """

    # -----------------------------
    # 0) Defensive checks
    # -----------------------------
    if iteration_col not in df_shap.columns:
        raise ValueError(f"Missing '{iteration_col}' in df_shap")
    if index_col not in df_shap.columns:
        raise ValueError(f"Missing '{index_col}' in df_shap")

    df = df_shap.copy()

    # Drop unwanted columns (e.g. base_value)
    if drop_cols:
        existing = [c for c in drop_cols if c in df.columns]
        if existing:
            df = df.drop(columns=existing)

    # -----------------------------
    # 1) Absolute SHAP values
    #    (we only abs the feature columns)
    # -----------------------------
    feature_cols = [c for c in df.columns if c not in [iteration_col, index_col]]
    df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce").abs()

    # -----------------------------
    # 2) Global aggregation per iteration (SRA)
    #    IMPORTANT FIX:
    #    We group by iteration and sum across rows (different indexes),
    #    but we MUST NOT sum the index_col (identifier).
    # -----------------------------
    df_sum = (
        df.drop(columns=[index_col])          # drop index id before sum
          .groupby(iteration_col, as_index=True)
          .sum(numeric_only=True)             # sum features across test rows
    )

    # df_sum shape: [n_iterations x n_features]
    # index: iteration values, columns: features

    # -----------------------------
    # 3) Convert to feature x iteration table (as you did)
    # -----------------------------
    df_transpose = (
        df_sum.T
            .reset_index()
            .rename(columns={"index": "feature"})
    )

    # This is the global importance per iteration
    # Columns: feature, iter0, iter1, ...
    df_values_global = df_transpose.copy()

    # -----------------------------
    # 4) Rank features per iteration
    #    For each iteration column: higher importance => rank 1
    # -----------------------------
    df_ranking_global_mat = df_transpose.drop(columns=["feature"]).rank(
        axis=0, ascending=False, method="average"
    )

    # -----------------------------
    # 5) Rank stability across iterations (per feature)
    #    Your formula is equivalent to:
    #      sample variance of ranks across iterations
    # -----------------------------
    num_iterations = df_ranking_global_mat.shape[1]
    if num_iterations < 2:
        raise ValueError("Need at least 2 iterations to compute rankstab_global.")

    mean_ranks = df_ranking_global_mat.mean(axis=1)
    rankstab = ((df_ranking_global_mat.sub(mean_ranks, axis=0)) ** 2).sum(axis=1) / (num_iterations - 1)

    # Build final df_ranking_global output
    df_ranking_global = df_ranking_global_mat.copy()
    df_ranking_global["rankstab_global"] = rankstab.values
    df_ranking_global.insert(0, "feature", df_transpose["feature"].values)

    return df_values_global, df_ranking_global


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
