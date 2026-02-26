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


def get_train_iterations_xai(
    pipeline, x_train, x_test, y_train, n_iterations=5, percentage=90
):
    # Compute SHAP transformer
    x_train_preprocessed = pipeline.named_steps["preprocessing"].transform(x_train)
    x_test_preprocessed = pipeline.named_steps["preprocessing"].transform(x_test)
    feature_names = (
        x_train_preprocessed.columns.tolist()
        if hasattr(x_train_preprocessed, "columns")
        else None
    )

    shap_predicted_df_full = []

    sample_size = int(len(x_train_preprocessed) * (percentage / 100))
    unique_samples = []

    for iteration in range(n_iterations):
        while True:
            sampled_data = pd.DataFrame(x_train_preprocessed).sample(
                n=sample_size, random_state=iteration
            )
            if not any(
                sampled_data.equals(existing_sample)
                for existing_sample in unique_samples
            ):
                unique_samples.append(sampled_data)
                break

        # Get matching target values for the sampled rows
        sampled_y = y_train.iloc[sampled_data.index]

        # Clone and train a new model on the sampled data
        model_clone = clone(pipeline.named_steps["model"])
        model_clone.fit(sampled_data, sampled_y)

        # Create a fresh SHAPTransformer with the new model
        shap_transformer = SHAPTransformer(model_clone)
        shap_transformer.fit(sampled_data)

        # Calculate the shap predicted for every iteration
        shap_predicted_df = get_prediction_xai(
            pipeline,
            shap_transformer,
            x_test_preprocessed,
            feature_names,
            preprocessed_data=True,
        )

        # Add iteration and index columns
        shap_predicted_df["n_iteration"] = iteration
        shap_predicted_df["index"] = shap_predicted_df.index

        shap_predicted_df_full.append(shap_predicted_df)

    df_shap = pd.concat(shap_predicted_df_full, ignore_index=True)

    # All the shap values change to positive
    df_abs = df_shap.abs()
    df_abs = df_abs.drop(columns=["base_value"])

    # Start SRA
    df = df_abs.groupby(["n_iteration"]).sum()
    df = df.drop(columns=["index"]).reset_index()
    df_transpose = (
        df.set_index("n_iteration").T.reset_index().rename(columns={"index": "feature"})
    )
    df_values_global = df_transpose.copy()
    df_ranking_global = df_transpose.drop(columns=["feature"]).rank(
        axis=0, ascending=False
    )

    # Calculate the rankstab for each row (iteration)
    num_columns = len(df_ranking_global.columns)
    rankstab_values = []

    for index, row in df_ranking_global.iterrows():
        sum_row_values = row.sum()
        rankstab = (1 / (num_columns - 1)) * sum(
            ((row[col] - (sum_row_values * (1 / num_columns))) ** 2)
            for col in row.index
        )
        rankstab_values.append(rankstab)

    df_ranking_global["rankstab_global"] = rankstab_values
    df_ranking_global = df_ranking_global.reset_index().rename(
        columns={"index": "feature"}
    )

    # Start SMLE
    # Rank the columns for each row with the highest value first, excluding specified columns
    exclude_columns = ["index", "n_iteration"]
    df = df_abs.copy()
    # Convert feature columns to float (replace commas with dots)
    feature_columns = df.columns.difference(exclude_columns)
    # Rank the features for each row
    df[feature_columns] = df[feature_columns].rank(axis=1, ascending=False)

    # Compute rankstab for each index
    rankstab_records = []
    for idx in df["index"].unique():
        df_sub = df[df["index"] == idx].drop(columns=["n_iteration", "index"])
        df_transposed = df_sub.T
        df_transposed.columns = range(1, len(df_transposed.columns) + 1)
        # Apply rankstab formula
        num_columns = df_transposed.shape[1]
        for _, row in df_transposed.iterrows():
            sum_row_values = row.sum()
            rankstab = (1 / (num_columns - 1)) * sum(
                ((row[col] - (sum_row_values * (1 / num_columns))) ** 2)
                for col in row.index
            )
            rankstab_records.append(
                {"feature": row.name, "index": idx, "rankstab_local": rankstab}
            )

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
    df_ranking_local["feature"] = df_ranking_local["feature"].astype("int64")
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
        df_ranking_global,
        df_resume_local,
        df_resume_global,
        df_resume,
    )


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
