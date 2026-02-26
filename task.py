import pandas as pd

from churn.model.functions import load_model, save_model
from churn.explainability.functions import (
    get_train_xai,
    get_prediction_xai,
    get_train_iterations_xai, compose_shap_df_preds_file_name, compose_shap_model_output_file_name,
    compose_shap_file_name,
)

from churn.model.functions import get_split_data
from churn.utils import get_feature_columns, compose_data_file_path, compose_feature_cols_file_path, compose_model_path


def xai_task(repo, product):

    exclude_cols = repo.exclude_cols()
    project_name = repo.model_name()
    is_local = repo.is_local()
    date_key = repo.date_train()
    config = extract_config(repo)
    random_state = 123
    df_cols = load_feature_columns(compose_feature_cols_file_path(config['repo'], product), config["is_local"])
    data = []
    env = "local" if is_local else "gcp"

    print(f"XAI train - product: {product}")
    for algo in config["algorithms"]:
        print(f"Process product: {product} - Algorithm: {algo}")
        print("Get feature columns")
        id_cols, numerical_cols, categorical_cols, target_cols = (
            get_feature_columns(algo, df_cols=df_cols, date_key=date_key, project_name=project_name,
                                exclude_cols=exclude_cols, step="explainability")
        )

        print("Feature columns obtained")
        selected_ft_numerical = [ft for ft in numerical_cols if ft.strip()]
        selected_ft_categorical = [ft for ft in categorical_cols if ft.strip()]
        selected_ft = selected_ft_numerical + selected_ft_categorical
        print("Selected features prepared")
        print("Read training data")
        df = pd.read_parquet(compose_data_file_path(repo, product, period='train'))

        if not repo.flag_fake_data():
            df = df[df["date_period"] == "train"]

        print("Prepare data")
        x_train, x_test, y_train, y_test, _, _ = get_split_data(
            data=df,
            feature_cols=selected_ft,
            numerical_cols=numerical_cols,
            categorical_cols=categorical_cols,
            ids_cols=id_cols,
            target_cols=target_cols,
            exclude_cols=exclude_cols,
            random_state=random_state,
        )

        print("Load model for XAI")
        pipeline = load_model(compose_model_path(repo, algo, product), env)
        print("shap training")
        shap_transformer = get_train_xai(pipeline, x_train)
        print("Path shap training")
        path_shap = compose_shap_file_name(config['repo'], algo, product)
        print(f"Save shap model: {path_shap}")
        save_model(path_shap, shap_transformer, environment="local" if is_local else "gcp")


def extract_config(repo):
    return {
        "algorithms": repo.algorithm() if isinstance(repo.algorithm(), list) else [repo.algorithm()],
        "exclude_cols": repo.exclude_cols(),
        "test_size": repo.test_size(),
        "random_state": repo.random_state(),
        "project_name": repo.model_name(),
        "is_local": repo.is_local(),
        "date_key": repo.date_train(),
        "spark": repo._spark,
        "test_mode": repo.flag_test(),
        "train_mode": repo.flag_train(),
        "predict_mode": repo.flag_prediction(),
        "xai_enabled": repo.flag_XAI(),
        "fake_data": repo.flag_fake_data(),
        "repo": repo
    }

def load_feature_columns(path_cols, is_local):
    df = pd.read_parquet(path_cols)
    if not is_local:
        df["feature_list"] = df["feature_list"].apply(lambda x: x.split(",") if isinstance(x, str) else x)
    return df

def process_product(config, product, algo, df_cols):

    project_name = config["project_name"]
    date_key = config["date_key"]
    exclude_cols = config["exclude_cols"]
    test_size = config["test_size"]
    random_state = config["random_state"]
    is_local = config["is_local"]
    train_mode = config["train_mode"]
    predict_mode = config["predict_mode"]
    xai_enabled = config["xai_enabled"]
    fake_data = config["fake_data"]

    pipeline = load_model(compose_model_path(config['repo'], algo, product), "local" if is_local else "gcp")

    id_cols, numerical_cols, categorical_cols, target_cols = get_feature_columns(algo, df_cols, date_key, project_name,
                                                                                 exclude_cols, step="explainability")

    selected_ft_numerical = [ft for ft in numerical_cols if ft.strip()]
    selected_ft_categorical = [ft for ft in categorical_cols if ft.strip()]
    selected_ft = selected_ft_numerical + selected_ft_categorical

    if train_mode and xai_enabled:

        df = pd.read_parquet(compose_data_file_path(config['repo'], product, "train"))
        if not fake_data:
            df = df[df["date_period"] == "train"]

        x_train, x_test, y_train, y_test, _, _ = get_split_data(
            df, selected_ft, selected_ft_numerical, selected_ft_categorical,
            id_cols, target_cols, exclude_cols, test_size, random_state
        )
        shap_transformer = get_train_xai(pipeline, x_train)
        path_shap = compose_shap_file_name(config['repo'], algo, product)
        save_model(path_shap, shap_transformer, environment="local" if is_local else "gcp")

        shap_iterations, ranking_local, ranking_global, resume_local, resume_global, resume = get_train_iterations_xai(
            pipeline, x_train, x_test, y_train, n_iterations=50, percentage=80
        )

        resume_values = {
            "global_rankstab": float(resume.get("rankstab_global", pd.Series()).sum()),
            "global_value_cv": float(resume.get("value_stab", pd.Series()).sum()),
            "global_rankstab_weight": float(resume.get("rankstab_global_weight", pd.Series()).sum()),
            "global_value_cv_weight": float(resume.get("value_stab_weight", pd.Series()).sum()),
            "local_rankstab": float(resume.get("rankstab_local", pd.Series()).sum()),
            "local_rankstab_weight": float(resume.get("rankstab_local_weight", pd.Series()).sum()),
        }

        for shap_df_name, df_out in {
            "shap_values_test_iteration": shap_iterations,
            "shap_ranking_local_ft_stability": ranking_local,
            "shap_ranking_global_ft_stability": ranking_global,
            "shap_resume_local_ft_stability": resume_local,
            "shap_resume_global_ft_stability": resume_global,
            "shap_resume_ft_stability": resume
        }.items():
            df_out.to_parquet(compose_shap_model_output_file_name(config['repo'], shap_df_name, project_name, product, algo))

        return {
            "date_key": date_key,
            "project_name": project_name,
            "model": algo,
            **resume_values
        }

    if predict_mode and xai_enabled:
        print("Prediction Mode XAI")
        df = pd.read_parquet(compose_data_file_path(config['repo'], product, 'prediction'))

        x_train = pd.DataFrame(df[selected_ft])

        print("Load XAI Model")
        shap_transformer = load_model(compose_shap_file_name(config['repo'], algo, product), environment="local" if is_local else "gcp")

        print("Get XAI predictions")
        shap_df = get_prediction_xai(pipeline,
                                     shap_transformer,
                                     x_train,
                                     selected_ft)

        shap_df = calculate_row_percentage(shap_df)

        if not fake_data:
            print("Add the 'party_nbr' column from the original data")
            if isinstance(df, pd.DataFrame) and "party_nbr" in df.columns:
                shap_df["party_nbr"] = df["party_nbr"].values

        print("Save XAI predictions")
        shap_df.to_parquet(compose_shap_df_preds_file_name(algo, config, product, project_name))

    return None


def calculate_row_percentage(df):
    # Calculate the row-wise sum
    row_sums = df.sum(axis=1)

    # Divide each element by the row sum to get the percentage
    percentage_df = df.div(row_sums, axis=0)

    return percentage_df


def save_xai_metrics(config, data):

    train_mode = config["train_mode"]
    xai_enabled = config["xai_enabled"]
    spark = config["spark"]
    repo = config["repo"]

    if train_mode and xai_enabled:
        spark_df = spark.createDataFrame(data)
        repo.save_table(df=spark_df, key="xai_metrics", mode="overwrite", return_table=False)