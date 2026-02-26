import os

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

# Import optional libraries with fallbacks in case the shared libs are missing
try:
    from lightgbm import LGBMClassifier
except Exception as e:  # could be OSError due to missing libomp or ImportError
    LGBMClassifier = None
    _lightgbm_import_error = e

try:
    from xgboost import XGBClassifier
except Exception as e:
    XGBClassifier = None
    _xgboost_import_error = e

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier

from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
# optional dependencies that may not be installed in every environment
try:
    from skopt import BayesSearchCV
except ImportError as e:
    BayesSearchCV = None
    _skopt_import_error = e

try:
    from src.model.search_spaces import search_spaces
    from src.model.search_spaces_test import search_spaces as search_spaces_test
    from src.model.encoders import NumericTransformer
except ImportError as e:
    # churn is a custom package; if it's missing we set names to None and
    # raise later if they are actually required.
    search_spaces = None
    search_spaces_test = None
    NumericTransformer = None
    _churn_import_error = e

import pandas as pd
try:
    import gcsfs
except ImportError as e:
    gcsfs = None
    _gcsfs_import_error = e
import joblib

def get_model(algorithm="RandomForest", random_state=None, **kwargs):
    """
    Select the machine learning algorithm based on the provided name.

    Parameters:
    - algorithm: The name of the algorithm ('RandomForest', 'LightGBM', 'XGBoost').
    - random_state: Random seed for reproducibility.
    - kwargs: Additional keyword arguments for ML models.

    Returns:
    - An instance of the selected algorithm.
    """
    if algorithm == "RandomForest":
        model = RandomForestClassifier(random_state=random_state, **kwargs)
    elif algorithm == "LightGBM":
        if LGBMClassifier is None:
            raise ImportError(
                "LightGBM is not available. "
                "Have you installed libomp (e.g. `brew install libomp`)? "
                f"Original error: {_lightgbm_import_error}"
            )
        model = LGBMClassifier(
            random_state=random_state, eval_metric="binary_logloss", **kwargs
        )
    elif algorithm == "XGBoost":
        if XGBClassifier is None:
            raise ImportError(
                "XGBoost is not available. "
                f"Original error: {_xgboost_import_error}"
            )
        model = XGBClassifier(
            random_state=random_state,
            use_label_encoder=False,
            eval_metric="aucpr",
            **kwargs,
        )
    elif algorithm == "BoostDT":
        model = GradientBoostingClassifier(
            random_state=random_state, subsample=1, **kwargs
        )
    elif algorithm == "BoostSGB":
        model = GradientBoostingClassifier(random_state=random_state, **kwargs)
    elif algorithm == "BagDT":
        model = BaggingClassifier(
            estimator=DecisionTreeClassifier(), random_state=random_state, **kwargs
        )
    elif algorithm == "BagNN":
        model = BaggingClassifier(
            estimator=MLPClassifier(random_state=random_state),
            random_state=random_state,
            **kwargs,
        )
    elif algorithm == "AdaBoost-NN":
        model = AdaBoostClassifier(
            estimator=MLPClassifier(random_state=random_state),
            random_state=random_state,
            **kwargs,
        )
    elif algorithm == "AdaBoost-DT":
        model = AdaBoostClassifier(
            estimator=DecisionTreeClassifier(), random_state=random_state, **kwargs
        )
    elif algorithm == "SVM-Rbf":
        model = SVC(random_state=random_state, probability=True, **kwargs)
    elif algorithm == "ANN":
        model = MLPClassifier(random_state=random_state, **kwargs)
    else:
        raise ValueError("Unsupported algorithm.")

    return model


def create_pipeline(
    algorithm="RandomForest",
    numerical_features=None,
    categorical_features=None,
    random_state=123,
):
    """
    Creates a sklearn pipeline that preprocesses data and trains a machine learning model.

    Parameters:
    - algorithm (str): Name of the algorithm ('RandomForest', 'LightGBM', 'XGBoost').
    - numerical_features (list): List of numerical feature column names.
    - categorical_features (list): List of categorical feature column names.
    - random_state (int): Random seed for reproducibility.

    Returns:
    - pipeline (Pipeline): A scikit-learn pipeline for preprocessing and model training.
    """

    # Choose the model
    model = get_model(
        algorithm=algorithm,
        random_state=random_state,
    )

    # Preprocessing steps for categorical features: Imputation (most frequent) and Ordinal Encoding
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            # Impute missing categorical values with the most frequent
            (
                "ordinal",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            ),  # Apply Ordinal Encoding
        ]
    )

    # Preprocessing steps for numerical features: Imputation (median) and Scaling
    numerical_transformer = Pipeline(
        steps=[
            (
                "imputer_outlier",
                NumericTransformer(
                    method_nans="simple",
                    replace_outliers=True,
                    outliers_lower_quantile=0.01,
                    outliers_upper_quantile=0.99,
                ),
            ),  # Impute missing numerical values with the median
            ("scaler", StandardScaler()),  # Standardize the numerical features
        ]
    )

    # Combine the preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                categorical_transformer,
                categorical_features,
            ),
            (
                "num",
                numerical_transformer,
                numerical_features,
            ),
        ]
    )
    pipeline = Pipeline(
        [
            ("preprocessing", preprocessor),
            ("model", model),
        ]
    )

    return pipeline


def get_split_data(
    data,
    feature_cols: list,
    numerical_cols: list,
    categorical_cols: list,
    ids_cols:list,
    target_cols,
    exclude_cols,
    test_size=0.3,
    random_state=123,
):
    """
    Prepare the dataset for training by creating X (features) and y (target),
    excluding specified columns, and splitting into train and test sets.

    Parameters:
    - data (pd.DataFrame): Input dataset.
    - feature_cols (list): List of all feature columns.
    - numerical_cols (list): List of numerical feature columns.
    - categorical_cols (list): List of categorical feature columns.
    - ids_cols (list): List of ID columns to exclude.
    - target_cols (list): List of target columns.
    - exclude_cols (list): List of additional columns to exclude.
    - test_size (float): Proportion of the dataset to include in the test split (default: 0.3).
    - random_state (int): Random seed for reproducibility (default: 123).

    Returns:
    - x_train (pd.DataFrame): Training feature data.
    - x_test (pd.DataFrame): Testing feature data.
    - y_train (pd.DataFrame): Training target data.
    - y_test (pd.DataFrame): Testing target data.
    - numerical (list): Processed numerical columns after exclusion.
    - categorical (list): Processed categorical columns after exclusion.
    """
    print(type(exclude_cols), type(ids_cols), type(target_cols))

    print("Exclude specified columns")
    exclude_cols = exclude_cols + ids_cols + target_cols
    print(f"exclude_cols: {exclude_cols}")
    features = [col for col in feature_cols if col not in exclude_cols]
    print(f"features: {features}")
    numerical = [col for col in numerical_cols if col not in exclude_cols]
    print(f"numerical: {numerical}")
    categorical = [col for col in categorical_cols if col not in exclude_cols]
    print(f"categorical: {categorical}")

    print(f"features: {features}")
    x = pd.DataFrame(data[features])
    y = pd.DataFrame(data[target_cols])

    print("Split into train and test sets")
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, stratify=y, random_state=random_state
    )

    return x_train, x_test, y_train, y_test, numerical, categorical


def get_train_model(
    data,
    feature_cols,
    numerical_cols,
    categorical_cols,
    ids_cols,
    target_cols,
    exclude_cols,
    product,
    algorithm="RandomForest",
    test_mode=False,
    **kwargs,
):

    random_state = 123

    print("Prepare data")
    x_train, x_test, y_train, y_test, numerical, categorical = get_split_data(
        data=data,
        feature_cols=feature_cols,
        numerical_cols=numerical_cols,
        categorical_cols=categorical_cols,
        ids_cols=ids_cols,
        target_cols=target_cols,
        exclude_cols=exclude_cols,
        random_state=random_state,
    )

    print(f"Select the algorithm: {algorithm}")
    pipeline = create_pipeline(
        algorithm=algorithm,
        numerical_features=numerical,
        categorical_features=categorical,
        random_state=random_state,
    )

    search_space = get_search_space(algorithm, product, test_mode)
    print(f'Search space for the algorithm {algorithm}, product {product}: {search_space}')

    if test_mode:
        n_iter = 1
        n_splits = 2
    else:
        n_iter = 50
        n_splits = 5

    print("CV strategy")
    cv_strategy = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )

    print("Bayes search")
    if BayesSearchCV is None:
        raise ImportError(
            "skopt.BayesSearchCV is not available. "
            "Install scikit-optimize (e.g. `pip install scikit-optimize`)" 
            f". Original error: {_skopt_import_error}"
        )
    bayes_search = BayesSearchCV(
        estimator=pipeline,
        search_spaces=search_space,
        n_iter=n_iter,
        scoring="average_precision",
        cv=cv_strategy,
        random_state=random_state,
        verbose=0,
        n_jobs=-1,
    )

    # Fit the model
    bayes_search.fit(x_train, y_train)

    # Save the best params and scores with the bayesian optimization and
    best_params = bayes_search.best_params_
    best_scores = bayes_search.best_score_
    pipeline = bayes_search.best_estimator_

    return pipeline, best_params, best_scores


def get_search_space(algorithm, product, test):
    # ensure that the search space definitions are available
    if search_spaces is None or search_spaces_test is None:
        raise ImportError(
            "Required module `churn` not found; search spaces cannot be retrieved. "
            "Install the appropriate package or make sure it is on PYTHONPATH. "
            f"Original error: {_churn_import_error}"
        )

    search_spaces_all = {
        True: search_spaces_test,
        False: search_spaces
    }

    if algorithm not in search_spaces_all[test]:
        raise ValueError(f"No search space for algorithm: {algorithm}{' on test mode' if test else ''}")

    alg_params = search_spaces_all[test][algorithm]
    if product in alg_params:
        return alg_params[product]
    elif 'default' in alg_params:
        return alg_params['default']
    else:
        raise ValueError(f"No search space for {product} and the algorithm {algorithm}{' on test mode' if test else ''}")


def save_model(path: str, model, environment: str = "local"):
    """
    Save a model to a specified path, supporting both local and GCP environments.

    :param path: Path to save the model. For GCP, this should include the bucket name (e.g., 'bucket-name/path/to/file').
    :param model: The model object to be saved.
    :param environment: Specify 'local' for local filesystem or 'gcp' for Google Cloud Storage.
    """
    try:
        if environment == "gcp":
            if gcsfs is None:
                raise ImportError(
                    "gcsfs is required for saving to GCP but is not installed. "
                    f"Original error: {_gcsfs_import_error}"
                )
            fs = gcsfs.GCSFileSystem()
            with fs.open(f"{path}", "wb") as f:
                joblib.dump(model, f)
            print(f"Model saved successfully to GCP at {path}")
        elif environment == "local":
            # Ensure the local directory exists
            local_dir = os.path.dirname(path)
            os.makedirs(local_dir, exist_ok=True)

            with open(path, "wb") as f:
                joblib.dump(model, f)
            print(f"Model saved successfully to local path {path}")
        else:
            raise ValueError("Unsupported environment. Use 'local' or 'gcp'.")
    except Exception as e:
        print(f"Failed to save model: {e}")


def load_model(path: str, environment: str = "local"):
    """
    Load a model from a specified path, supporting both local and GCP environments.

    :param path: Path to load the model from. For GCP, this should include the bucket name (e.g., 'bucket-name/path/to/file').
    :param environment: Specify 'local' for local filesystem or 'gcp' for Google Cloud Storage.
    :return: The loaded model object, or None if loading fails.
    """
    try:
        if environment == "gcp":
            fs = gcsfs.GCSFileSystem()
            with fs.open(f"{path}", "rb") as f:
                model = joblib.load(f)
            print(f"Model loaded successfully from GCP at {path}")
        elif environment == "local":
            with open(path, "rb") as f:
                model = joblib.load(f)
            print(f"Model loaded successfully from local path {path}")
        else:
            raise ValueError("Unsupported environment. Use 'local' or 'gcp'.")

        return model
    except Exception as e:
        print(f"Failed to load model: {e}")
        return None