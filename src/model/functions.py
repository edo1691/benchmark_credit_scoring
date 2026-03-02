import os

import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    BaggingClassifier,
    AdaBoostClassifier,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import train_test_split, StratifiedKFold

# Optional dependencies
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

try:
    from skopt import BayesSearchCV
    from skopt.space import Space  # <- version-agnostic validation helper
except ImportError as e:
    BayesSearchCV = None
    Space = None
    _skopt_import_error = e

try:
    from src.model.search_spaces import search_spaces
    from src.model.search_spaces_test import search_spaces as search_spaces_test
    from src.model.encoders import NumericTransformer
except ImportError as e:
    search_spaces = None
    search_spaces_test = None
    NumericTransformer = None
    _churn_import_error = e

# GCP optional
try:
    print(
        "Attempting to import gcsfs for GCP support. If you don't have gcsfs installed, "
    )
except ImportError as e:
    gcsfs = None
    _gcsfs_import_error = e


def get_model(algorithm="RandomForest", random_state=None, **kwargs):
    """
    Select the machine learning algorithm based on the provided name.
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
                "XGBoost is not available. " f"Original error: {_xgboost_import_error}"
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
            estimator=DecisionTreeClassifier(),
            random_state=random_state,
            **kwargs,
        )

    elif algorithm == "BagNN":
        model = BaggingClassifier(
            estimator=MLPClassifier(random_state=random_state),
            random_state=random_state,
            **kwargs,
        )

    elif algorithm == "AdaBoost-NN":
        # NOTE: AdaBoost passes sample_weight to the base estimator.
        # Depending on sklearn version, MLPClassifier may NOT support sample_weight.
        model = AdaBoostClassifier(
            estimator=MLPClassifier(random_state=random_state),
            random_state=random_state,
            **kwargs,
        )

    elif algorithm == "AdaBoost-DT":
        model = AdaBoostClassifier(
            estimator=DecisionTreeClassifier(),
            random_state=random_state,
            **kwargs,
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
    Creates a sklearn pipeline that preprocesses data and trains a model.
    """
    if NumericTransformer is None:
        raise ImportError(
            "NumericTransformer is not available (src.model.encoders import failed). "
            f"Original error: {_churn_import_error}"
        )

    model = get_model(algorithm=algorithm, random_state=random_state)

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "ordinal",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            ),
        ]
    )

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
            ),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, categorical_features),
            ("num", numerical_transformer, numerical_features),
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
    data: pd.DataFrame,
    feature_cols: list,
    numerical_cols: list,
    categorical_cols: list,
    ids_cols: list,
    target_cols: list,
    exclude_cols: list,
    test_size=0.3,
    random_state=123,
):
    """
    Prepare X/y and split train/test.
    IMPORTANT: y is returned as a 1D Series (required for robust sklearn scoring).
    """
    if not isinstance(target_cols, (list, tuple)) or len(target_cols) != 1:
        raise ValueError(
            f"target_cols must be a list with exactly 1 column. Got: {target_cols}"
        )

    exclude_cols = list(exclude_cols) + list(ids_cols) + list(target_cols)

    features = [col for col in feature_cols if col not in exclude_cols]
    numerical = [col for col in numerical_cols if col not in exclude_cols]
    categorical = [col for col in categorical_cols if col not in exclude_cols]

    x = data[features].copy()

    # y MUST be 1D for clean sklearn behavior
    y = data[target_cols[0]].astype(int).copy()

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, stratify=y, random_state=random_state
    )

    return x_train, x_test, y_train, y_test, numerical, categorical


def _validate_search_space(search_space: dict):
    """
    Validate skopt dimensions early so you don't get
    'Not all points are within the bounds of the space.' mid-optimization.

    Uses skopt.space.Space, which exists across skopt versions.
    """
    if Space is None:
        raise ImportError(
            "skopt.space.Space is not available. "
            "Install scikit-optimize (pip install scikit-optimize). "
            f"Original error: {_skopt_import_error}"
        )

    if not isinstance(search_space, dict) or len(search_space) == 0:
        raise ValueError(f"search_space must be a non-empty dict. Got: {search_space!r}")

    # Version-agnostic validation: will raise on invalid dimensions
    Space(list(search_space.values()))


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

    pipeline = create_pipeline(
        algorithm=algorithm,
        numerical_features=numerical,
        categorical_features=categorical,
        random_state=random_state,
    )

    search_space = get_search_space(algorithm, product, test_mode)
    print(
        f"Search space for algorithm={algorithm}, product={product}, test_mode={test_mode}: {search_space}"
    )

    # Validate space NOW (prevents skopt bound errors later)
    _validate_search_space(search_space)

    if test_mode:
        n_iter = 1
        n_splits = 2
    else:
        n_iter = 50
        n_splits = 5

    cv_strategy = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )

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
        error_score="raise",  # debug mode; switch to -np.inf after everything is stable
    )

    bayes_search.fit(x_train, y_train)

    best_params = bayes_search.best_params_
    best_score = bayes_search.best_score_
    best_pipeline = bayes_search.best_estimator_

    return best_pipeline, best_params, best_score


def get_search_space(algorithm, product, test):
    """
    Pull search space from search_spaces_test when test=True, else search_spaces.
    """
    if search_spaces is None or search_spaces_test is None:
        raise ImportError(
            "Search spaces could not be imported. "
            "Make sure src.model.search_spaces and src.model.search_spaces_test exist. "
            f"Original error: {_churn_import_error}"
        )

    search_spaces_all = {True: search_spaces_test, False: search_spaces}

    if algorithm not in search_spaces_all[test]:
        raise ValueError(
            f"No search space for algorithm: {algorithm}{' on test mode' if test else ''}"
        )

    alg_params = search_spaces_all[test][algorithm]
    if product in alg_params:
        return alg_params[product]
    if "default" in alg_params:
        return alg_params["default"]

    raise ValueError(
        f"No search space for {product} and algorithm {algorithm}{' on test mode' if test else ''}"
    )


def save_model(path: str, model, environment: str = "local"):
    """
    Save a model to a specified path, supporting both local and GCP environments.
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