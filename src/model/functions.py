import os
import numpy as np
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.base import clone  # <-- FIX: sklearn clone (not joblib.clone)

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
from sklearn.metrics import average_precision_score

# Optional deps
try:
    from lightgbm import LGBMClassifier
except Exception as e:
    LGBMClassifier = None
    _lightgbm_import_error = e

try:
    from xgboost import XGBClassifier
except Exception as e:
    XGBClassifier = None
    _xgboost_import_error = e

try:
    from skopt import BayesSearchCV
    from skopt.space import Space, Integer, Real, Categorical
except ImportError as e:
    BayesSearchCV = None
    Space = None
    Integer = None
    Real = None
    Categorical = None
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

# Optional GCP dep
try:
    import gcsfs  # type: ignore
except ImportError as e:
    gcsfs = None
    _gcsfs_import_error = e


# -------------------------------
# Picklable helper (module-level)
# -------------------------------
def to_dense_float(X):
    """Convert sparse->dense (if needed) and cast to float64 for MLP stability."""
    if hasattr(X, "toarray"):  # scipy sparse
        X = X.toarray()
    return np.asarray(X, dtype=np.float64)


def normalize_search_space(space: dict) -> dict:
    """
    Convert tuple/list shorthand into explicit skopt dimensions.
    """
    if Integer is None or Real is None or Categorical is None:
        raise ImportError(
            "skopt.space Integer/Real/Categorical not available. "
            "Install scikit-optimize (pip install scikit-optimize). "
            f"Original error: {_skopt_import_error}"
        )

    out = {}
    for k, v in space.items():
        if isinstance(v, (Integer, Real, Categorical)):
            out[k] = v
            continue

        if isinstance(v, tuple) and len(v) == 2:
            low, high = v
            if isinstance(low, (int, np.integer)) and isinstance(high, (int, np.integer)):
                out[k] = Integer(int(low), int(high))
            else:
                out[k] = Real(float(low), float(high))
            continue

        if isinstance(v, (list, tuple)):
            out[k] = Categorical(list(v))
            continue

        raise ValueError(f"Unsupported search-space spec for {k}: {v!r}")

    return out


def _validate_search_space(search_space: dict):
    if Space is None:
        raise ImportError(
            "skopt.space.Space is not available. "
            "Install scikit-optimize (pip install scikit-optimize). "
            f"Original error: {_skopt_import_error}"
        )
    if not isinstance(search_space, dict) or len(search_space) == 0:
        raise ValueError(f"search_space must be a non-empty dict. Got: {search_space!r}")
    Space(list(search_space.values()))  # raises if invalid


def sample_one_params(search_space: dict, random_state: int = 123) -> dict:
    """
    Draw 1 random point from a skopt search_space dict and return as params dict.
    """
    if Space is None:
        raise ImportError(
            "skopt.space.Space is not available. "
            "Install scikit-optimize (pip install scikit-optimize). "
            f"Original error: {_skopt_import_error}"
        )

    rng = np.random.RandomState(random_state)

    keys = list(search_space.keys())
    dims = list(search_space.values())
    space = Space(dims)

    point = space.rvs(n_samples=1, random_state=rng)[0]
    return dict(zip(keys, point))


def get_model(algorithm="RandomForest", random_state=None, **kwargs):
    if algorithm == "RandomForest":
        return RandomForestClassifier(random_state=random_state, **kwargs)

    if algorithm == "LightGBM":
        if LGBMClassifier is None:
            raise ImportError(
                "LightGBM is not available. "
                "Have you installed libomp (e.g. `brew install libomp`)? "
                f"Original error: {_lightgbm_import_error}"
            )
        return LGBMClassifier(random_state=random_state, eval_metric="binary_logloss", **kwargs)

    if algorithm == "XGBoost":
        if XGBClassifier is None:
            raise ImportError("XGBoost is not available. " f"Original error: {_xgboost_import_error}")
        return XGBClassifier(
            random_state=random_state,
            use_label_encoder=False,
            eval_metric="aucpr",
            **kwargs,
        )

    if algorithm == "BoostDT":
        return GradientBoostingClassifier(random_state=random_state, subsample=1, **kwargs)

    if algorithm == "BoostSGB":
        return GradientBoostingClassifier(random_state=random_state, **kwargs)

    if algorithm == "BagDT":
        return BaggingClassifier(estimator=DecisionTreeClassifier(), random_state=random_state, **kwargs)

    if algorithm == "BagNN":
        return BaggingClassifier(
            estimator=MLPClassifier(random_state=random_state),
            random_state=random_state,
            **kwargs,
        )

    if algorithm == "AdaBoost-NN":
        return AdaBoostClassifier(
            estimator=MLPClassifier(random_state=random_state),
            random_state=random_state,
            **kwargs,
        )

    if algorithm == "AdaBoost-DT":
        return AdaBoostClassifier(estimator=DecisionTreeClassifier(), random_state=random_state, **kwargs)

    if algorithm == "SVM-Rbf":
        return SVC(random_state=random_state, probability=True, **kwargs)

    if algorithm == "ANN":
        return MLPClassifier(random_state=random_state, **kwargs)

    raise ValueError(f"Unsupported algorithm: {algorithm}")


def create_pipeline(
    algorithm="RandomForest",
    numerical_features=None,
    categorical_features=None,
    random_state=123,
):
    if NumericTransformer is None:
        raise ImportError(
            "NumericTransformer is not available (src.model.encoders import failed). "
            f"Original error: {_churn_import_error}"
        )

    model = get_model(algorithm=algorithm, random_state=random_state)

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
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

    needs_dense = algorithm in {"ANN", "BagNN", "AdaBoost-NN"}

    if needs_dense:
        return Pipeline(
            [
                ("preprocessing", preprocessor),
                ("to_dense_float", FunctionTransformer(to_dense_float, validate=False)),
                ("model", model),
            ]
        )

    return Pipeline([("preprocessing", preprocessor), ("model", model)])


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
    if not isinstance(target_cols, (list, tuple)) or len(target_cols) != 1:
        raise ValueError(f"target_cols must be a list with exactly 1 column. Got: {target_cols}")

    exclude_cols = list(exclude_cols) + list(ids_cols) + list(target_cols)

    features = [col for col in feature_cols if col not in exclude_cols]
    numerical = [col for col in numerical_cols if col not in exclude_cols]
    categorical = [col for col in categorical_cols if col not in exclude_cols]

    x = data[features].copy()
    y = data[target_cols[0]].astype(int).copy()  # y must be 1D

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, stratify=y, random_state=random_state
    )

    return x_train, x_test, y_train, y_test, numerical, categorical


def get_search_space(algorithm, product, test_mode):
    if search_spaces is None or search_spaces_test is None:
        raise ImportError(
            "Search spaces could not be imported. "
            "Make sure src.model.search_spaces and src.model.search_spaces_test exist. "
            f"Original error: {_churn_import_error}"
        )

    spaces = search_spaces_test if test_mode else search_spaces

    if algorithm not in spaces:
        raise ValueError(f"No search space for algorithm: {algorithm}{' on test mode' if test_mode else ''}")

    alg_params = spaces[algorithm]
    if product in alg_params:
        return alg_params[product]
    if "default" in alg_params:
        return alg_params["default"]

    raise ValueError(f"No search space for product={product} algorithm={algorithm}")


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
    ann_random_search_iter: int = 25,  # used only for ANN (no BayesOpt)
    **kwargs,
):
    """
    Train model with hyperparameter search.

    - For ANN: avoids Bayesian optimization. Uses random search over the same skopt space + CV.
    - For others: BayesSearchCV as before.
    """
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

    search_space_raw = get_search_space(algorithm, product, test_mode)
    search_space = normalize_search_space(search_space_raw)
    print(f"Search space for algorithm={algorithm}, product={product}, test_mode={test_mode}: {search_space}")

    # ---------------------------------------------------------------------
    # ANN: avoid BayesSearchCV entirely -> random search on the defined space
    # ---------------------------------------------------------------------
    if algorithm == "ANN":
        print("Byeasian for ANN is unstable, using random search instead (no BayesSearchCV).")
        _validate_search_space(search_space)

        # keep it small on test_mode
        n_iter = 3 if test_mode else int(ann_random_search_iter)

        best_params = None
        best_score = -np.inf

        # CV loop (comparable to BayesSearchCV scoring)
        n_splits = 2 if test_mode else 5
        cv_strategy = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        for i in range(n_iter):
            params_i = sample_one_params(search_space, random_state=random_state + i)
            pipeline_i = clone(pipeline)
            pipeline_i.set_params(**params_i)

            fold_scores = []
            for train_idx, val_idx in cv_strategy.split(x_train, y_train):
                X_tr = x_train.iloc[train_idx]
                y_tr = y_train.iloc[train_idx]
                X_va = x_train.iloc[val_idx]
                y_va = y_train.iloc[val_idx]

                pipeline_i.fit(X_tr, y_tr)
                y_score = pipeline_i.predict_proba(X_va)[:, 1]
                fold_scores.append(average_precision_score(y_va, y_score))

            mean_score = float(np.mean(fold_scores))

            if mean_score > best_score:
                best_score = mean_score
                best_params = params_i

        if best_params is None:
            raise RuntimeError("ANN random search failed to produce best_params.")

        best_pipeline = clone(pipeline)
        best_pipeline.set_params(**best_params)
        best_pipeline.fit(x_train, y_train)

        return best_pipeline, best_params, best_score

    # -------------------------------
    # NON-ANN: normal BayesSearchCV mode
    # -------------------------------
    _validate_search_space(search_space)

    n_iter = 10 if test_mode else 30
    n_splits = 3 if test_mode else 5
    cv_strategy = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    if BayesSearchCV is None:
        raise ImportError(
            "skopt.BayesSearchCV is not available. "
            "Install scikit-optimize (pip install scikit-optimize). "
            f"Original error: {_skopt_import_error}"
        )

    n_jobs = 1 if algorithm in {"BagNN", "AdaBoost-NN"} else -1

    bayes_search = BayesSearchCV(
        estimator=pipeline,
        search_spaces=search_space,
        n_iter=n_iter,
        scoring="average_precision",
        cv=cv_strategy,
        random_state=random_state,
        verbose=0,
        n_jobs=n_jobs,
        n_points=1,
        error_score="raise",
    )

    bayes_search.fit(x_train, y_train)

    best_params = bayes_search.best_params_
    best_score = bayes_search.best_score_
    best_pipeline = bayes_search.best_estimator_

    return best_pipeline, best_params, best_score


def save_model(path: str, model, environment: str = "local"):
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
    try:
        if environment == "gcp":
            if gcsfs is None:
                raise ImportError(
                    "gcsfs is required for loading from GCP but is not installed. "
                    f"Original error: {_gcsfs_import_error}"
                )
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