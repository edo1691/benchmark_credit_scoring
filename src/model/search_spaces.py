from skopt.space import Categorical, Real, Integer

search_spaces = {
    "XGBoost": {
        "default": {
            "model__n_estimators": (50, 200),
            "model__max_depth": (3, 8),
            "model__learning_rate": Real(0.01, 0.1, prior="log-uniform"),
            "model__scale_pos_weight": (1, 10),
            "model__gamma": (1.0, 5),
            "model__subsample": (0.5, 1.0),
            "model__colsample_bytree": (0.5, 1.0),
            "model__colsample_bynode": (0.3, 1.0),
            "model__colsample_bylevel": (0.3, 1.0),
            "model__min_child_weight": (10, 25),
            "model__reg_alpha": (1.0, 10.0),
            "model__reg_lambda": (5.0, 10.0),
            "model__booster": ["gbtree"],
        },
        #  "close_saving": {
        #             "model__n_estimators": (20, 40),
        #     "model__max_depth": (2, 5),
        #     "model__learning_rate": Real(0.01, 0.1, prior="log-uniform"),
        #    "model__scale_pos_weight": (1, 10),
        #    "model__gamma": (1.0, 5),
        #    "model__subsample": (0.5, 1.0),
        #     "model__colsample_bytree": (0.5, 1.0),
        #     "model__colsample_bynode": (0.3, 1.0),
        #     "model__colsample_bylevel": (0.3, 1.0),
        #     "model__min_child_weight": (10, 25),
        #     "model__reg_alpha": (1.0, 10.0),
        #     "model__reg_lambda": (5.0, 10.0),
        #     "model__booster": ["gbtree"],
        # },

# In case the new_model uses a different search space from the default, uncomment and set the parameters below
#        "new_model": {
#            "model__n_estimators": (20, 40, 70, 120),
#            "model__max_depth": (3, 5, 8),
#            "model__learning_rate": Real(0.01, 0.1, prior="log-uniform"),
#            "model__scale_pos_weight": (1, 5, 10),
#            "model__gamma": (1.0, 2, 5),
#            "model__subsample": (0.5, 1.0),
#            "model__colsample_bytree": (0.5, 1.0),
#            "model__colsample_bynode": (0.3, 1.0),
#            "model__colsample_bylevel": (0.3, 1.0),
#            "model__min_child_weight": (10, 25),
#            "model__reg_alpha": (1.0, 10.0),
#            "model__reg_lambda": (5.0, 10.0),
#            "model__booster": ["gbtree"],
#        }
    },



    "RandomForest": {
        "default": {
            "model__n_estimators": (50, 300),
            "model__max_depth": (1, 50),
            "model__min_samples_split": (2, 20),
            "model__min_samples_leaf": (1, 10),
            "model__max_features": ["sqrt"],
            "model__bootstrap": [True, False],
            "model__class_weight": [
                "balanced",
                "balanced_subsample",
                None,
            ],
        }
    },
    "LightGBM": {
        "default": {
            "model__n_estimators": (50, 300),
            "model__max_depth": (3, 6),
            "model__learning_rate": Real(0.01, 0.2, prior="log-uniform"),
            "model__scale_pos_weight": (1, 10),
            "model__num_leaves": (20, 150),
            "model__min_child_samples": (20, 150),
            "model__min_gain_to_split": (0.001, 0.1),
            "model__feature_fraction": (0.5, 1.0),
            "model__bagging_fraction": (0.5, 1.0),
            "model__bagging_freq": (1, 10),
            "model__lambda_l1": (0.0, 10.0),
            "model__lambda_l2": (1.0, 10.0),
            "model__min_split_gain": (0.0, 0.1),
            "model__max_bin": (100, 500),
            "model__boosting_type": ["gbdt"],
            "model__verbose": [-1],
        }
    },

    "BoostDT": {
        "default": {
            "model__n_estimators": Integer(50, 100),
            "model__learning_rate": Real(0.01, 0.1, prior="log-uniform"),
            "model__max_depth": Integer(3, 7),
            "model__min_samples_split": Integer(2, 5),
            "model__min_samples_leaf": Integer(1, 4),
            "model__max_features": Categorical(["log2"]),
        }
    },
    "BoostSGB": {
        "default": {
            "model__n_estimators": Integer(50, 100),
            "model__learning_rate": Real(0.01, 0.1, prior="log-uniform"),
            "model__max_depth": Integer(3, 7),
            "model__min_samples_split": Integer(2, 5),
            "model__min_samples_leaf": Integer(1, 4),
            "model__subsample": Real(0.7, 0.9),
            "model__max_features": Categorical(["log2"]),
        }
    },
    "BagDT": {
        "default": {
            "model__n_estimators": Integer(low=5, high=10),
            "model__max_samples": Real(low=0.7, high=1.0),
            "model__estimator__max_depth": Integer(low=3, high=5),
            "model__estimator__min_samples_split": Integer(low=2, high=5),
        }
    },
    "BagNN": {
        "default": {
            "model__n_estimators": Integer(low=5, high=10),
            "model__max_samples": Real(low=0.7, high=1.0),
            "model__estimator__hidden_layer_sizes": Integer(low=5, high=10),
            "model__estimator__activation": Categorical(["relu"]),
            "model__estimator__alpha": Real(
                low=1e-5, high=1e-2, prior="log-uniform"
            ),
        }
    },
    "AdaBoost-DT": {
        "default": {
            "model__n_estimators": Integer(low=5, high=10),
            "model__learning_rate": Real(low=0.01, high=0.1, prior="log-uniform"),
            "model__estimator__max_depth": Integer(low=2, high=3),
            "model__estimator__min_samples_split": Integer(low=2, high=3),
        }
    },
    "AdaBoost-NN": {
        "default": {
            "model__n_estimators": Integer(low=5, high=10),
            "model__learning_rate": Real(low=0.01, high=0.1, prior="log-uniform"),
            "model__estimator__hidden_layer_sizes": Integer(low=5, high=10),
            "model__estimator__activation": Categorical(["relu"]),
            "model__estimator__alpha": Real(
                low=1e-5, high=1e-2, prior="log-uniform"
            ),
        }
    },
    "SVM-Rbf": {
        "default": {
            "model__C": Real(0.1, 10, prior="log-uniform"),
            "model__gamma": Categorical(["scale"]),
        }
    },
    "ANN": {
        "default": {
            "model__hidden_layer_sizes": Categorical([(50,), (100,), (100, 50)]),
            "model__activation": Categorical(["relu"]),
            "model__alpha": Real(1e-5, 1e-2, prior="log-uniform"),
        }
    },
}