import logging
import pandas as pd
import numpy as np
import warnings

from collections import OrderedDict
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import normalize
from sklearn.impute import SimpleImputer, KNNImputer
from pandas.api.types import is_number, is_categorical_dtype

warnings.filterwarnings("ignore")

__all__ = []
_logger = logging.getLogger(__name__)


class NumericTransformer(BaseEstimator, TransformerMixin):
    """Impute numeric variables and replace outliers based on given quantiles.
    Currently, a limitation is that this class is able to impute only
    (mean, median or mode) values and not specific values e.g -1 or 0.
    :param list numeric_var:
        List of numeric features that should be imputed.
    :param Union[int,str] missing_values:
        The placeholder for the missing values. All occurrences of
        missing_values will be imputed.
        To specify entries encoded as np.nan as missing values, please use
        string value "NaN". Default "NaN".
    :param str strategy:
        Imputation strategy, one of "mean", "median" or
        "most_frequent"(=mode). Default "mean".
    :param bool replace_outliers:
        To replace outliers or not.
    :param float outliers_lower_quantile:
        The lower quantile below which the values will be considered outliers
    :param float outliers_upper_quantile:
        The upper quantile above which the values will be considered outliers
    :param bool verbose:
        Flag signalizing if the algorithm is verbose or not. Default false.
    :ivar sklearn.preprocessing.SimpleImputer imputer:
        Instance of SimpleImputer that is used to impute missing values in the data.
    :ivar pandas.Series lower_quantile:
        Lower quantiles for each column. Values in a column smaller than
         the corresponding quantile will be considered an outlier
    :ivar pandas.Series upper_quantile:
        Upper quantiles for each column. Values in a column greater than
        the corresponding quantile will be considered an outlier
    :ivar pandas.Series min_values:
        Smallest value in each column that is greater than the respective
         lower quantile. Outliers will be mapped to this value.
    :ivar pandas.Series max_values:
        Greatest value in each column that is smaller than the respective
        upper quantile. Outliers will be mapped to this value.
    """

    def __init__(
        self,
        numeric_var=None,
        missing_values=np.nan,
        strategy="median",
        method_nans="knn",
        replace_outliers=False,
        outliers_lower_quantile=0.05,
        outliers_upper_quantile=0.95,
        verbose=False,
    ):

        if strategy not in {"mean", "median", "most_frequent"}:
            raise ValueError(
                "Invalid imputation strategy! Please use 'mean', 'median' or 'most_frequent'"
            )

        if method_nans not in {"knn", "simple"}:
            raise ValueError(
                "Invalid imputation strategy! Please use 'knn', 'simple' or 'XXXXXXX'"
            )

        if isinstance(missing_values, str) and missing_values != "NaN":
            raise ValueError(
                "Error! For specifing missing values please use either an integer or"
                " string 'NaN' for numpy.nan enconded values"
            )

        if any(
            x < 0 or x > 1 for x in [outliers_lower_quantile, outliers_upper_quantile]
        ):
            raise ValueError(
                "outliers lower and upper quantile must be be between 0 and 1"
            )

        if outliers_lower_quantile > outliers_upper_quantile:
            raise ValueError(
                "outliers_lower_quantile value is greater than outliers_upper_quantile"
            )

        self.numeric_var = numeric_var
        self.strategy = strategy
        self.method_nans = method_nans
        self.missing_values = missing_values

        if self.method_nans == "simple":
            self.imputer = SimpleImputer(
                missing_values=self.missing_values, strategy=self.strategy
            )

        if self.method_nans == "knn":
            self.imputer = KNNImputer(
                missing_values=self.missing_values, n_neighbors=5, weights="uniform"
            )

        self.replace_outliers = replace_outliers
        self.outliers_lower_quantile = outliers_lower_quantile
        self.outliers_upper_quantile = outliers_upper_quantile
        self.verbose = verbose

    def fit(self, data, *args):
        """Fit imputer and computes quantiles values using the training data.
        :param pandas.DataFrame data:
            DataFrame (shape: ``n_samples, n_features``) used to perform fitting/learning for the imputer object.
        :return:
            NumericTransformer object, fitted on the data frame 'data'.
        :rtype NumericTransformer:
        :raises ValueError:
            In case an empty data frame is passed to the function or when some
            of the specified columns are not actually present in the data frame.
        """
        if self.numeric_var is None:
            self.numeric_var = [
                col for col in data if not is_categorical_dtype(data.dtypes[col])
            ]

        if self.replace_outliers:
            if self.verbose:
                print("Outliers replacement fit start;")
            self._fit_outliers(data)
            if self.verbose:
                print("Outliers replacement fit done;")

        if self.verbose:
            print("Impute fit start;")
        self.imputer.fit(data.loc[:, self.numeric_var])
        if self.verbose:
            print("Impute fit done;")

        return self

    def transform(self, data, *args):
        """Perform imputation of numeric variables and replaces outliers.
        :param pandas.DataFrame data:
            DataFrame (shape: ``n_samples, n_features``) which needs to be imputed.
        :return:
            Transformed data frame with aggregated and dummified levels of
            categorical variables.
        :rtype: pandas.DataFrame
        :raises ValueError:
            In case a non-valid data frame (empty, None or not pandas) is passed
            to the function or when some of the specified columns are not
            actually present in the data frame and when to_keep_dictionary
            contains non consistent columns w.r.t data frame and specified
            numeric variables.
        """

        if self.replace_outliers:
            if self.verbose:
                print("Outliers replacement transform start;")
            data = self._transform_outliers(data)
            if self.verbose:
                print("Outliers replacement transform done;")

        imputed = self.imputer.transform(data.loc[:, self.numeric_var])
        data = data.assign(
            **{key: val for key, val in zip(self.numeric_var, imputed.T)}
        )
        if self.verbose:
            print("Impute transform done;")

        return data

    def _fit_outliers(self, df):
        """Calculate quantiles and find the minimum and maximum values that are not considered outliers.
        :param pandas.DataFrame df:
            shape: ``n_samples, n_features``.
        """
        numeric_df = df.apply(pd.to_numeric, errors="coerce")

        numeric_df = numeric_df.select_dtypes(include=[np.number])

        if numeric_df.empty:
            raise ValueError("No numeric columns found or convertible in the DataFrame")

        # Compute quantiles for outliers thresholds
        quantiles = numeric_df.quantile(
            [self.outliers_lower_quantile, self.outliers_upper_quantile]
        )

        self._lower_quantile = quantiles.loc[self.outliers_lower_quantile]
        if not isinstance(self._lower_quantile, pd.Series):
            self._lower_quantile = pd.Series(
                self._lower_quantile, index=numeric_df.columns
            )
        self._upper_quantile = quantiles.loc[self.outliers_upper_quantile]
        if not isinstance(self._upper_quantile, pd.Series):
            self._upper_quantile = pd.Series(
                self._upper_quantile, index=numeric_df.columns
            )

        self._min_values = numeric_df[numeric_df >= self._lower_quantile].min()
        self._max_values = numeric_df[numeric_df <= self._upper_quantile].max()

        return self

    def _transform_outliers(self, df):
        """Replace the values considered outliers with the min/max values computed during fitting.
        :param df: pandas.DataFrame, shape = (n_samples, n_features)
        :return:
            Data with outliers replaced.
        :rtype: pandas.DataFrame
        """
        for col in df.select_dtypes(include=[np.number]):
            s = df[col].copy()

            s[s < self._lower_quantile[col]] = self._min_values[col]
            s[s > self._upper_quantile[col]] = self._max_values[col]

            df = df.assign(**{col: s})

        return df


class CategoricalTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        categorical_var=None,
        frequency=0.05,
        max_levels=50,
        aggregation_label="Others",
        missing_label="Missing",
        target_type=None,
        aggregation_strategy="frequency",
        levels_pruning_rescale=False,
        encoding_strategy="dummify",
        verbose=False,
    ):
        if aggregation_strategy not in {"frequency", "infogain", "prob-diff", "none"}:
            raise ValueError(
                "Invalid levels_pruning_strategy! Please use 'none', 'frequency', 'infogain', or 'prob-diff'"
            )

        self.categorical_var = categorical_var
        self.frequency = frequency
        self.max_levels = max_levels
        self.aggregation_strategy = aggregation_strategy
        self.target_type = target_type
        self.levels_pruning_rescale = levels_pruning_rescale
        self.aggregation_label = aggregation_label
        self.missing_label = missing_label
        self.encoding_strategy = encoding_strategy
        self.verbose = verbose

        self.categorical_var_ = None
        self.categorical_dtype_columns_ = None
        self.to_keep_dict_ = None
        self.to_drop_ = None

    def fit(self, data, y=None):
        self.categorical_var_ = self._get_categorical_var(data)

        # make sure first tests are run
        self.categorical_dtype_columns_ = {
            col for col in self.categorical_var_ if is_categorical_dtype(data[col])
        }

        _logger.info(
            "Aggregating levels with less than %s %% of occurrence to a maximum of %s levels",
            self.frequency * 100,
            self.max_levels,
        )

        if self.aggregation_strategy not in {"none", "frequency"} and y is None:
            raise TypeError("Need to pass target when using non-frequency pruning.")

        self.to_keep_dict_ = OrderedDict()

        for cat in self.categorical_var_:
            s = optional_fillna(data[cat], self.missing_label)
            levels = self._determine_levels(cat, s, y, frequency=self.frequency)
            if levels:
                self.to_keep_dict_[cat] = levels

        self.to_drop_ = list(set(self.categorical_var_) - set(self.to_keep_dict_))

        _logger.info("Aggregate fit done;")
        return self

    def _get_categorical_var(self, data):
        if self.categorical_var is None:
            return [col for col in data if is_categorical_dtype(data[col])]

        elif isinstance(self.categorical_var, str):
            return [self.categorical_var]

        else:
            return [*self.categorical_var]

    def _determine_levels(self, cat, s, y, frequency):
        # Do not aggregate columns, just save all the levels in the dictionary
        if self.aggregation_strategy == "none":
            levels = s.unique()
            levels = list(levels)

            if self.aggregation_label is not None:
                levels = levels + [self.aggregation_label]

            return mixed_type_sorted(set(levels))

        elif (
            self.aggregation_strategy == "prob-diff"
            and self.target_type == "continuous"
        ):
            rank = _Rank.continuous(
                s, y, levels_pruning_rescale=self.levels_pruning_rescale
            )

        elif (
            self.aggregation_strategy == "prob-diff"
            and self.target_type == "categorical"
        ):
            rank = _Rank.categorical(
                s, y, levels_pruning_rescale=self.levels_pruning_rescale
            )

        elif (
            self.aggregation_strategy == "infogain" and self.target_type == "continuous"
        ):
            raise ValueError(
                "Invalid target type for infogain, only categorical is currently supported."
            )

        elif (
            self.aggregation_strategy == "infogain"
            and self.target_type == "categorical"
        ):
            rank = _Rank.categorical_infogain(s, y)

        elif self.aggregation_strategy == "frequency":
            distrib = s.value_counts(normalize=True, dropna=False, ascending=False)
            rank = [(0, level) for level in distrib[distrib > frequency].index]

        else:
            raise ValueError(
                "unknown aggregation strategy {}".format(self.aggregation_strategy)
            )

        # Remove the small levels, i.e keep only the top <max_levels>
        frq_levels = [col for score, col in rank[: self.max_levels]]

        frq_levels = frq_levels + [self.aggregation_label]
        frq_levels = set(frq_levels)

        # if not constant or near-constant feature
        # NOTE: missing can have lower frequency ...
        if len(frq_levels) <= 1:
            return []

        return mixed_type_sorted(frq_levels)

    def transform(self, data):
        encoding_strategy = _ensure_encoding_strategy(self.encoding_strategy)

        to_assign = {}
        to_drop = []

        for col in self.to_keep_dict_:
            s = data[col]

            if col in self.categorical_dtype_columns_:
                s = s.astype("category")

            s = optional_fillna(s, self.missing_label)
            s = fix_categories(s, self.to_keep_dict_[col], self.aggregation_label)

            col_to_assign, col_to_drop = encoding_strategy(
                col,
                s,
                self.to_keep_dict_[col],
                aggregation_label=self.aggregation_label,
            )
            to_assign.update(col_to_assign)
            to_drop.extend(col_to_drop)

        return (
            data.drop(columns=self.to_drop_).drop(columns=to_drop).assign(**to_assign)
        )


def optional_fillna(s, missing_label):
    if missing_label is None:
        return s

    if not is_categorical_dtype(s):
        return s.fillna(missing_label)

    if missing_label not in s.cat.categories:
        s = s.copy()
        s.cat.add_categories([missing_label], inplace=True)

    return s.fillna(missing_label)


def fix_categories(s, levels, aggregation_label):
    if is_categorical_dtype(s):
        to_add = [cat for cat in levels if cat not in s.cat.categories]

    else:
        to_add = []

    s = s.copy()

    if is_categorical_dtype(s) and to_add:
        s.cat.add_categories(to_add, inplace=True)

    if pd.isnull(s).any():
        raise ValueError("cannot handle nulls, please use fillna")

    if is_categorical_dtype(s):
        # use compare levels via codes and use fast integer arithmetic
        # NOTE: using numpy functions is only correct if there are no NANs
        # Workaround by Jesus Sono
        # agg_code = s.cat.categories.get_loc(aggregation_label)
        # codes_to_remove, = np.nonzero(~s.cat.categories.isin(levels))
        # s.cat.codes[np.in1d(s.cat.codes, codes_to_remove)] = agg_code

        to_replace = {
            label: aggregation_label
            for label in s.cat.categories
            if label not in levels
        }
        s.replace(to_replace, inplace=True)

    else:
        s[~s.isin(levels)] = aggregation_label

    if is_categorical_dtype(s):
        s.cat.reorder_categories(
            [v for v in levels if not pd.isnull(v)], inplace=True, ordered=False
        )
    return s


def mixed_type_sorted(levels):
    # NOTE: nan does not sort well, add pd.isnull
    # NOTE: sort numbers together to avoid subtle bugs
    return sorted(
        levels,
        key=lambda v: (
            (
                "_number"
                if is_number(v)
                else (type(v).__module__ + "." + type(v).__name__)
            ),
            pd.isnull(v),
            v,
        ),
    )


def _ensure_encoding_strategy(encoding_strategy):
    if callable(encoding_strategy):
        return encoding_strategy

    elif encoding_strategy == "dummify":
        return _dummify

    elif encoding_strategy == "encode":
        return _label_encode

    elif encoding_strategy == "identity":
        return _identity_encode

    else:
        raise ValueError("cannot interpret {} as an encoding strategy")


def _identity_encode(col, s, levels, aggregation_label):
    return {col: s}, []


def _dummify(col, s, levels, aggregation_label):
    return {
        str(col)
        + "_"
        + str(level).replace(" ", "_"): pd.Series(s == level, index=s.index, dtype=int)
        for level in levels
    }, [col]


def _label_encode(col, s, levels, aggregation_label):
    lookup_vec = pd.Series(data=np.arange(len(levels)), index=levels)
    t = pd.Series(data=levels.index(aggregation_label), index=s.index, dtype=int)
    found = s.isin(levels)
    t[found] = lookup_vec.loc[s[found]].values
    return {col: t}, []


class _Rank:
    """Class for holding the rank functions and the helper functions."""

    @classmethod
    def _prepare(cls, category, y, levels_pruning_rescale=None):
        """
        Prepare the data used for three ranking functions.

        The preparation generates the variables as following:
            * ``cross_tabulation``: the cross tabulation DataFrame by combining
                ``category`` and ``y``
            * ``category_items``: the unique category items from
                ``cross_tabulation``
            * ``counts_in_level``: numpy array, ``cross_tabulation``'s content
            * ``num_in_level``: contains number of examples for each unique
                value in ``y``
            * ``level_prob``: normalized numpy array of ``num_in_level``
            * ``weight``: its value depends on ``levels_pruning_rescale``.

        :param array-like category:
            category items of all the examples
        :param array-like y:
            levels of all the examples
        :param None/bool levels_pruning_rescale:
            if None, don't compute the weight ; if True uses the ``level_prob``
            as ``weight``; if False, uses an array with same shape as
            ``level_prob``, filled with ones
        :return: prepared variables
        :rtype: list
        """
        # creates the cross tabulation DatFrame
        cross_tabulation = pd.crosstab(category, y)  # shape: m * n
        category_items = cross_tabulation.index  # length: m
        # converts the cross tabulation DatFrame to numpy array, prepare for the
        # vectorized computation
        counts_in_level = cross_tabulation.values  # shape: m * n
        # counts about the appearance of each category item in level
        num_in_level = counts_in_level.sum(axis=1)  # m
        result = [
            cross_tabulation,
            category_items,
            counts_in_level,
            num_in_level,
        ]
        # if the levels_pruning_rescale is None, the function
        # categorical_infogain is using this function, then we can directly
        # return the variables.
        if levels_pruning_rescale is None:
            return result
        # for the functions categorical and continuous, the num_in_level array
        # should have the shape: 1 * m
        num_in_level = num_in_level.reshape(1, -1)
        result[-1] = num_in_level
        # normalizes the num_in_level
        level_prob = normalize(num_in_level, norm="l1")  # shape 1 * m
        # decides which value should be used as weight
        weight = (
            level_prob
            if levels_pruning_rescale
            else np.ones(level_prob.shape, dtype=int)
        )  # shape 1 * m
        result.append(weight)
        return result

    @classmethod
    def _cal_score(cls, target_given_level, target_value_prob, weight):
        """
        Calculate the score used for ranking.

        Use the given parameters to calculate the scores according to the
        formula:
        weight * |target_given_level - target_value_prob| / target_value_prob

        :param array-like target_given_level:
            target in given level
        :param array-like target_value_prob:
            target value probability
        :param array_like weight:
            weight of each category item for calculating their scores
        :return: 1-D array contains scores for each category item
        :rtype: numpy.array
        """
        subtraction_result = np.zeros(target_given_level.shape)  # shape m * n
        # specifies which value in the array is invalid and skip it in
        # subtraction
        valid_array = target_given_level != 0  # shape m * n
        np.subtract(
            target_given_level,
            target_value_prob,
            out=subtraction_result,
            where=valid_array,
        )  # shape m * n
        subtraction_result = np.absolute(subtraction_result)
        scores = subtraction_result.dot((1 / target_value_prob).T).dot(weight)
        # the scores is an array with shape  m * m, only the diagonal is what we
        # want
        return scores.diagonal()

    @classmethod
    def _rank(cls, scores, corresponding_items):
        """
        Rank the corresponding_items.

        According to the order, packs the score and the corresponding item
        together as a tuple, then ranks the tuples by the score in desc order.
        In the tuple, the first element is the score and the second is the
        category item.

        :param array-like scores:
            score of teh each category item
        :param array-like corresponding_items:
            items in one category
        :return: sorted list of tuples in form of (score, item)
        :rtype: list of 2-element-tuples
        """
        # noinspection PyTypeChecker
        return sorted(zip(scores, corresponding_items), reverse=True)

    @classmethod
    def categorical(cls, category, y, levels_pruning_rescale=False):
        """
        Rank the category for the case that y is also a categorical array.

        :param array-like category:
            category item in each example
        :param array-like y:
            corresponding level of each example
        :param bool levels_pruning_rescale:
            if True uses the ``level_prob`` as ``weight``; if False, uses an
            array with same shape as ``level_prob``, filled with ones
        :return: sorted list contains the tuples in form of
            (score, category item) in desc order
        :rtype: list of 2-element-tuples
        """
        prepared_vars = cls._prepare(category, y, levels_pruning_rescale)
        _c, category_items, counts_in_level, _n, weight = prepared_vars

        target_value_prob = normalize(
            counts_in_level.sum(axis=0).reshape(1, -1), norm="l1"
        )  # shape 1 * n
        target_given_level = normalize(
            counts_in_level, norm="l1", axis=1
        )  # shape m * n

        scores = cls._cal_score(target_given_level, target_value_prob, weight)
        return cls._rank(scores, category_items)

    @classmethod
    def continuous(cls, category, y, levels_pruning_rescale=False):
        """
        Rank the category for the case that y is a continuous array.

        :param array-like category:
            category item in each example
        :param array-like y:
            corresponding level of each example
        :param bool levels_pruning_rescale:
            if True uses the ``level_prob`` as ``weight``; if False, uses an
            array with same shape as ``level_prob``, filled with ones
        :return: sorted list contains the tuples in form of
            (score, category item) in desc order
        :rtype: list of 2-element-tuples
        """
        prepared_vars = cls._prepare(category, y, levels_pruning_rescale)
        (
            cross_tabulation,
            category_items,
            counts_in_level,
            num_in_level,
            weight,
        ) = prepared_vars
        target = np.array(cross_tabulation.columns).reshape(-1, 1)  # n * 1
        target_value_prob = np.array([[y.mean()]])
        target_given_level = (
            (counts_in_level.dot(target))
            .dot(1 / num_in_level)
            .diagonal()
            .reshape(-1, 1)
        )  # shape m * 1
        scores = cls._cal_score(target_given_level, target_value_prob, weight)
        return cls._rank(scores, category_items)

    @classmethod
    def categorical_infogain(cls, category, y):
        """
        Rank the category according to the information gain.

        :param array-like category:
            array contains occurred category item in each example
        :param array-like y:
            array contains corresponding level of each example
        :return:
            a sorted list of 2-element-tuples, the first element in the tuple is
            the score and the second one is the name of the category item
        :rtype: list of 2-element-tuples
        """
        prepared_vars = cls._prepare(category, y)
        # create a cross tabulation DataFrame, which contains counts of each
        # category item in each level
        (
            cross_tabulation,
            category_items,
            counts_in_level,
            num_in_level,
        ) = prepared_vars
        # the index of the cross tabulation DataFrame is the category items
        # category_items = cross_tabulation.index
        # take out the content and convert it to numpy array
        # total counts of category items in each level
        counts_total = counts_in_level.sum(axis=0)  # shape: n
        # complement of the counts in level
        counts_out_level = counts_total - counts_in_level  # shape: m * n

        # sum per row in complement of the counts in level
        num_out_level = counts_out_level.sum(axis=1)  # shape: m
        # total appearance of all category item
        num_total = num_in_level.sum()  # number
        weight_in_level = num_in_level / num_total  # shape: m
        weight_out_level = num_out_level / num_total  # shape: m

        # the entropy for counts_total
        entropy_total = cal_entropy(counts_total.reshape(1, -1))[0]
        # the entropy for counts_in_level
        entropy_in_level = cal_entropy(counts_in_level)  # shape: m
        # the entropy for counts_out_level
        entropy_out_level = cal_entropy(counts_out_level)  # shape m

        # weighted avg entropy in level
        wae_in_level = np.diag(weight_in_level).dot(entropy_in_level)  # shape: m
        # weighted avg entropy out level
        wae_out_level = np.diag(weight_out_level).dot(entropy_out_level)  # shape: m
        # calculates the information gain according to the formula:
        # IG = entropy(parent) - [weights average] * entropy(children)
        info_gain = entropy_total - (wae_in_level + wae_out_level)  # shape: m
        # since the info_gain is a 1-D array with the same length of
        # category_items, use zip to generate an iterable obj and apply it to
        # sorted function.
        return cls._rank(info_gain, category_items)


def cal_entropy(np_array):
    """
    Calculate the entropy for each example in the given numpy array.

    That's the vectorized version using numpy array.

    :param numpy.array np_array:
        cross tabulation in numpy array with shape
        ``(m_category_items, n_labels)`` to calculate, n is bigger than 1.
    :return:
        1-D numpy array with shape ``(m_category_items,)``, which contains the
        entropy for each category item
    :rtype: numpy.array
    """
    # independently normalize each sample
    p_array = normalize(np_array, norm="l1", axis=1)  # shape: m * 1
    # array for storing the log2 result
    log_p_array = np.zeros(p_array.shape)  # shape: m * 1
    # boolean array, in which values of True indicate to calculate the log2 at
    # that position, values of False indicate to leave the value in the output
    # alone. In other words, we use this array to set the entropy of 0 in
    # p_array to 0.
    valid_array = p_array != 0  # shape: m * 1
    # calculate the logarithm and put the result in log_p_array
    np.log2(p_array, out=log_p_array, where=valid_array)
    # calculates entropy according to the formula:
    # ``S = -sum_i(p_i * log(p_i)``
    entropy = (-1 * p_array).dot(log_p_array.T)  # shape: m * m
    # since p is an array with shape ``(m_category_items, n_labels)``, the
    # result we get from last step is an array with shape
    # ``(m_category_items, m_category_items)`` and the entropy is the diagonal
    # of this result.
    return entropy.diagonal()
