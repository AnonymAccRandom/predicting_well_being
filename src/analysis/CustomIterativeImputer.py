from typing import Optional, Union

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone, BaseEstimator
from sklearn.impute import IterativeImputer


class CustomIterativeImputer(IterativeImputer):
    """
    Custom iterative imputer extending sklearn's `IterativeImputer` to include additional functionality
    for handling categorical features and predictive mean matching (PMM).

    This class inherits all functionality from `IterativeImputer` and adds support for:
    - Categorical feature imputation using specific indices (`categorical_idx`).
    - Predictive mean matching (PMM) for continuous variables with a specified number of neighbors (`pmm_k`).

    For attributes and parameters not explicitly documented here, refer to the `IterativeImputer` class.

    Attributes:
        categorical_idx (Optional[list[int]]): Indices of categorical features to be handled differently
            during imputation (e.g., binary thresholding).
        pmm_k (int): Number of nearest neighbors to use for predictive mean matching (PMM).
    """

    def __init__(
        self,
        estimator: Optional[BaseEstimator] = None,
        missing_values: Union[int, float, str, None] = np.nan,
        sample_posterior: bool = False,
        max_iter: int = 10,
        tol: float = 1e-3,
        n_nearest_features: Optional[int] = None,
        initial_strategy: str = "mean",
        imputation_order: str = "ascending",
        skip_complete: bool = False,
        min_value: float = -np.inf,
        max_value: float = np.inf,
        verbose: int = 0,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
        add_indicator: bool = False,
        keep_empty_features: bool = False,
        categorical_idx: Optional[list[int]] = None,
        pmm_k: int = 5,
    ) -> None:
        """
        Initializes the CustomIterativeImputer with both inherited and custom parameters.

        This subclass retains all parameters of `IterativeImputer` while introducing additional parameters
        (`categorical_idx` and `pmm_k`) to enhance functionality for specific imputation scenarios.

        Args:
            estimator: The estimator to use for regression or classification during imputation. If None,
                       a default BayesianRidge model is used. Inherited from `IterativeImputer`.
            missing_values: Placeholder for missing values. Defaults to `np.nan`. Inherited from `IterativeImputer`.
            sample_posterior: If True, samples posterior values during imputation. Defaults to False.
                              Inherited from `IterativeImputer`.
                              **Note**: This does not work with non-Bayesian models currently used but can
                              be used with BayesianRidge.
            max_iter: Maximum number of imputation iterations. Defaults to 10. Inherited from `IterativeImputer`.
            tol: Convergence tolerance for stopping criteria. Defaults to 1e-3. Inherited from `IterativeImputer`.
            n_nearest_features: Number of nearest features to use for imputation. Defaults to None (all features).
                                Inherited from `IterativeImputer`.
            initial_strategy: Strategy to initialize missing values (e.g., "mean", "median").
                              Inherited from `IterativeImputer`.
            imputation_order: Order in which features are imputed (e.g., "ascending", "random").
                              Inherited from `IterativeImputer`.
            skip_complete: Whether to skip features without missing values during imputation. Defaults to False.
                           Inherited from `IterativeImputer`.
            min_value: Minimum allowable value for imputed values. Defaults to `-np.inf`.
                       Inherited from `IterativeImputer`.
            max_value: Maximum allowable value for imputed values. Defaults to `np.inf`.
                       Inherited from `IterativeImputer`.
            verbose: Controls verbosity of output. Defaults to 0. Inherited from `IterativeImputer`.
            random_state: Random state for reproducibility. Defaults to None. Inherited from `IterativeImputer`.
            add_indicator: Whether to add a missingness indicator for features. Defaults to False.
                           Inherited from `IterativeImputer`.
            keep_empty_features: Whether to keep features with no observed values during imputation.
                                 Defaults to False. Inherited from `IterativeImputer`.
            categorical_idx: Indices of categorical features. Defaults to None. Used for imputation of binary
                             or categorical features using specialized handling.
            pmm_k: Number of nearest neighbors to use for predictive mean matching (PMM). Defaults to 5.
                   This parameter determines the degree of randomization in PMM-based imputation.
        """
        super().__init__(
            estimator=estimator,
            missing_values=missing_values,
            sample_posterior=sample_posterior,
            max_iter=max_iter,
            tol=tol,
            n_nearest_features=n_nearest_features,
            initial_strategy=initial_strategy,
            imputation_order=imputation_order,
            skip_complete=skip_complete,
            min_value=min_value,
            max_value=max_value,
            verbose=verbose,
            random_state=random_state,
            add_indicator=add_indicator,
            keep_empty_features=keep_empty_features,
        )
        self.categorical_idx = categorical_idx
        self.pmm_k = pmm_k

    def _impute_one_feature(
        self,
        X_filled: Union[np.ndarray, "pd.DataFrame"],
        mask_missing_values: np.ndarray,
        feat_idx: int,
        neighbor_feat_idx: np.ndarray,
        estimator: Optional[BaseEstimator] = None,
        fit_mode: bool = True,
        params: Optional[dict] = None,
    ) -> tuple[Union[np.ndarray, "pd.DataFrame"], BaseEstimator]:
        """
        Imputes missing values for a single feature using a specified estimator.

        This method predicts missing values for a target feature (`feat_idx`) using other features
        (`neighbor_feat_idx`) as predictors. Depending on the feature type and configuration, the
        following approaches are used:
        - Continuous features: Uses predictive mean matching (PMM) or direct predictions.
        - Binary features: Imputes probabilities using `predict_proba`.
        - Sample posterior: Samples imputed values from a truncated normal distribution.

        With this method, we overwrite the _impute_one_feature method of the parent class
        (sklearns Iterative Imputer). Most functionality are comparable, with the slight
        differences described above.

        **Implementation**:
        1. **Estimator Initialization**:
           - If no estimator is provided, a new one is cloned from `self._estimator`.
           - In `fit_mode=False`, the provided estimator must already be fitted.

        2. **Data Preparation**:
           - The rows with observed values (`~missing_row_mask`) are used to fit the model.
           - The rows with missing values (`missing_row_mask`) are used for prediction.

        3. **Fit Mode**:
           - If `fit_mode=True`, the estimator is fitted using the observed rows of `neighbor_feat_idx` as predictors
             and `feat_idx` as the target.

        4. **Prediction**:
           - If `sample_posterior=True`, posterior sampling is applied using the mean (`mus`) and standard deviation
             (`sigmas`) of the estimator's predictions. In case of categorical features, values are thresholded to ensure
             binary outcomes.
           - Otherwise, PMM is applied:
             - Observed predictions (`y_pred_obs`) are compared to missing predictions (`y_pred_mis`) to find the
               closest neighbors.
             - One of the nearest observed values is randomly selected for each missing value.

        5. **Feature Update**:
           - The imputed values are assigned to the missing entries of `feat_idx` in `X_filled`.

        Args:
            X_filled: The input data with the latest imputations.
            mask_missing_values: A boolean mask indicating the missing values in the input data.
            feat_idx: The index of the feature being imputed.
            neighbor_feat_idx: Indices of the neighboring features used as predictors.
            estimator: The model used to impute the missing values. If None, it is cloned from `self._estimator`.
            fit_mode: Whether to fit the estimator or use it directly for predictions. Default is True.
            params: Additional parameters to pass to the estimator's `fit` method.

        Returns:
            tuple:
                - X_filled: The updated input data with imputed values for the missing entries in the current feature.
                - estimator: The fitted estimator used for imputing the feature.

        Raises:
            ValueError: If `fit_mode=False` and no pre-fitted estimator is provided.
        """
        if estimator is None and fit_mode is False:
            raise ValueError(
                "If fit_mode is False, then an already-fitted "
                "estimator should be passed in."
            )

        if estimator is None:
            estimator = clone(self._estimator)

        missing_row_mask = mask_missing_values[:, feat_idx]

        if fit_mode:
            X_train = safe_indexing(
                safe_indexing(X_filled, neighbor_feat_idx, axis=1),
                ~missing_row_mask,
                axis=0,
            )
            y_train = safe_indexing(
                safe_indexing(X_filled, feat_idx, axis=1),
                ~missing_row_mask,
                axis=0,
            )

            if hasattr(estimator, "set_params"):
                estimator.set_params(
                    feat_idx=feat_idx,
                    neighbor_feat_idx=neighbor_feat_idx,
                )
            if params is None:
                estimator.fit(X_train, y_train)
            else:
                estimator.fit(X_train, y_train, **params)

        else:
            X_train = safe_indexing(
                safe_indexing(X_filled, neighbor_feat_idx, axis=1),
                ~missing_row_mask,
                axis=0,
            )

            y_train = safe_indexing(
                safe_indexing(X_filled, feat_idx, axis=1),
                ~missing_row_mask,
                axis=0,
            )

        if np.sum(missing_row_mask) == 0:
            return X_filled, estimator

        X_test = safe_indexing(
            safe_indexing(X_filled, neighbor_feat_idx, axis=1),
            missing_row_mask,
            axis=0,
        )
        if self.sample_posterior:
            mus, sigmas = estimator.predict(X_test, return_std=True)
            imputed_values = np.zeros(mus.shape, dtype=X_filled.dtype)
            positive_sigmas = sigmas > 0
            imputed_values[~positive_sigmas] = mus[~positive_sigmas]
            mus_too_low = mus < self._min_value[feat_idx]
            imputed_values[mus_too_low] = self._min_value[feat_idx]
            mus_too_high = mus > self._max_value[feat_idx]
            imputed_values[mus_too_high] = self._max_value[feat_idx]

            # Sample from truncated normal distribution
            inrange_mask = positive_sigmas & ~mus_too_low & ~mus_too_high
            mus = mus[inrange_mask]
            sigmas = sigmas[inrange_mask]
            a = (self._min_value[feat_idx] - mus) / sigmas
            b = (self._max_value[feat_idx] - mus) / sigmas
            truncated_normal = stats.truncnorm(a=a, b=b, loc=mus, scale=sigmas)
            imputed_values[inrange_mask] = truncated_normal.rvs(
                random_state=self.random_state_
            )

            if feat_idx in self.categorical_idx:
                imputed_values = (imputed_values >= 0.5).astype(int)

        else:
            is_binary = np.array_equal(np.unique(y_train), [0, 1])

            if is_binary:
                # For binary variables, predict_proba gives probabilities
                imputed_probs = estimator.predict_proba(X_test)[:, 1]
                y_pred_obs = estimator.predict_proba(X_train)[:, 1]
                y_pred_mis = imputed_probs
            else:
                # For continuous variables, use regular predict
                imputed_values = estimator.predict(X_test, return_std=False)
                y_pred_obs = estimator.predict(X_train, return_std=False)
                y_pred_mis = imputed_values

            imputed_values_pmm = np.empty_like(y_pred_mis)

            # Use the class-level random_state for reproducibility
            rng = np.random.RandomState(self.random_state)

            for i, y_pred in enumerate(y_pred_mis):
                distances = np.abs(y_pred_obs - y_pred)
                nn_indices = np.argsort(distances)[: self.pmm_k]
                imputed_value = rng.choice(y_train[nn_indices])

                if is_binary:
                    imputed_value = (imputed_value >= 0.5).astype(int)

                imputed_values_pmm[i] = imputed_value
            imputed_values = imputed_values_pmm

        safe_assign(
            X_filled,
            imputed_values,
            row_indexer=missing_row_mask,
            column_indexer=feat_idx,
        )
        return X_filled, estimator


def safe_indexing(
    X: Union[np.ndarray, pd.DataFrame, pd.Series, list],
    indices: Union[np.ndarray, list, slice],
    axis: int = 0,
) -> Union[np.ndarray, pd.DataFrame, pd.Series, list]:
    """
    Safely index an array-like object along the specified axis.

    This function supports indexing for:
    - Pandas DataFrame or Series: Uses `.iloc` for safe indexing.
    - NumPy arrays (1D or 2D): Handles row or column indexing.
    - Python lists: Supports only row indexing (axis=0).

    Args:
        X: The input object to index. Can be a NumPy array, Pandas DataFrame, Series, or Python list.
        indices: The indices to use for selection. Can be a list, NumPy array, or slice.
        axis: The axis along which to index.
            - `0`: Rows.
            - `1`: Columns (only for 2D arrays or DataFrames).

    Returns:
        The indexed subset of `X`, with the same type as the input object.

    Raises:
        ValueError: If the axis is invalid for the given input type or object.
        IndexError: If attempting to index along an invalid axis for 1D NumPy arrays.
        ValueError: If the input object type is unsupported.
    """
    if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
        if axis == 0:
            return X.iloc[indices]
        elif axis == 1:
            return X.iloc[:, indices]
        else:
            raise ValueError(f"Invalid axis {axis} for pandas object.")

    elif isinstance(X, np.ndarray):
        X = np.asarray(X)

        if X.ndim == 1:
            if axis != 0:
                raise IndexError("Axis out of bounds for 1D array.")
            return X[indices]

        elif X.ndim == 2:
            if axis == 0:
                return X[indices, :]
            elif axis == 1:
                return X[:, indices]
            else:
                raise ValueError(f"Invalid axis {axis} for 2D array.")

    elif isinstance(X, list):
        if axis != 0:
            raise ValueError("safe_indexing only supports axis=0 for lists.")
        return [X[i] for i in indices]

    else:
        raise ValueError(f"Unsupported data type: {type(X)}")


def safe_assign(
    X: Union[np.ndarray, pd.DataFrame, pd.Series],
    values: Union[np.ndarray, pd.Series, list],
    row_indexer: Optional[Union[np.ndarray, list]] = None,
    column_indexer: Optional[Union[np.ndarray, list, int]] = None,
) -> None:
    """
    Safely assign values to specific indices of an array-like object.

    This function modifies the input object `X` by assigning the provided `values`
    to the specified rows and/or columns. It supports:
    - Pandas DataFrame or Series: Uses `.iloc` for safe assignment.
    - NumPy arrays (2D or higher): Handles assignment along rows and/or columns.

    Args:
        X: The input object to modify. Can be a NumPy array, Pandas DataFrame, or Series.
        values: The values to assign. The shape of `values` must match the shape of the
                specified indices in `X`.
        row_indexer: Indices of rows to update. Can be a list, NumPy array, or None.
        column_indexer: Indices of columns to update. Can be a list, NumPy array, single integer, or None.

    Raises:
        ValueError: If neither `row_indexer` nor `column_indexer` is provided.
        ValueError: If the input object type is unsupported.
    """
    if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
        if row_indexer is not None and column_indexer is not None:
            X.iloc[row_indexer, column_indexer] = values
        elif row_indexer is not None:
            X.iloc[row_indexer] = values
        elif column_indexer is not None:
            X.iloc[:, column_indexer] = values
        else:
            raise ValueError("Either row_indexer or column_indexer must be provided.")

    elif isinstance(X, np.ndarray):
        if row_indexer is not None and column_indexer is not None:
            if np.isscalar(column_indexer):
                values = np.asarray(values).reshape(-1, 1)
            X[np.ix_(row_indexer, [column_indexer])] = values

        elif row_indexer is not None:
            X[row_indexer, :] = values

        elif column_indexer is not None:
            X[:, column_indexer] = values

        else:
            raise ValueError("Either row_indexer or column_indexer must be provided.")

    else:
        raise ValueError("Unsupported data type for safe_assign")
