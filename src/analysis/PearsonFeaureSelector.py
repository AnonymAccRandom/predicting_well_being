from typing import Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class PearsonFeatureSelector(BaseEstimator, TransformerMixin):
    """
    Feature selector based on Pearson correlation.

    This class selects features based on their Pearson correlation with the target variable.
    It allows for:
    - Selecting a fixed number of top features based on absolute correlation.
    - Selecting features whose correlation exceeds a specified threshold.

    NOTE: Feature selection is not included in the final version of this paper.

    Attributes:
        num_features (Optional[int]): Number of top features to select based on absolute correlation.
        correlation_threshold (Optional[float]): Minimum absolute correlation to consider a feature informative.
        target_prefix (str): Prefix used to identify features for selection based on correlation.
        selected_features_ (Optional[list[str]]): List of selected feature names after fitting.
    """

    def __init__(
        self,
        num_features: Optional[int] = None,
        correlation_threshold: Optional[float] = None,
        target_prefix: str = "sens_",
    ) -> None:
        """
        Initialize the PearsonFeatureSelector.

        Args:
            num_features: The number of top features to select based on absolute correlation.
            correlation_threshold: The minimum absolute correlation to consider a feature informative.
            target_prefix: The prefix used to identify features for selection (default: "sens_").

        Raises:
            ValueError: If both `num_features` and `correlation_threshold` are None.
        """
        if num_features is None and correlation_threshold is None:
            raise ValueError(
                "Either num_features or correlation_threshold must be specified."
            )

        self.num_features = num_features
        self.correlation_threshold = correlation_threshold
        self.target_prefix = target_prefix
        self.selected_features_ = None

    def fit(
        self, X: pd.DataFrame, y: Union[pd.Series, pd.DataFrame]
    ) -> "PearsonFeatureSelector":
        """
        Fit the selector by computing Pearson correlations for features and selecting the top ones.

        Args:
            X: DataFrame of input features.
            y: Target variable as a Series or DataFrame.

        Returns:
            PearsonFeatureSelector: The fitted selector with identified features.

        Raises:
            ValueError: If no features matching the `target_prefix` are found in `X`.
        """
        target_features = [
            col for col in X.columns if col.startswith(self.target_prefix)
        ]

        if not target_features:
            raise ValueError(
                f"No features found with the prefix '{self.target_prefix}'."
            )

        correlations = X[target_features].apply(lambda col: col.corr(y))
        abs_correlations = correlations.abs()

        if self.num_features is not None:
            selected_features = abs_correlations.nlargest(
                self.num_features
            ).index.tolist()

        elif self.correlation_threshold is not None:
            selected_features = abs_correlations[
                abs_correlations >= self.correlation_threshold
            ].index.tolist()

        self.selected_features_ = selected_features + [
            col for col in X.columns if not col.startswith(self.target_prefix)
        ]

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the dataset by selecting the previously identified features.

        Args:
            X: DataFrame of input features to transform.

        Returns:
            pd.DataFrame: Transformed DataFrame with only the selected features.

        Raises:
            RuntimeError: If the selector has not been fitted.
        """
        if self.selected_features_ is None:
            raise RuntimeError("You must fit the selector before transforming data.")

        return X[self.selected_features_]

    def get_support(self, indices: bool = False) -> Union[np.ndarray, list[int]]:
        """
        Get the mask or indices of selected features.

        Args:
            indices: If True, return the indices of selected features.
                     If False, return a boolean mask of the selected features.

        Returns:
            np.ndarray or list[int]: Indices or boolean mask of the selected features.

        Raises:
            RuntimeError: If the selector has not been fitted.
        """
        if self.selected_features_ is None:
            raise RuntimeError("You must fit the selector before getting support.")

        feature_indices = [i for i, col in enumerate(self.selected_features_)]

        if indices:
            return feature_indices

        else:
            mask = [False] * len(self.selected_features_)
            for idx in feature_indices:
                mask[idx] = True

            return mask
