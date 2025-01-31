from typing import Any, Union

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin


class AdaptiveImputerEstimator(BaseEstimator):
    """
    Adaptive estimator that selects between a regressor and a classifier based on the feature type.

    This estimator dynamically chooses the appropriate model (regressor or classifier) during fitting,
    depending on whether the target feature is categorical. The decision is guided by a list of categorical
    feature indices provided at initialization. It adheres to scikit-learns interface which may explain
    some unintuitive code snippets.

    Attributes:
        regressor (RegressorMixin): Regressor model to be used for continuous features.
        classifier (ClassifierMixin): Classifier model to be used for categorical features.
        categorical_idx (List[int]): List of indices representing categorical features.
        feat_idx (Optional[int]): Index of the feature to be imputed or predicted. Default is None.
        neighbor_feat_idx (Optional[List[int]]): List of indices for neighboring features to be used as predictors. Default is None.
        model (Optional[Union[RegressorMixin, ClassifierMixin]]): The selected model (regressor or classifier) based on the feature type.
    """

    def __init__(
        self,
        regressor: RegressorMixin,
        classifier: ClassifierMixin,
        categorical_idx: list[int],
        feat_idx: int = None,
        neighbor_feat_idx: list[int] = None,
        **params: Any
    ) -> None:
        """
        Initializes the AdaptiveImputerEstimator with a regressor, a classifier, and feature indices.

        Args:
            regressor: The regressor model to use for continuous features.
            classifier: The classifier model to use for categorical features.
            categorical_idx: List of indices for features considered categorical.
            feat_idx: The index of the feature to impute or predict.
            neighbor_feat_idx: List of indices for neighboring features to use as predictors.
            params: Additional parameters for the regressor or classifier.
        """
        self.regressor = regressor
        self.classifier = classifier
        self.categorical_idx = categorical_idx
        self.feat_idx = feat_idx
        self.neighbor_feat_idx = neighbor_feat_idx
        self.model = None

    def is_categorical(self, feat_idx: int) -> bool:
        """
        Determines if a feature is categorical.

        Args:
            X: The dataset containing features.
            feat_idx: The index of the feature to check.

        Returns:
            bool: True if the feature is categorical, False otherwise.
        """
        return feat_idx in self.categorical_idx

    def fit(self, X: np.ndarray, y: np.ndarray) -> "AdaptiveImputerEstimator":
        """
        Fits the selected model (regressor or classifier) to the provided data.

        This method determines whether the feature to be imputed is categorical or continuous
        and selects the appropriate model accordingly.

        Args:
            X: The predictor variables.
            y: The target variable for training.

        Returns:
            AdaptiveImputerEstimator: The fitted estimator.
        """
        X = np.array(X)

        if self.is_categorical(self.feat_idx):
            self.model = self.classifier

        else:
            self.model = self.regressor

        self.model.fit(X, y)
        return self

    def predict(
        self, X: np.ndarray, return_std: bool = False
    ) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        """
        Makes predictions using the fitted model.

        If the model supports uncertainty estimation, this method can optionally return standard deviations
        along with the predictions.

        Args:
            X: The predictor variables for which predictions are made.
            return_std: Whether to return standard deviations along with predictions.

        Returns:
            Union[np.ndarray, tuple[np.ndarray, np.ndarray]]: Predictions or (predictions, standard deviations).
        """
        if return_std:
            return self.model.predict(X, return_std=return_std)
        else:
            return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts class probabilities using the fitted classifier.

        This method is only available when the model is a classifier.

        Args:
            X: The predictor variables for which probabilities are predicted.

        Returns:
            np.ndarray: Predicted class probabilities.
        """
        return self.model.predict_proba(X)
