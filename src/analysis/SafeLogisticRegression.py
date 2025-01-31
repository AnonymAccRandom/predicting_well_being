import numpy as np
from sklearn.linear_model import LogisticRegression


class SafeLogisticRegression:
    """
    A wrapper around sklearn's LogisticRegression that handles the edge case
    where the target variable `y` contains only a single class.

    If the target variable contains only a single class during training,
    the model will predict that class for all inputs without fitting the LogisticRegression model.

    Attributes:
        model (LogisticRegression): The underlying sklearn LogisticRegression model.
        single_class_ (Optional[int or float]): The single class present during training,
                                                or None if multiple classes are present.
    """

    def __init__(self, *args, **kwargs) -> None:
        """
        Initialize the SafeLogisticRegression wrapper.

        Args:
            *args: Positional arguments for sklearn's LogisticRegression.
            **kwargs: Keyword arguments for sklearn's LogisticRegression.
        """
        self.model = LogisticRegression(*args, **kwargs)
        self.single_class_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the SafeLogisticRegression model to the data.

        If the target variable `y` contains only a single unique class,
        the class is stored, and the LogisticRegression model is not fitted.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: Target array of shape (n_samples,).
        """
        unique_classes = np.unique(y)

        if len(unique_classes) == 1:
            self.single_class_ = unique_classes[0]

        else:
            self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for the input data.

        If the model was trained with only a single class, all predictions
        will be that class. Otherwise, predictions are made using the fitted LogisticRegression model.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            np.ndarray: Predicted class labels of shape (n_samples,).
        """
        if self.single_class_ is not None:
            return np.full(X.shape[0], self.single_class_)

        else:
            return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for the input data.

        If the model was trained with only a single class, the probabilities
        will reflect full certainty for that class (e.g., [0, 1] for class 1 or [1, 0] for class 0).
        Otherwise, probabilities are computed using the fitted LogisticRegression model.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            np.ndarray: Predicted probabilities of shape (n_samples, n_classes).
        """
        if self.single_class_ is not None:
            proba = np.zeros((X.shape[0], 1))
            return (
                np.hstack([proba, np.ones((X.shape[0], 1))])
                if self.single_class_ == 1
                else np.hstack([np.ones((X.shape[0], 1)), proba])
            )

        else:
            return self.model.predict_proba(X)
