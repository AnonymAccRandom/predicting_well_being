import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler


class CustomScaler(BaseEstimator, TransformerMixin):
    """
    CustomScaler scales continuous columns while leaving binary columns unchanged.

    The scaler:
    - Automatically detects continuous and binary columns during fitting.
    - Uses `StandardScaler` for scaling continuous features.
    - Leaves binary features and other specified columns unscaled.

    This class is compatible with scikit-learn's `Pipeline` and metadata routing and adheres to sklearns interfaces
    (e.g., in the fit, transform, and inverse_transform methods).

    Attributes:
        scaler (StandardScaler): Scaler used for continuous features.
        continuous_cols (Optional[List[str]]): List of continuous feature names determined during fitting.
        binary_cols (Optional[List[str]]): List of binary feature names determined during fitting.
        other_cols (Optional[List[str]]): List of other feature names (e.g., prefixed with `other_`) excluded from scaling.
    """

    def __init__(self) -> None:
        """
        Initializes the CustomScaler instance.
        """
        self.scaler = StandardScaler()
        self.continuous_cols = None
        self.binary_cols = None
        self.other_cols = None

    def fit(self, X: pd.DataFrame, y=None) -> "CustomScaler":
        """
        Fits the scaler to the continuous columns in X. Adheres to sklearns fit interface.

        Args:
            X: Input DataFrame with features to fit the scaler.
            y: Ignored. Included for compatibility with scikit-learn's API.

        Returns:
            CustomScaler: The fitted scaler instance.

        Raises:
            TypeError: If X is not a pandas Data
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input X must be a pandas DataFrame.")

        self._separate_binary_continuous_cols(X)

        if self.continuous_cols:
            self.scaler.fit(X[self.continuous_cols])

        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """
        Scales the continuous columns in X and leaves binary and other columns unchanged.

        Args:
            X: Input DataFrame with features to transform.
            y: Ignored. Included for compatibility with scikit-learn's API.

        Returns:
            pd.DataFrame: Transformed DataFrame with scaled continuous features, unscaled binary features,
                          and original column order.

        Raises:
            TypeError: If X is not a pandas DataFrame.
            RuntimeError: If the scaler has not been fitted before calling transform.
        """
        X_cont = X[self.continuous_cols]
        try:
            X_other = X[self.other_cols]
        except KeyError:
            X_other = pd.DataFrame()

        if not isinstance(X_cont, pd.DataFrame):
            raise TypeError("Input X must be a pandas DataFrame.")

        if self.continuous_cols is None or self.binary_cols is None:
            raise RuntimeError("CustomScaler has not been fitted yet.")

        if self.continuous_cols:
            X_scaled = self.scaler.transform(X_cont)
            X_scaled_df = pd.DataFrame(
                X_scaled, columns=self.continuous_cols, index=X_cont.index
            )
        else:
            X_scaled_df = pd.DataFrame(index=X_cont.index)

        X_binary = (
            X[self.binary_cols]
            if self.binary_cols
            else pd.DataFrame(index=X_cont.index)
        )

        X_processed_df = pd.concat([X_scaled_df, X_binary, X_other], axis=1)
        X_processed_df = X_processed_df[X.columns]  # maintain coriginal olumn order

        return X_processed_df

    def inverse_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """
        Inverses the scaling of continuous features and restores original values.

        Args:
            X: Input DataFrame with transformed features to inverse-transform.
            y: Ignored. Included for compatibility with scikit-learn's API.

        Returns:
            pd.DataFrame: DataFrame with continuous features restored to their original scale.

        Raises:
            TypeError: If X is not a pandas DataFrame.
            RuntimeError: If the scaler has not been fitted before calling inverse_transform.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input X must be a pandas DataFrame.")

        if self.continuous_cols is None or self.binary_cols is None:
            raise RuntimeError("CustomScaler has not been fitted yet.")

        if self.continuous_cols:
            X_continuous = X[self.continuous_cols]
            X_continuous_original = self.scaler.inverse_transform(X_continuous)
            X_continuous_original_df = pd.DataFrame(
                X_continuous_original, columns=self.continuous_cols, index=X.index
            )

        else:
            X_continuous_original_df = pd.DataFrame(index=X.index)

        X_binary = (
            X[self.binary_cols] if self.binary_cols else pd.DataFrame(index=X.index)
        )

        X_original_df = pd.concat([X_continuous_original_df, X_binary], axis=1)
        X_original_df = X_original_df[X.columns]

        return X_original_df

    def _separate_binary_continuous_cols(self, X: pd.DataFrame) -> None:
        """
        Determines continuous and binary features based on the input data.

        Args:
            X: Input DataFrame to analyze and classify columns.

        Sets:
            binary_cols: List of binary feature names.
            continuous_cols: List of continuous feature names.
            other_cols: List of other feature names (e.g., prefixed with "other_").
        """
        data = X.copy()
        data = data.drop(
            [col for col in data.columns if col.startswith("other_")], axis=1
        )

        binary_cols = data.columns[(data.isin([0, 1]) | data.isna()).all()]
        continuous_cols = [col for col in data.columns if col not in binary_cols]
        other_cols = [col for col in X.columns if col.startswith("other_")]

        self.binary_cols = binary_cols.tolist()
        self.continuous_cols = continuous_cols
        self.other_cols = other_cols
