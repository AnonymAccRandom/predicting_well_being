from typing import Optional, Union

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

from src.utils.Logger import Logger


class NonLinearImputer:
    """
    Implements a recursive partitioning-based imputation method for missing data, following the iterative imputation
    framework inspired by the Multiple Imputation by Chained Equations (MICE) algorithm. This class uses decision trees
    (classification or regression) to model and impute missing values for both categorical and continuous variables.
    See Doove et al. (2014) for details on the algorithm: http://dx.doi.org/10.1016/j.csda.2013.10.025

    The algorithm follows these steps:
    1. **Initialization**:
       - Randomly initialize missing values for each column by sampling from observed values or using fallback defaults.
       - Columns are processed in increasing order of missing values, ensuring that models are trained with as much
         information as possible.
    2. **Iterative Imputation**:
       - For each column with missing values:
         a. Fit a decision tree model (regression or classification) using the observed data.
         b. Use the tree model to predict "leaves" for the rows with missing values.
         c. Within each leaf, impute missing values by randomly sampling from observed values ("donors") within that leaf.
         d. Append the newly imputed data to the working dataset for subsequent column processing.
    3. **Repeat**:
       - Perform the above steps iteratively for a predefined number of iterations (`max_iter`).

    This method ensures consistency in imputation by leveraging relationships between features while preserving the
    statistical properties of the data.

    Attributes:
        logger (Logger): Logger instance for capturing warnings and status messages.
        max_iter (int): Maximum number of iterations to refine imputations. Default is 10.
        random_state (Optional[int]): Random seed for reproducibility.
        tree_max_depth (Optional[int]): Maximum depth for decision trees used in modeling.
        rng_ (np.random.RandomState): Local random number generator instance.
        models_ (dict): Dictionary storing fitted tree models for each column.
        fallback_values_ (dict): Default fallback values (e.g., mean for continuous, mode for categorical) for each column.
        columns_ (Optional[pd.Index]): List of columns in the dataset used during fitting.
        categorical_mappings_ (dict): Mapping of categorical variable indices to their unique values (for encoding/decoding).
    """

    def __init__(
        self,
        logger: Logger,
        max_iter: int = 10,
        random_state: Optional[int] = None,
        tree_max_depth: Optional[int] = None,
    ) -> None:
        """
        Initialize the NonLinearImputer.

        Args:
            logger: Logger instance for logging warnings and progress.
            max_iter: Maximum number of iterations for the imputation process. Default is 10.
            random_state: Seed for random operations to ensure reproducibility. Default is None.
            tree_max_depth: Maximum depth of the decision trees used for imputation. Default is None.
        """
        self.logger = logger
        self.max_iter = max_iter
        self.random_state = random_state
        self.rng_ = np.random.RandomState(self.random_state)
        self.models_ = {}
        self.fallback_values_ = {}
        self.columns_ = None
        self.categorical_mappings_ = {}
        self.tree_max_depth = tree_max_depth

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> "NonLinearImputer":
        """
        Fits the NonLinearImputer to the given dataset by iteratively building decision trees for each column with missing
        values and imputing those missing values.

        This method implements a recursive partitioning approach (similar to MICE with decision trees).
        For each column with missing data:
        - Initial imputations are created using random sampling from observed values in the same column.
        - Decision trees are fitted using observed values in the column as targets, and other columns as features.
        - Missing values are imputed using values predicted by the tree, based on the most similar observed cases
          (i.e., donors) determined by the tree structure.

        The process is repeated for a specified number of iterations (`max_iter`) to refine the imputations.
        Additionally, fallback values (mean or mode) are calculated for each column to handle edge cases.

        Args:
            X: DataFrame with missing values to be imputed. Columns can be a mix of continuous and categorical data.
            y: Optional target variable, not used in this implementation. Present for compatibility with sklearn API.

        Returns:
            NonLinearImputer: The fitted NonLinearImputer instance with imputed values stored in `_df_imputed_`.

        """
        np.random.seed(self.random_state)
        df_imputed = X.copy()
        self.columns_ = df_imputed.columns

        na_counts = df_imputed.isna().sum()
        columns_with_na = na_counts[na_counts > 0].sort_values().index.tolist()
        print("N columns with NaN:", len(columns_with_na))

        missing_indices_dict = {}

        for col in columns_with_na:
            obs_values = df_imputed[col].dropna().values
            missing_indices = df_imputed.index[df_imputed[col].isna()]
            if len(obs_values) > 0:
                df_imputed.loc[missing_indices, col] = self.rng_.choice(
                    obs_values, size=len(missing_indices)
                )
            else:
                df_imputed.loc[missing_indices, col] = 0  # default imputation

            missing_indices_dict[col] = missing_indices

        for col in df_imputed.columns:
            unique_values = df_imputed[col].dropna().unique()

            if len(unique_values) == 2 and set(unique_values).issubset({0, 1}):
                # Store the mode for binary columns (0/1)
                self.fallback_values_[col] = df_imputed[col].mode()[0]
            else:
                # Store the mean for continuous columns
                self.fallback_values_[col] = df_imputed[col].mean()

        for iteration in range(self.max_iter):
            for col in columns_with_na:
                observed_mask = df_imputed[col].notna()
                y_obs = df_imputed.loc[observed_mask, col]
                X_obs = df_imputed.loc[observed_mask, df_imputed.columns != col]

                if X_obs.shape[0] == 0:
                    continue

                if pd.api.types.is_numeric_dtype(y_obs) and len(np.unique(y_obs)) > 2:
                    tree_model = DecisionTreeRegressor(
                        random_state=self.random_state,
                        max_depth=self.tree_max_depth,
                    )

                else:
                    tree_model = DecisionTreeClassifier(
                        random_state=self.random_state, max_depth=self.tree_max_depth
                    )

                    y_obs, uniques = pd.factorize(y_obs)
                    self.categorical_mappings_[col] = uniques

                tree_model.fit(X_obs, y_obs)

                missing_indices = missing_indices_dict[col]
                X_mis = df_imputed.loc[missing_indices, df_imputed.columns != col]

                self._impute_values(
                    df_imputed, col, tree_model, y_obs, X_obs, X_mis, missing_indices
                )

                self.models_[col] = tree_model

        self._df_imputed_ = df_imputed

        return self

    def _impute_values(
        self,
        df_imputed: pd.DataFrame,
        col: str,
        model: Union[DecisionTreeRegressor, DecisionTreeClassifier],
        y_obs: Union[np.ndarray, pd.Series],
        X_obs: pd.DataFrame,
        X_mis: pd.DataFrame,
        missing_indices: pd.Index,
    ) -> None:
        """
        Impute missing values in a single column using a trained decision tree model and donor sampling.

        This method leverages the structure of a decision tree to identify donors for imputation:
        - The observed data (`X_obs`, `y_obs`) is used to fit the model.
        - For missing values (`X_mis`), the tree is applied to assign each sample to a leaf node.
        - Donors (observed values) in the same leaf are identified and used to impute missing values.

        If no donors are available for a given leaf, the fallback strategy uses:
        - The mean for continuous variables.
        - The mode for categorical variables.

        Steps:
        1. **Apply Tree to Missing and Observed Data**:
           - Use `model.apply()` to assign leaf indices for both observed (`X_obs`) and missing (`X_mis`) samples.
           - Leaves act as clusters of similar samples.

        2. **Identify Donors**:
           - For each unique leaf containing missing values:
             - Identify donors from observed samples (`y_obs`) in the same leaf.

        3. **Impute Values**:
           - If donors exist for a leaf, randomly sample from the donors to fill missing values.
           - If no donors exist for a leaf:
             - For continuous variables: Impute with the mean of `y_obs`.
             - For categorical variables: Impute with the mode of `y_obs`.

        4. **Restore Categories for Categorical Columns**:
           - If the column is categorical (tracked in `self.categorical_mappings_`), map numeric imputed values back to the original categories.

        Args:
            df_imputed: DataFrame that contains the data being imputed. Missing values in `col` will be replaced.
            col: Name of the column being imputed.
            model: Trained decision tree model (either regressor or classifier).
            y_obs: Observed values of the column being imputed, used as targets during tree training.
            X_obs: Observed data for other columns, used as predictors during tree training.
            X_mis: Missing data for other columns, used as predictors for imputation.
            missing_indices: Index positions of the rows in `df_imputed` where `col` has missing values.
        """
        leaves_for_missing = model.apply(X_mis)
        leaves_for_observed = model.apply(X_obs)

        y_obs_array = np.array(y_obs)

        imputed_values = np.empty(len(X_mis), dtype=df_imputed[col].dtype)
        leaves_unique = np.unique(leaves_for_missing)

        for leaf in leaves_unique:
            donor_mask = leaves_for_observed == leaf
            donors = y_obs_array[donor_mask]
            indices_in_leaf = np.where(leaves_for_missing == leaf)[0]

            if donors.size > 0:
                imputed_values_leaf = self.rng_.choice(
                    donors, size=len(indices_in_leaf), replace=True
                )
                imputed_values[indices_in_leaf] = imputed_values_leaf

            else:
                if np.issubdtype(df_imputed[col].dtype, np.number):
                    imputed_values[indices_in_leaf] = np.mean(y_obs_array)
                else:
                    imputed_values[indices_in_leaf] = pd.Series(y_obs_array).mode()[0]

        if col in self.categorical_mappings_:
            uniques = self.categorical_mappings_[col]
            imputed_values = uniques[imputed_values.astype(int)]

        df_imputed.loc[missing_indices, col] = imputed_values

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Impute missing values in the input DataFrame `X` using the fitted decision tree models.

        This method:
        - Uses the models fitted during `fit()` to impute missing values in `X`.
        - For columns that did not have missing values during training, fallback values (mean or mode) are used for imputation.
        - Handles categorical variables by ensuring consistent encoding and decoding with the training data.

        Steps:
        -------
        1. **Validation**:
           - Checks if all columns in the input DataFrame match the columns seen during `fit()`.
           - Raises a `ValueError` if there are mismatched columns.

        2. **Identify Columns with Missing Values**:
           - Identifies columns in `X` that contain missing values.

        3. **Imputation**:
           - For each column with missing values:
             - **Case 1**: If the column had no missing values during training but has missing values in `X`, use the fallback value (mean for continuous, mode for categorical).
             - **Case 2**: If the column was modeled during training:
               - Impute missing values using the trained decision tree model.
               - Randomly sample observed values from `X` as a fallback before applying the model.

        4. **Handle Categorical Variables**:
           - Ensures categorical variables are encoded consistently with the mappings learned during training.

        5. **Fallback for Remaining NaNs**:
           - After processing, any remaining NaN values in `X` are filled using fallback values.

        Args:
            X: A DataFrame with missing values to be imputed. The column names and structure must match the DataFrame used during `fit()`.

        Returns:
            pd.DataFrame: A DataFrame with missing values imputed. The structure and column order remain consistent with the input.
        """
        df = X.copy()
        if not all(col in df.columns for col in self.columns_):
            raise ValueError(
                "Input data must contain the same columns as during fitting."
            )

        columns_with_na = df.columns[df.isna().any()].tolist()
        for col in columns_with_na:
            missing_mask = df[col].isna()

            if col not in self.models_:
                if col in self.fallback_values_:
                    df.loc[missing_mask, col] = self.fallback_values_[col]
                continue

            model = self.models_[col]

            obs_values = df[col].dropna().values
            if len(obs_values) > 0:
                df.loc[missing_mask, col] = self.rng_.choice(
                    obs_values, size=missing_mask.sum()
                )

            X_mis = df.loc[missing_mask, df.columns != col]

            X_obs = self._df_imputed_.loc[
                self._df_imputed_[col].notna(), self._df_imputed_.columns != col
            ]
            y_obs = self._df_imputed_.loc[self._df_imputed_[col].notna(), col]

            if col in self.categorical_mappings_:
                uniques = self.categorical_mappings_[col]
                y_obs = pd.Categorical(y_obs, categories=uniques).codes

            else:
                y_obs = np.array(y_obs)

            self._impute_values(df, col, model, y_obs, X_obs, X_mis, missing_mask)

        for col in df.columns:
            if df[col].isna().any():
                if col in self.fallback_values_:
                    fallback_value = self.fallback_values_[col]

                    print(
                        f"Filling NaNs in column '{col}' with fallback value: {fallback_value}"
                    )
                    self.logger.log(
                        f"    WARNING: Filling NaNs in column '{col}' with fallback value: {fallback_value}"
                    )
                    df[col].fillna(fallback_value, inplace=True)

        return df
