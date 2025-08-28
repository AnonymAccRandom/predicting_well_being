from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.experimental import (
    enable_iterative_imputer,
)  # (!) do not delete, even if not explicitly used
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.analysis.AdaptiveImputerEstimator import AdaptiveImputerEstimator
from src.analysis.CustomIterativeImputer import CustomIterativeImputer
from src.analysis.CustomScaler import CustomScaler
from src.analysis.NonLinearImputer import NonLinearImputer
from src.analysis.SafeLogisticRegression import SafeLogisticRegression
from src.utils.Logger import Logger


class Imputer(BaseEstimator, TransformerMixin):
    """
    Custom imputer class supporting both linear (e.g., ElasticNet) and non-linear (e.g., RandomForestRegressor) imputation methods.

    This class handles:
    - Imputation at the country-level or individual-level (we fit seperate imputations models for a) the variables that
        vary by individual and b) for the variables that vary by country / year combinations)
    - In principle compatibility with predictive mean matching (PMM) and Bayesian posterior sampling (if a model that supports
        this is selected).

    The class is designed to be compatible with sklearns API (e.g., Pipeline ingegration, fit method, etc.).
    Linear imputations are performed using the IterativeImputer from sklearn, while non-linear imputations are performed using
    a custom tree-based imputation implementation (see "NonLinearImputer" class).

    Attributes:
        logger (Any): Logger object for logging messages.
        model (str): The name of the model used for imputation (e.g., "rfr", "enr").
        fix_rs (int): Fixed random state for reproducibility.
        num_imputations (int): Number of imputations to perform.
        max_iter (int): Maximum number of iterations for iterative imputation.
        conv_thresh (float): Convergence threshold (applicable only for tree-based imputations).
        tree_max_depth (int): Maximum depth of trees (applicable only for tree-based imputations).
        percentage_of_features (float): Percentage of features to use (applicable only for linear imputations).
        n_features_thresh (int): Threshold for the number of features (applicable only for linear imputations).
        sample_posterior (bool): If True, samples posterior values during imputation.
        pmm_k (int): Number of nearest neighbors to use for Predictive Mean Matching (PMM).
        country_group_by (str): Column name used for grouping countries.
        years_col (str): Column name containing year information.
        country_imputer (Optional[CustomIterativeImputer]): Imputer used for country-level imputation.
        individual_imputer (Optional[CustomIterativeImputer]): Imputer used for individual-level imputation.
        fitted_country_scaler (Optional[CustomScaler]): Fitted scaler for country-level data.
        fitted_individual_scaler (Optional[CustomScaler]): Fitted scaler for individual-level data.
    """

    def __init__(
        self,
        logger: Logger,
        model: str,
        fix_rs: int,
        num_imputations: int,
        max_iter: int,
        conv_thresh: float,
        tree_max_depth: int,
        percentage_of_features: float,
        n_features_thresh: int,
        sample_posterior: bool,
        pmm_k: int,
        country_group_by: str,
        years_col: str,
    ) -> None:
        """
        Initializes the Imputer with the specified configuration.

        Args:
            logger: Logger object for logging messages.
            model: Name of the model used for imputation (e.g., "rfr" for RandomForestRegressor, "enr" for ElasticNet).
            fix_rs: Random state for reproducibility.
            num_imputations: Number of imputations to perform.
            max_iter: Maximum number of iterations for iterative imputation.
            conv_thresh: Convergence threshold for RandomForestRegressor.
            tree_max_depth: Maximum depth for trees in RandomForestRegressor.
            percentage_of_features: Percentage of features to consider for ElasticNet.
            n_features_thresh: Minimum number of features for ElasticNet.
            sample_posterior: If True, enables Bayesian posterior sampling during imputation.
            pmm_k: Number of neighbors for Predictive Mean Matching (PMM).
            country_group_by: Column name to group data by country.
            years_col: Column name representing year information.
        """
        self.logger = logger
        self.model = model
        self.fix_rs = fix_rs
        self.num_imputations = num_imputations
        self.max_iter = max_iter
        self.conv_thresh = conv_thresh
        self.tree_max_depth = tree_max_depth
        self.percentage_of_features = percentage_of_features
        self.n_features_thresh = n_features_thresh
        self.sample_posterior = sample_posterior
        self.pmm_k = pmm_k
        self.country_group_by = country_group_by
        self.years_col = years_col
        self.logger = logger
        self.country_imputer = None
        self.individual_imputer = None
        self.fitted_country_scaler = None
        self.fitted_individual_scaler = None

    def fit(
        self, X: pd.DataFrame, num_imputation: int, y: pd.Series = None
    ) -> "Imputer":
        """
        Fits the imputer for both country-level and individual-level variables.

        This method:
        - Identifies country-level variables (prefixed with 'mac_') and fits a country-level imputer.
        - Identifies individual-level variables and fits an individual-level imputer.

        Args:
            X: DataFrame containing features with missing values.
            y: Ignored, included for compatibility with sklearn API.
            num_imputation: Number of the current imputation.

        Returns:
            Imputer: The fitted Imputer instance.
        """
        df = X.copy()
        country_var_cols = [col for col in df.columns if col.startswith("mac_")]

        if country_var_cols:
            self.logger.log(f"        Imputing country-level variables")
            self.country_imputer = self._fit_country_level_imputer(
                df=df, country_var_cols=country_var_cols, num_imputation=num_imputation
            )

        individual_var_cols = [
            col
            for col in df.columns
            if not col.startswith("mac_") and not col.startswith("other_")
        ]

        if individual_var_cols:
            self.logger.log(f"        Imputing individual-level variables")
            self.individual_imputer = self._fit_individual_level_imputer(
                df=df,
                individual_var_cols=individual_var_cols,
                num_imputation=num_imputation,
            )

        return self

    def transform(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        """
        Applies the appropriate imputation method based on the model type.

        This method:
        - Applies country-level imputation to variables prefixed with 'mac_'.
        - Applies individual-level imputation to all other variables (excluding 'other_' prefixed columns).
        - Combines the imputed data with non-imputed 'other_' columns.

        Args:
            X: DataFrame to apply imputations to.
            y: Ignored, included for compatibility with sklearn API.

        Returns:
            pd.DataFrame: Transformed DataFrame with imputed values.
        """
        df = X.copy()
        country_var_cols = [col for col in df.columns if col.startswith("mac_")]

        if country_var_cols and self.country_imputer:
            df = self._apply_country_level_imputation(
                df=df,
                country_var_cols=country_var_cols,
            )

        other_var_cols = pd.DataFrame(
            {col: df.pop(col) for col in df.columns if col.startswith("other_")}
        )
        individual_var_cols = [col for col in df.columns if not col.startswith("mac_")]

        if individual_var_cols and self.individual_imputer:
            df = self._apply_individual_level_imputation(
                df=df,
                individual_var_cols=individual_var_cols,
            )

        df_imputed = pd.DataFrame(df, columns=df.columns, index=df.index)
        df_imputed = pd.concat([df_imputed, other_var_cols], axis=1)

        if self.country_group_by in df_imputed.columns:
            df_imputed = df_imputed.drop(self.country_group_by, axis=1)

        return df_imputed

    def _fit_country_level_imputer(
        self, df: pd.DataFrame, country_var_cols: list[str], num_imputation: int
    ) -> CustomIterativeImputer:
        """
        Fits the imputer for country-level variables.

        This method:
        - Groups the data by country and year.
        - Scales the country-level variables.
        - Fits the appropriate imputer based on the model type (linear or non-linear).

        Args:
            df: DataFrame containing country-level variables.
            country_var_cols: List of column names for country-level variables.
            num_imputation: Number of the current imputation.

        Returns:
            CustomIterativeImputer: Fitted imputer for country-level variables.
        """
        country_df, _, _ = self.prepare_country_df(
            df=df.copy(), country_var_cols=country_var_cols
        )

        scaler_country_vars = CustomScaler()
        self.fitted_country_scaler = scaler_country_vars.fit(country_df)
        country_df_scaled = self.fitted_country_scaler.transform(country_df)

        if self.model == "elasticnet":
            country_imputer = self._fit_linear_imputer(
                df=country_df_scaled[country_var_cols],
                num_imputation=num_imputation,
            )

        elif self.model == "randomforestregressor":
            country_imputer = self._fit_nonlinear_imputer(
                df=country_df_scaled[country_var_cols],
                num_imputation=num_imputation,
            )

        else:
            raise ValueError(f"Imputations for model {self.model} not implemented")

        return country_imputer

    def _fit_individual_level_imputer(
        self, df: pd.DataFrame, individual_var_cols: list[str], num_imputation: int
    ) -> CustomIterativeImputer:
        """
        Fits the imputer for individual-level variables.

        This method:
        - Scales individual-level variables.
        - Fits the appropriate imputer based on the model type (linear or non-linear).

        Args:
            df: DataFrame containing individual-level variables.
            individual_var_cols: List of column names for individual-level variables.
            num_imputation: Number of the current imputation.

        Returns:
            CustomIterativeImputer: Fitted imputer for individual-level variables.
        """
        individual_df = df[individual_var_cols]

        scaler_individual_vars = CustomScaler()
        self.fitted_individual_scaler = scaler_individual_vars.fit(individual_df)
        individual_df_scaled = self.fitted_individual_scaler.transform(individual_df)

        individual_df_scaled = individual_df_scaled.drop(
            self.fitted_individual_scaler.other_cols, axis=1
        )

        if self.model == "elasticnet":
            self.logger.log(f"          Fit linear imputer")
            individual_imputer = self._fit_linear_imputer(
                df=individual_df_scaled, num_imputation=num_imputation
            )

        elif self.model == "randomforestregressor":
            self.logger.log(f"          Fit nonlinear imputer")
            individual_imputer = self._fit_nonlinear_imputer(
                df=individual_df_scaled, num_imputation=num_imputation
            )

        else:
            raise ValueError(f"Imputations for model {self.model} not implemented")

        return individual_imputer

    def prepare_country_df(
        self, df: pd.DataFrame, country_var_cols: list[str]
    ) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
        """
        Groups the data by country and year for imputing country-level variables.

        This method:
        - Explodes the years column to create a separate row for each year.
        - Groups the data by the specified country and year columns.
        - Prepares the DataFrame for imputation by aggregating the country-level variables.

        Args:
            df: DataFrame containing the input data.
            country_var_cols: List of column names corresponding to country-level variables.

        Returns:
            tuple: A tuple containing:
                - `country_df`: Grouped DataFrame with country-level variables.
                - `group_cols`: List of columns used for grouping (e.g., country and year).
                - `df_exploded`: Exploded version of the input DataFrame with rows expanded for each year.
        """
        df_exploded = df.copy()
        df_exploded["original_index"] = df_exploded.index

        df_exploded[self.years_col] = df[self.years_col].apply(
            lambda x: list(x) if isinstance(x, tuple) else [x]
        )
        df_exploded = df_exploded.explode(self.years_col)

        group_cols = [self.country_group_by, self.years_col]
        country_df = (
            df_exploded.groupby(group_cols)[country_var_cols].first().reset_index()
        )

        return country_df, group_cols, df_exploded

    def _apply_country_level_imputation(
        self, df: pd.DataFrame, country_var_cols: list[str]
    ) -> pd.DataFrame:
        """
        Applies country-level imputation to the specified columns.

        This method:
        - Prepares the DataFrame by grouping by country and year.
        - Scales the country-level variables.
        - Applies the fitted imputer to fill in missing values.
        - Rescales the imputed values and merges them back with the original DataFrame.

        Args:
            df: DataFrame to apply country-level imputation to.
            country_var_cols: List of column names corresponding to country-level variables.

        Returns:
            pd.DataFrame: DataFrame with imputed country-level variables, merged back into the original dataset.
        """
        df_tmp = df.copy()
        country_df, group_cols, df_exploded = self.prepare_country_df(
            df=df_tmp, country_var_cols=country_var_cols
        )
        country_df_scaled = self.fitted_country_scaler.transform(country_df)

        country_array_imputed_scaled = self.country_imputer.transform(
            X=country_df_scaled[country_var_cols],
        )

        country_df_imputed_scaled = pd.DataFrame(
            country_array_imputed_scaled,
            columns=country_var_cols,
        )

        country_df_imputed = self.fitted_country_scaler.inverse_transform(
            country_df_imputed_scaled
        )
        country_df_imputed[group_cols] = country_df[group_cols]

        df_exploded = df_exploded.drop(columns=country_var_cols)
        other_columns = df_exploded.columns.drop("original_index")
        df_exploded = df_exploded.merge(country_df_imputed, on=group_cols, how="left")

        individual_var_df = df[other_columns].copy()
        df_exploded_country = df_exploded.drop(columns=other_columns)
        df_country_grouped = df_exploded_country.groupby(
            df_exploded_country["original_index"]
        )

        df_country_aggregated = df_country_grouped.agg("mean")
        df_merged = pd.concat([individual_var_df, df_country_aggregated], axis=1)
        assert (
            df_merged.index.all() == df.index.all()
        ), "Indices between merged and original df not matching"

        return df_merged

    def _apply_individual_level_imputation(
        self, df: pd.DataFrame, individual_var_cols: list[str]
    ) -> pd.DataFrame:
        """
        Applies individual-level imputation to the specified columns.

        This method:
        - Scales the individual-level variables.
        - Applies the fitted imputer to fill in missing values.
        - Rescales the imputed values and merges them back into the dataset.

        Args:
            df: DataFrame to apply individual-level imputation to.
            individual_var_cols: List of column names corresponding to individual-level variables.

        Returns:
            pd.DataFrame: DataFrame with imputed individual-level variables, merged back into the original dataset.
        """
        individual_df = df[individual_var_cols]
        individual_df_scaled = self.fitted_individual_scaler.transform(individual_df)

        individual_array_imputed_scaled = self.individual_imputer.transform(
            X=individual_df_scaled[individual_var_cols],
        )

        individual_df_imputed_scaled = pd.DataFrame(
            individual_array_imputed_scaled,
            columns=individual_df.columns,
            index=individual_df.index,
        )
        individual_df_imputed = self.fitted_individual_scaler.inverse_transform(
            individual_df_imputed_scaled
        )

        df = df.drop(columns=individual_var_cols)
        df = pd.concat([individual_df_imputed, df], axis=1, join="outer")

        return df

    def _fit_linear_imputer(
        self, df: pd.DataFrame, num_imputation: int
    ) -> CustomIterativeImputer:
        """
        Fits a linear imputer using sklearn's IterativeImputer.

        This method:
        - Configures an adaptive estimator to handle both continuous and binary variables.
        - Reduces the number of features used for imputation based on a threshold.
        - Fits the imputer with specified settings.

        As a classifier, we use a enhaced version of sklearns implementation of logistic regression
        (see 'SafeLogisticRegression' class).

        Args:
            df: DataFrame to fit the imputer to.
            num_imputation: The number of the current imputation.

        Returns:
            CustomIterativeImputer: Fitted linear imputer.
        """

        n_features = int(len(df.columns) * self.percentage_of_features)
        if n_features < self.n_features_thresh:
            n_features = None

        binary_cols = df.columns[(df.isin([0, 1]) | df.isna()).all(axis=0)].tolist()
        binary_col_indices = [df.columns.get_loc(col) for col in binary_cols]

        adaptive_estimator = AdaptiveImputerEstimator(
            regressor=Ridge(),
            classifier=SafeLogisticRegression(penalty="l2"),
            categorical_idx=binary_col_indices,
        )

        imputer = CustomIterativeImputer(
            estimator=adaptive_estimator,
            sample_posterior=self.sample_posterior,
            max_iter=self.max_iter,
            random_state=self.fix_rs + num_imputation,
            categorical_idx=binary_col_indices,
            n_nearest_features=n_features,
            pmm_k=self.pmm_k,
        )

        imputer.fit(df)

        return imputer

    def _fit_nonlinear_imputer(
        self, df: pd.DataFrame, num_imputation: int
    ) -> NonLinearImputer:
        """
        Fits a nonlinear imputer using a random forest regressor.

        This method:
        - Configures the NonLinearImputer for tree-based imputation.
        - Fits the imputer to the data.

        For details on the method, see the "NonlinearImputer" class.

        Args:
            df: DataFrame to fit the imputer to.
            num_imputation: Number of imputations to perform.

        Returns:
            NonLinearImputer: Fitted nonlinear imputer.
        """
        imputer = NonLinearImputer(
            logger=self.logger,
            max_iter=self.max_iter,
            random_state=self.fix_rs + num_imputation,
            tree_max_depth=self.tree_max_depth,
        )

        imputer.fit(df)

        return imputer

    def evaluate_imputation_robustness(
        self,
        X: pd.DataFrame,
        num_imp: int,
        holdout_frac: float = 0.1,
        variables_to_evaluate: Optional[list[str]] = None,
        random_state: Optional[int] = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, float]:
        """
        Benchmarks this imputer by masking a slice of observed values, imputing them with a fresh clone, and computing quality metrics.

        The routine:
        • Selects candidate columns (numeric/boolean, not ``other_*``) unless an explicit list is supplied.
        • Masks *holdout_frac* of each candidate’s observed entries (≥ 1 value) using a reproducible RNG.
        • Fits an unfitted clone of ``self`` on the partly masked data, leaving the production instance untouched.
        • Compares the clone’s imputations with the ground truth and returns per-variable metrics:
          - ``RB``  – raw bias *E( Q̂ ) − Q*
          - ``PB``  – percent bias *100 × |RB / Q|* (``NaN`` if the true mean is 0)
          - ``MAE`` – mean absolute error
          - ``RMSE`` – root mean square error
          - ``n_holdout`` – number of masked cells

        Note: Not thoroughly tested and not used in the final version!

        Args:
            X: DataFrame to benchmark.
            num_imp: Number of imputations (passed to ``fit`` as ``num_imputation``).
            holdout_frac: Fraction of observed values to hide in each column.
            variables_to_evaluate: Optional explicit list of columns; if ``None`` auto-selects as described above.
            random_state: Seed for reproducibility; falls back to ``self.fix_rs``.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame, float]:
                - Metrics table sorted by ``RMSE`` (columns: ``variable``, ``n_holdout``, ``RB``, ``PB``, ``MAE``, ``RMSE``).
                - Summary table with mean (``M``) and standard deviation (``SD``) for ``RB``, ``PB``, ``MAE``, ``RMSE``.
                - Percent of features with ``PB`` below 5.
        """
        rng = np.random.default_rng(random_state or self.fix_rs)
        df = X.copy()

        if variables_to_evaluate is None:
            variables_to_evaluate = [
                c
                for c in df.columns
                if df[c].isna().sum() < len(df)
                and not c.startswith("other_")
                and pd.api.types.is_numeric_dtype(df[c])
            ]

        mask = pd.DataFrame(False, index=df.index, columns=df.columns)

        for col in variables_to_evaluate:
            candidates = df[col].dropna().index.to_numpy()
            if not len(candidates):
                continue
            n_holdout = max(1, int(len(candidates) * holdout_frac))
            hold_idx = rng.choice(candidates, n_holdout, replace=False)
            mask.loc[hold_idx, col] = True
            df.loc[hold_idx, col] = np.nan

        tmp = clone(self)
        tmp.fit(df, num_imputation=num_imp)
        df_imp = tmp.transform(df)

        rows = []
        for col in variables_to_evaluate:
            idx = mask.index[mask[col].values]
            if len(idx) == 0:
                continue
            true_vals = X.loc[idx, col]
            imp_vals = df_imp.loc[idx, col]
            rb = float((imp_vals - true_vals).mean())
            q_mean = float(true_vals.mean())
            pb = float(np.nan) if np.isclose(q_mean, 0) else 100.0 * abs(rb / q_mean)
            rows.append(
                {
                    "variable": col,
                    "n_holdout": len(idx),
                    "RB": rb,
                    "PB": pb,
                    "MAE": mean_absolute_error(true_vals, imp_vals),
                    "RMSE": mean_squared_error(true_vals, imp_vals, squared=False),
                }
            )

        metrics_df = (
            pd.DataFrame(rows)
            .sort_values("RMSE")
            .reset_index(drop=True)
            .loc[:, ["variable", "n_holdout", "RB", "PB", "MAE", "RMSE"]]
        )

        pb_below_5_pct = float((metrics_df["PB"] < 5).mean() * 100)

        summary_df = (
            metrics_df[["RB", "PB", "MAE", "RMSE"]]
            .agg(["mean", "std"])
            .T.rename(columns={"mean": "M", "std": "SD"})
        )

        self.logger.log(f"      Perc of Features below 5% bias: {pb_below_5_pct}")
        pretty = summary_df.round(3).to_string(justify="right")
        self.logger.log("      Summary DataFrame containing M and SD of metrics ")
        self.logger.log(f"      {pretty}")

        return metrics_df, summary_df, pb_below_5_pct

