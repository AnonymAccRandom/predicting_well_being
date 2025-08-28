import numpy as np
import pandas as pd
from missmecha.analysis import (
    MCARTest,
    compute_missing_rate,
)
from scipy.stats import ttest_ind, chi2_contingency
from statsmodels.regression.mixed_linear_model import MixedLM

from src.utils.utilfuncs import NestedDict, apply_name_mapping


class SampleMissingsAnalyzer:
    """
    Analyzes missingness patterns within and across individual samples for individual-level predictors.

    This class provides tools to:
    - Compare distributions between missing and non-missing entries across predictors.
    - Identify whether missingness is systematically related to observed values.
    - Assess between-sample variability using intraclass correlations (ICCs).
    - Save detailed test results and summary statistics for further inspection.

    Attributes:
        full_data: Stores the full input DataFrame for all samples and predictors.
        name_mapping: Mapping used to translate internal variable names into user-friendly labels.
        data_saver: Object providing methods to persist analysis results (e.g., to Excel or JSON).
    """

    def __init__(self, full_data: pd.DataFrame, name_mapping: NestedDict, data_saver: "DataSaver") -> None:
        """
        Initializes the SampleMissingsAnalyzer with the full dataset and required utilities.

        Args:
            full_data: The complete DataFrame containing individual-level predictor data.
            name_mapping: A nested dictionary used to convert technical variable names to descriptive labels.
            data_saver: A utility object responsible for saving outputs (e.g., Excel files).
        """
        self.full_data = full_data
        self.data_saver = data_saver
        self.name_mapping = name_mapping

    def compare_missings(self, method: str = "pairwise") -> None:
        """
        Compares whether systematic differences exist between missing and non-missing values
        for individual-varying predictors across all datasets.

        This method:
        - Filters out predictors unrelated to individuals (e.g., "other", "mac" prefixes).
        - Groups the dataset by prefixes found in the index to analyze sub-datasets.
        - Identifies columns with missing values and runs one of two tests:
            - "little": Performs Little's MCAR test to check for randomness in missingness.
            - "pairwise": Conducts pairwise t-tests to compare distributions of observed vs. missing groups.
        - Saves pairwise test results when applicable and prints key summaries to the console.

        Args:
            method: The statistical test to use for evaluating missingness patterns.
                - "little": Runs Little's MCAR test.
                - "pairwise": Performs pairwise t-tests and saves the results.
        """
        df = self.full_data.copy(deep=True)
        df = df.loc[:, ~pd.Index(df.columns).str.startswith(("other", "mac"))]

        prefixes = set(df.index.astype(str).str.extract(r"^([A-Za-z]+_+)")[0])

        for prefix in prefixes:
            sub_df = df[df.index.str.startswith(prefix)]

            sub_df = sub_df.dropna(axis=1, how="all")
            sub_df = sub_df.loc[:, sub_df.isna().any(axis=0)]

            if sub_df.shape[1] == 0 or sub_df.shape[0] == 0:
                continue

            print(prefix)
            compute_missing_rate(sub_df)

            if method == "little":
                mcartest_little = MCARTest(method=method)
                try:
                    mcartest_little(sub_df)  # Automatically prints results
                except np.linalg.LinAlgError:
                    print("WARNING: LinAlgError encountered, computations failed")
                    continue

            elif method == "pairwise":
                print(sub_df.columns)
                p_matrix, n_sig, n_non_sig = self._conduct_pairwise_missing_tests(sub_df)
                print(p_matrix.columns)
                filename = f"../results/descriptives/{prefix}pairwise_missing_tests.xlsx"
                self.data_saver.save_excel(p_matrix, filename)
                print(prefix)
                print(f"Number of non-significant tests: {n_non_sig}")
                print(f"Number of significant tests: {n_sig}")
                print("----")


    def compare_samples(self, decimals: int = 3) -> None:
        """
        Computes ICCs (intraclass correlations) for each variable across sample groups
        using a one-way random-effects ANOVA.

        This method:
        - Copies the full dataset.
        - Removes variables starting with "other" or "mac".
        - Extracts the prefix from the index to identify sample groups.
        - Fits a random-intercept model: y ~ 1 + (1|prefix) for each variable.
        - Returns ICC values only.

        Args:
            decimals: Number of decimals for rounding.

        Returns:
            pd.DataFrame: Results with columns ['variable', 'icc']
        """
        df = self.full_data.copy(deep=True)
        df = df.loc[:, ~pd.Index(df.columns).str.startswith(("other", "mac"))].copy()
        df["prefix"] = (
            df.index.to_series().astype(str).str.extract(r"^([A-Za-z]+_+)").iloc[:, 0].fillna("__unlabeled__")
        )

        results = []
        for col in df.columns.drop("prefix"):
            sub_df = df[["prefix", col]].dropna()
            if sub_df["prefix"].nunique() < 2:
                results.append((col, np.nan))
                continue

            model = MixedLM.from_formula(f"{col} ~ 1", groups="prefix", data=sub_df)
            fit = model.fit(reml=True, method="lbfgs", disp=False)

            var_between = fit.cov_re.iloc[0, 0]
            var_within = fit.scale
            icc = var_between / (var_between + var_within)

            print(f"{col}: {icc}")
            results.append((col, icc))

        result = pd.DataFrame(results, columns=["variable", "icc"]).round(decimals)# .sort_values("icc", ascending=False)
        mean_icc = result["icc"].mean()
        sd_icc = result["icc"].std()

        result["variable"] = apply_name_mapping(
            result["variable"].tolist(),
            self.name_mapping,
            prefix=True  # or False, depending on your variable naming format
        )

        filename = f"../results/descriptives/ICCs.xlsx"
        self.data_saver.save_excel(result, filename)
        print(f"Mean ICC: {mean_icc}")
        print(f"SD ICC: {sd_icc}")

    def _conduct_pairwise_missing_tests(
            self,
            X: pd.DataFrame,
            decimals: int = 3
    ) -> pd.DataFrame:
        """
        Conducts pairwise statistical tests to assess whether the presence of missing values
        in one variable is systematically related to the values of other variables.

        This method:
        - Iterates over all variable pairs in the dataset.
        - Splits other variables into groups based on whether the current variable is missing.
        - Applies:
            - Chi-squared test if the target variable is binary.
            - Welch’s t-test if the target variable is continuous.
        - Stores p-values in a symmetric matrix and rounds them.
        - Replaces technical variable names with descriptive names using a name mapping.
        - Counts the number of significant and non-significant test results.

        Args:
            X: A DataFrame containing variables to test for pairwise missingness dependence.
            decimals: Number of decimal places to round p-values in the output matrix.

        Returns:
            pd.DataFrame: Symmetric matrix of p-values with descriptive labels.
            int: Number of statistically significant tests (p < 0.05).
            int: Number of non-significant tests (p ≥ 0.05).
        """
        vars = X.columns
        p_matrix = pd.DataFrame(np.nan, index=vars, columns=vars)

        for var in vars:
            for tvar in vars:
                group1 = X.loc[X[var].isnull(), tvar].dropna()
                group2 = X.loc[X[var].notnull(), tvar].dropna()

                is_binary = X[tvar].nunique() <= 2

                if len(group1) > 1 and len(group2) > 1:
                    if is_binary:
                        contingency = pd.crosstab(
                            pd.Series([0] * len(group1) + [1] * len(group2), name="missing_flag"),
                            pd.Series(np.concatenate([group1, group2]), name="value")
                        )
                        p = chi2_contingency(contingency, correction=False)[1]
                    else:
                        p = ttest_ind(group1, group2, equal_var=False).pvalue

                    p_matrix.loc[var, tvar] = p

        p_matrix = p_matrix.round(decimals=decimals)

        # Apply pretty names to both rows and columns
        pretty_names = apply_name_mapping(list(vars), self.name_mapping, prefix=True)
        assert len(vars) == len(pretty_names), "Not all vars were transformed to pretty format"

        p_matrix.index = pretty_names
        p_matrix.columns = pretty_names

        n_sig = (p_matrix < 0.05).sum().sum()
        n_non_sig = (p_matrix >= 0.05).sum().sum()

        return p_matrix, n_sig, n_non_sig
