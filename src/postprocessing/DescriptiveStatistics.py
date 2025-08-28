import os
from typing import Union, Optional

import numpy as np
import pandas as pd
from src.utils.DataLoader import DataLoader
import openpyxl
import pingouin as pg

from src.utils.DataSaver import DataSaver
from src.utils.utilfuncs import (
    apply_name_mapping,
    format_df,
    separate_binary_continuous_cols,
    NestedDict,
    remove_leading_zero,
    inverse_code,
)


class DescriptiveStatistics:
    """
    Computes descriptive statistics as specified in the PreRegistration.

    This includes:
    - Mean (M) and Standard Deviation (SD) of all features and criteria used in the ML-based analysis.
    - Descriptive statistics for individual well-being (wb) items and wb scores (criteria), including:
        - Mean (M) and Standard Deviation (SD).
        - Intraclass Correlation Coefficients (ICC1).
        - Within-person (WP) and between-person (BP) correlations.
    - Reliability of the wb scores per dataset.

    The class is initialized with configuration dictionaries, name mappings, and the full dataset used
    for the descriptive analysis.

    Attributes:
        cfg_preprocessing (NestedDict): Yaml config specifying details on preprocessing (e.g., scales, items).
        cfg_analysis (NestedDict): Yaml config specifying details on the ML analysis (e.g., CV, models).
        cfg_postprocessing (NestedDict): Yaml config specifying details on postprocessing (e.g., tables, plots).
        desc_cfg (NestedDict): Config for generating descriptives, part of cfg_postprocessing.
        name_mapping (NestedDict): Mapping of feature names for presentation purposes.
        data_loader (DataLoader): Utility for loading different data formats.
        data_saver (DataSaver): Utility for saving results in various formats.
        datasets (list[str]): List of datasets to include in the analysis.
        esm_id_col_dct (dict): Dictionary mapping datasets to their unique ID column for ESM data.
        esm_tp_col_dct (dict): Dictionary mapping datasets to their time-point column for ESM data.
        data_base_path (str): Path to the preprocessed data used for analysis.
        desc_results_base_path (str): Path where the descriptive results will be saved.
        full_data (pd.DataFrame): Full dataset containing all features and criteria.
    """

    def __init__(
        self,
        cfg_preprocessing: NestedDict,
        cfg_analysis: NestedDict,
        cfg_postprocessing: NestedDict,
        name_mapping: NestedDict,
        full_data: pd.DataFrame,
        base_result_path: str,
    ) -> None:
        """
        Initializes the DescriptiveStatistics class.

        Args:
           cfg_preprocessing: Yaml config specifying details on preprocessing (e.g., scales, items).
           cfg_analysis: Yaml config specifying details on the ML analysis (e.g., CV, models)
           cfg_postprocessing: Yaml config specifying details on postprocessing (e.g., tables, plots)
           name_mapping: Mapping of feature names in code to features names in paper.
           full_data: Full dataset containing all features and criteria used in the analysis.
           base_result_path: Base paths of all results.
        """
        self.cfg_preprocessing = cfg_preprocessing
        self.cfg_analysis = cfg_analysis
        self.cfg_postprocessing = cfg_postprocessing
        self.desc_cfg = self.cfg_postprocessing["create_descriptives"]
        self.name_mapping = name_mapping

        self.data_loader = DataLoader()
        self.data_saver = DataSaver()
        self.datasets = self.cfg_preprocessing["general"]["datasets_to_be_included"]

        self.esm_id_col_dct = self.cfg_preprocessing["general"]["esm_id_col"]
        self.esm_tp_col_dct = self.cfg_preprocessing["general"]["esm_timestamp_col"]

        self.data_base_path = self.cfg_preprocessing["general"][
            "path_to_preprocessed_data"
        ]
        self.base_result_path = base_result_path
        self.desc_results_base_path = os.path.join(
            self.base_result_path,
            self.cfg_postprocessing["general"]["data_paths"]["descriptives"],
        )

        self.full_data = full_data

    def create_m_sd_var_table(
        self,
        vars_to_include: list[str],
        binary_stats_to_calc: list[str],
        continuous_stats_to_calc: dict[str, str],
        table_decimals: int,
        store_table: bool,
        filename: str,
        store_index: bool,
    ) -> None:
        """
        Creates a table with descriptive statistics for variables grouped by prefixes.

        - Computes mean (M) and standard deviation (SD) for continuous variables.
        - Computes frequencies (counts and percentages) for binary variables.
        - Groups variables based on their prefixes (e.g., 'pl_', 'sens_', 'srmc_', 'mac_').
        - Formats the table in APA style and optionally saves it as an Excel file.

        Args:
            vars_to_include: List of variable prefixes to group and analyze.
            binary_stats_to_calc: Dictionary of statistics to compute for binary variables (e.g., counts, percentages).
            continuous_stats_to_calc: Dictionary of aggregation functions for continuous variables (e.g., mean, SD).
            table_decimals: Number of decimal places to format numeric values in the final table.
            store_table: Boolean indicating whether to save the final table as an Excel file.
            filename: Filename for saving the table if `store_table` is True.
            store_index: If to store the df index in the excel file.
        """
        full_df = self.full_data.copy()
        results = []

        for cat in vars_to_include:
            prefix = f"{cat}_"
            prefixed_cols = [col for col in full_df.columns if col.startswith(prefix)]
            if not prefixed_cols:
                continue

            df_subset = full_df[prefixed_cols]
            binary_vars, continuous_vars = separate_binary_continuous_cols(df_subset)

            if continuous_vars:
                cont_stats = self.calculate_cont_stats(
                    df=df_subset,
                    continuous_vars=continuous_vars,
                    prefix=prefix,
                    stats=continuous_stats_to_calc,
                    var_as_index=False,
                )
                results.append(cont_stats)

            if binary_vars:
                bin_stats = self.calculate_bin_stats(
                    df=df_subset,
                    binary_vars=binary_vars,
                    prefix=prefix,
                    stats=binary_stats_to_calc,
                )
                results.append(bin_stats)

        final_table = pd.concat(results, ignore_index=True)
        final_table = format_df(
            df=final_table,
            capitalize=False,
            decimals=table_decimals,
        )

        final_table["Group"] = final_table["Group"].replace(
            self.name_mapping["category_names"]
        )
        final_table = final_table[
            ["Group"] + [col for col in final_table.columns if col != "Group"]
        ]

        final_table["M / %"] = final_table.apply(
            lambda row: row["M"] if pd.notna(row["M"]) else row["%"],
            axis=1,
        )
        final_table = final_table.drop(columns=["M", "%"])

        final_table["scale_endpoints"] = final_table["Variable"].apply(
            lambda feat: self.get_scale_endpoints(self.cfg_preprocessing.copy(), feat)
        )

        # reorder columns
        cols = list(final_table.columns)
        m_col, sd_col = cols.index("M / %"), cols.index("SD")
        cols[m_col], cols[sd_col] = cols[sd_col], cols[m_col]
        final_table = final_table[cols]

        final_table["Variable"] = apply_name_mapping(
            features=list(final_table["Variable"]),
            name_mapping=self.name_mapping,
            prefix=True,
        )

        final_table = final_table.reset_index(drop=True)

        if store_table:
            file_path = os.path.join(self.desc_results_base_path, filename)
            self.data_saver.save_excel(final_table, file_path, index=store_index)

    def get_scale_endpoints(
        self, data: Union[NestedDict, dict, list], feature_name: str
    ) -> Union[np.nan, tuple[float, float]]:
        """
        Searches a nested dictionary or list structure to locate an entry where the "name" key matches the given feature name.
        If found, it extracts and returns the "scale_endpoints" as a tuple of (min, max). Handles feature names with specific prefixes.

        Args:
            data (Union[dict, list]): The nested structure (dictionary or list) to search.
            feature_name (str): The name of the feature to search for.
                If the name starts with a prefix ("pl_", "srmc_", "mac_", "sens:"), the prefix is stripped before matching.

        Returns:
            Union[np.nan, Tuple[float, float]]:
                - A tuple (min, max) if the feature is found and has scale endpoints.
                - None if the feature is not found or does not have scale endpoints.

        Notes:
            - Supported prefixes for feature names include: "pl_", "srmc_", "mac_", and "sens:".
            - The method supports recursive search through both dictionary values and list elements.
        """
        if any(
            feature_name.startswith(prefix)
            for prefix in ["pl_", "srmc_", "mac_", "sens:"]
        ):
            feature_name_no_prefix = feature_name.split("_", 1)[-1]
        else:
            feature_name_no_prefix = feature_name

        if isinstance(data, dict):
            # Check if this dictionary node has the target feature name
            if data.get("name") == feature_name_no_prefix:
                scale_endpoints = data.get("scale_endpoints")
                if scale_endpoints is not None:
                    return scale_endpoints["min"], scale_endpoints["max"]

            for key, value in data.items():
                result = self.get_scale_endpoints(value, feature_name_no_prefix)
                if result is not None:
                    return result

        elif isinstance(data, list):
            for item in data:
                result = self.get_scale_endpoints(item, feature_name_no_prefix)
                if result is not None:
                    return result

        return None

    @staticmethod
    def calculate_cont_stats(
        df: pd.DataFrame,
        continuous_vars: list[str],
        stats: dict[str, str],
        var_as_index: bool = True,
        prefix: Union[str, None] = None,
    ) -> pd.DataFrame:
        """
        Calculates descriptive statistics for continuous variables in APA format.

        Args:
            df: DataFrame containing the data.
            continuous_vars: List of column names corresponding to continuous variables.
            stats: Dictionary where keys are aggregation functions (e.g., 'mean', 'std') and
                   values are their desired column names in the output.
            var_as_index: If True, uses variable names as the DataFrame index. Defaults to True.
            prefix: Optional prefix to add as a "Group" column in the output.

        Returns:
            pd.DataFrame: A DataFrame containing the calculated statistics with descriptive column names.
        """
        aggregated = df[continuous_vars].agg(list(stats.keys())).transpose()
        aggregated.columns = [stats[func] for func in stats.keys()]

        if not var_as_index:
            aggregated = aggregated.reset_index().rename(columns={"index": "Variable"})

        if prefix:
            aggregated["Group"] = prefix.rstrip("_")

        return aggregated

    @staticmethod
    def calculate_bin_stats(
        df: pd.DataFrame, binary_vars: list[str], prefix: str, stats: list[str]
    ) -> Union[pd.DataFrame, None]:
        """
        Calculates binary statistics for specified variables.

        - Computes frequency, total count, and percentage of occurrences for each binary variable.
        - Formats the output in APA style with options for custom statistics.

        Args:
            df: DataFrame containing the data.
            binary_vars: List of binary variable names to analyze.
            prefix: Prefix to add as a group label in the output.
            stats: List of statistics to include, such as '%' for percentage.

        Returns:
            pd.DataFrame: A DataFrame with binary statistics including 'Group', 'Variable', and '%'.
        """
        if "%" in stats:
            bin_stats = (
                df[binary_vars].apply(lambda x: x.value_counts(dropna=True)).transpose()
            )

            bin_stats.index.name = "Variable"
            bin_stats = bin_stats.reset_index()

            # Calculate statistics
            bin_stats["Frequency"] = bin_stats.get(1, bin_stats.get(1.0, np.nan))
            bin_stats["Total"] = df[binary_vars].notna().sum().values
            bin_stats["%"] = (bin_stats["Frequency"] / bin_stats["Total"]) * 100
            bin_stats["Group"] = prefix.rstrip("_")

            return bin_stats[["Group", "Variable", "%"]].reset_index(drop=True)

        else:
            return None


    def create_wb_items_stats_per_dataset(
        self,
        dataset: str,
        state_df: pd.DataFrame,
        trait_df: pd.DataFrame,
        esm_id_col: str,
        esm_tp_col: str,
    ) -> dict[str, Optional[pd.DataFrame]]:
        """
        Computes well-being item statistics for a given dataset.

        This function:
        - Takes state and trait data for the specified dataset as input.
        - Applies name mapping, reorders columns, and renames them with "state_" or "trait_" prefixes.
        - Calculates descriptive statistics for well-being items (M, SD).
        - Optionally computes between-person (BP) and within-person (WP) correlations, and ICC1/ICC2 for state items.
        - Computes trait correlations if trait data is available.

        Args:
            dataset: Name of the dataset for which statistics are computed.
            state_df: pd.DataFrame containing state data for the given dataset.
            trait_df: pd.DataFrame containing trait data for the given dataset.
            esm_id_col: Column name identifying unique IDs in the ESM data.
            esm_tp_col: Column name identifying time points in the ESM data.

        Returns:
            dict: A dictionary containing:
                - "m_sd": Descriptive statistics for well-being items (M, SD).
                - "wp_corr": Within-person correlation (if applicable).
                - "bp_corr": Between-person correlation (if applicable).
                - "icc1": ICC1 (if applicable).
                - "icc2": ICC2 (if applicable).
                - "trait_corr": Correlation matrix for trait items (if applicable).
        """
        result_dct = {}

        if state_df is not None:
            state_df = state_df.drop(
                columns=["wb_state", "pa_state", "na_state", "country"], errors="ignore"
            )

            if dataset == "emotions":
                state_df = self.merge_wb_items(
                    state_df, prefix_a="occup_", prefix_b="int_"
                )

            state_df.columns = apply_name_mapping(
                features=state_df.columns, name_mapping=self.name_mapping, prefix=False
            )
            state_order = self.desc_cfg["wb_items"]["state_order"]

            state_df = state_df[
                [col for col in state_df.columns if col not in state_order]
                + [col for col in state_order if col in state_df.columns]
            ]
            state_df.columns = [
                f"state_{col}" if col not in [esm_id_col, esm_tp_col] else col
                for col in state_df.columns
            ]

        if trait_df is not None:
            trait_df = trait_df.drop(
                columns=["wb_trait", "pa_trait", "na_trait", "country"], errors="ignore"
            )

            trait_df.columns = apply_name_mapping(
                features=trait_df.columns, name_mapping=self.name_mapping, prefix=False
            )
            trait_order = self.desc_cfg["wb_items"]["trait_order"]

            trait_df = trait_df[
                [col for col in trait_df.columns if col not in trait_order]
                + [col for col in trait_order if col in trait_df.columns]
            ]
            trait_df.columns = [f"trait_{col}" for col in trait_df.columns]

        wb_items_df = (
            pd.concat([state_df, trait_df], axis=0, ignore_index=True)
            if state_df is not None or trait_df is not None
            else None
        )

        if wb_items_df is not None:
            df_filtered = wb_items_df.drop(columns=esm_id_col, errors="ignore")
            cols_filtered = df_filtered.select_dtypes(include=[np.number]).columns

            m_sd_df_wb_items = self.calculate_cont_stats(
                df=df_filtered,
                continuous_vars=cols_filtered,
                stats=self.desc_cfg["m_sd_table"]["cont_agg_dct"],
                var_as_index=True,
            )
            result_dct["m_sd"] = m_sd_df_wb_items

            if (
                state_df is not None
                and dataset in self.esm_id_col_dct
                and dataset in self.esm_tp_col_dct
            ):
                wp_corr, bp_corr, icc1, icc2 = self.calc_bp_wp_statistics(
                    df=state_df,
                    id_col=self.esm_id_col_dct[dataset],
                    tp_col=self.esm_tp_col_dct[dataset],
                )

                result_dct.update(
                    {"wp_corr": wp_corr, "bp_corr": bp_corr, "icc1": icc1, "icc2": icc2}
                )

            result_dct["trait_corr"] = trait_df.corr() if trait_df is not None else None

        return result_dct

    @staticmethod
    def merge_wb_items(
        state_df: pd.DataFrame, prefix_a: str, prefix_b: str
    ) -> pd.DataFrame:
        """
        Merges columns in `state_df` that differ only by the specified prefixes.

        - Columns with matching suffixes for the given prefixes are averaged into a new column.
        - The new column name is derived from the suffix without any prefix.
        - Original columns with the specified prefixes are dropped after merging.

        Args:
            state_df: Input DataFrame containing columns to merge.
            prefix_a: First prefix to identify columns.
            prefix_b: Second prefix to identify columns.

        Returns:
            pd.DataFrame: A new DataFrame with merged columns.
        """
        if not isinstance(prefix_a, str) or not isinstance(prefix_b, str):
            raise ValueError("prefix_a and prefix_b must be strings.")

        cols_a = [col for col in state_df.columns if col.startswith(prefix_a)]
        cols_b = [col for col in state_df.columns if col.startswith(prefix_b)]

        common_suffixes = {col[len(prefix_a) :] for col in cols_a}.intersection(
            col[len(prefix_b) :] for col in cols_b
        )

        if not common_suffixes:
            print("No common suffixes found between the provided prefixes.")
            return state_df.copy()

        merged_df = state_df.copy()
        for suffix in common_suffixes:
            col_a, col_b = prefix_a + suffix, prefix_b + suffix

            merged_df[suffix] = state_df[[col_a, col_b]].mean(axis=1)
            merged_df.drop([col_a, col_b], axis=1, inplace=True)

        return merged_df

    @staticmethod
    def calc_bp_wp_statistics(
        df: pd.DataFrame, id_col: str, tp_col: str
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Calculates within-person (WP) and between-person (BP) statistics for the given variables
        (see also Revelle, R-package "psych").

        This method computes:
        - Within-person (pooled within-group) correlations.
        - Between-person (between-group) correlations.
        - Intraclass Correlation Coefficients (ICC1 and ICC2).

        Methodology:
        - Variables to analyze are determined by excluding the `id_col` and `tp_col` columns.
        - The function calculates sum of squares (SSB and SSW) for each variable:
            - SSB (Between-group sum of squares): Measures variance between groups.
            - SSW (Within-group sum of squares): Measures variance within groups.
        - Mean squares (MSb and MSw) are calculated from SSB and SSW:
            - ICC1: Proportion of variance attributable to group membership.
            - ICC2: Reliability of group means as a measure of the construct.
        - Correlation matrices:
            - WP correlations: Correlations of variables within groups.
            - BP correlations: Correlations of group-level means.

        Args:
            df: DataFrame containing the data with variables, group IDs, and time points.
            id_col: Column name identifying group IDs (e.g., person or cluster identifiers).
            tp_col: Column name identifying time points within groups.

        Returns:
            tuple:
                - corr_W (pd.DataFrame): Pooled within-person correlation matrix.
                - corr_B (pd.DataFrame): Between-person correlation matrix.
                - ICC1_series (pd.Series): ICC1 values for each variable.
                - ICC2_series (pd.Series): ICC2 values for each variable.
        """
        variables_to_correlate = df.columns.drop([id_col, tp_col], errors="ignore")
        N, K, n_list = len(df), df[id_col].nunique(), []
        overall_mean = df[variables_to_correlate].mean()
        SSB_dict, SSW_dict = {var: 0 for var in variables_to_correlate}, {
            var: 0 for var in variables_to_correlate
        }

        for group in df[id_col].unique():
            group_df = df[df[id_col] == group]
            n_i = len(group_df)
            n_list.append(n_i)
            group_means = group_df[variables_to_correlate].mean()
            for var in variables_to_correlate:
                SSB_dict[var] += n_i * (group_means[var] - overall_mean[var]) ** 2
                SSW_dict[var] += ((group_df[var] - group_means[var]) ** 2).sum()

        df_between, df_within, n_bar = K - 1, N - K, np.mean(n_list)
        MSb_dict, MSw_dict = {}, {}
        ICC1_series, ICC2_series = pd.Series(dtype=float), pd.Series(dtype=float)

        for var in variables_to_correlate:
            MSb = SSB_dict[var] / df_between
            MSw = SSW_dict[var] / df_within
            MSb_dict[var], MSw_dict[var] = MSb, MSw
            numerator, denominator = MSb - MSw, MSb + MSw * (n_bar - 1)
            ICC1 = numerator / denominator if denominator != 0 else np.nan
            ICC2 = (MSb - MSw) / MSb if MSb != 0 else np.nan
            ICC1_series[var], ICC2_series[var] = np.round(ICC1, 2), np.round(ICC2, 2)

        SSW = np.zeros((len(variables_to_correlate), len(variables_to_correlate)))
        SSB = np.zeros_like(SSW)
        overall_mean_vector = df[variables_to_correlate].mean().values

        for group in df[id_col].unique():
            group_df = df[df[id_col] == group]
            n_i = len(group_df)
            mu_i = group_df[variables_to_correlate].mean().values
            group_centered = group_df[variables_to_correlate] - mu_i

            # Compute within-group sum of squares and cross-products matrix
            cov_within = group_centered.cov(ddof=0)
            SSW_i = cov_within.values * (n_i - 1)
            SSW += SSW_i

            # Compute between-group sum of squares and cross-products matrix
            delta_mu = (mu_i - overall_mean_vector).reshape(-1, 1)  # Column vector
            SSB_i = n_i * np.dot(delta_mu, delta_mu.T)
            SSB += SSB_i

        S_W, S_B = SSW / df_within, SSB / df_between
        std_W, std_B = np.sqrt(np.diag(S_W)), np.sqrt(np.diag(S_B))
        corr_W = S_W / np.outer(std_W, std_W)
        corr_B = np.where(np.outer(std_B, std_B) == 0, 0, S_B / np.outer(std_B, std_B))

        corr_W = pd.DataFrame(
            corr_W, index=variables_to_correlate, columns=variables_to_correlate
        )
        corr_B = pd.DataFrame(
            corr_B, index=variables_to_correlate, columns=variables_to_correlate
        )

        return corr_W, corr_B, ICC1_series, ICC2_series

    def compute_rel(
        self,
        state_df: pd.DataFrame,
        trait_df: pd.DataFrame,
        dataset: str,
        decimals: int,
    ) -> dict[str, float]:
        """
        Computes the reliability of well-being measures for the given dataset.

        - Trait reliabilities are computed using internal consistency.
        - State reliabilities are computed using split-half reliability.
        - The function returns a combined dictionary of reliabilities with descriptive keys.

        Args:
            dataset: The dataset for which to compute reliability.
            state_df: pd.DataFrame containing state data for the given dataset.
            trait_df: pd.DataFrame containing trait data for the given dataset.
            decimals: decimals to round the rel values to.

        Returns:
            Dict[str, float]: A dictionary with criteria as keys and their reliabilities as values.
        """
        state_rel_series = pd.Series(dtype=float)
        trait_rel_series = pd.Series(dtype=float)
        crit_cols = self.cfg_preprocessing["var_assignments"]["crit"]
        crit_avail_dct = self.cfg_analysis["crit_available"]

        for crit in crit_cols:
            if dataset not in crit_avail_dct[crit]:
                continue

            if "trait" in crit:
                df = trait_df.copy()

                rel = self.compute_internal_consistency(
                    df=self.prepare_rel_data(
                        df=df,
                        dataset=dataset,
                        crit_type="trait",
                    ),
                    construct=crit,
                    dataset=dataset,
                )
                trait_rel_series[crit] = rel

            elif "state" in crit:
                df = state_df.copy()

                rel_state_cfg = self.cfg_postprocessing["create_descriptives"]["rel"][
                    "state"
                ]
                n_per_person_col_name = rel_state_cfg["n_per_person_col_name"]
                person_id_col_name = rel_state_cfg["id_name"]
                state_cols_to_select = rel_state_cfg["crits"] + [
                    n_per_person_col_name,
                    person_id_col_name,
                ]

                rel = self.compute_split_half_rel(
                    df=self.prepare_rel_data(
                        df=df,
                        dataset=dataset,
                        crit_type="state",
                        n_measures_person_col=n_per_person_col_name,
                        person_id_col=person_id_col_name,
                        state_cols_to_select=state_cols_to_select,
                    ),
                    dataset=dataset,
                    construct=crit,
                    user_id_col=person_id_col_name,
                    measurement_idx_col=n_per_person_col_name,
                )
                state_rel_series[crit] = rel

            else:
                raise ValueError("Unknown criterion")

        trait_rel_series.index = apply_name_mapping(
            features=list(trait_rel_series.index),
            name_mapping=self.name_mapping,
            prefix=False,
        )
        state_rel_series.index = apply_name_mapping(
            features=list(state_rel_series.index),
            name_mapping=self.name_mapping,
            prefix=False,
        )

        return {
            **state_rel_series.round(decimals).to_dict(),
            **trait_rel_series.round(decimals).to_dict(),
        }

    def prepare_rel_data(
        self,
        df: pd.DataFrame,
        dataset: str,
        crit_type: str,
        state_cols_to_select: Optional[list[str]] = None,
        n_measures_person_col: Optional[str] = None,
        person_id_col: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Prepares data for reliability computation.

        - For "state" data, processes timestamps, creates a per-person measurement index, and filters columns.
        - For "trait" data, loads the relevant dataset without additional processing.

        Args:
            dataset: The dataset identifier.
            df: DataFrame containing the data to process.
            crit_type: Type of criterion ("trait" or "state").
            state_cols_to_select: List of state columns to retain (used for "state" data).
            n_measures_person_col: Column name for the number of measures per person (used for "state" data).
            person_id_col: Column name for the unique person identifier (used for "state" data).

        Returns:
            pd.DataFrame: Processed data ready for reliability analysis.
        """
        if crit_type == "state":
            timestamp_col = self.cfg_preprocessing["general"]["esm_timestamp_col"][
                dataset
            ]
            id_col = self.cfg_preprocessing["general"]["esm_id_col"][dataset]

            df[timestamp_col] = pd.to_datetime(df[timestamp_col])
            df[n_measures_person_col] = (
                df.sort_values(by=timestamp_col).groupby(id_col).cumcount()
            )
            df[person_id_col] = df[id_col].apply(lambda x: f"{dataset}_{x}")
            df = df.sort_values([id_col, n_measures_person_col])

            if state_cols_to_select:
                df = df[df.columns.intersection(state_cols_to_select)]

        elif crit_type != "trait":
            raise ValueError('crit_type must be "trait" or "state"')

        return df

    def compute_internal_consistency(
        self, df: pd.DataFrame, construct: str, dataset: str
    ) -> Optional[float]:
        """
        Computes internal consistency (Cronbach's alpha) for the specified trait reliability measure.

        The function uses item names configured for each trait (e.g., "pa_trait", "na_trait", "wb_trait").
        It handles the inverse coding of certain items (for "wb_trait") and calculates Cronbach's alpha
        for the selected items.

        Args:
            df: DataFrame containing the items for the specified trait.
            construct: The trait for which to compute the internal consistency ("pa_trait", "na_trait", or "wb_trait").
            dataset: The dataset identifier used to fetch the correct item names.

        Returns:
            float or None: Cronbach's alpha value if the conditions are met, otherwise None.
        """
        item_cfg = self.cfg_preprocessing["person_level"]["criterion"]

        pa_trait_items = item_cfg[0]["item_names"].get(dataset, [])
        na_trait_items = item_cfg[1]["item_names"].get(dataset, [])

        item_dct = {
            "pa_trait": pa_trait_items,
            "na_trait": na_trait_items,
            "wb_trait": pa_trait_items + na_trait_items,
        }

        selected_items = item_dct[construct]

        if construct == "wb_trait" and selected_items:
            df[na_trait_items] = inverse_code(
                df[na_trait_items], min_scale=1, max_scale=6
            )

        df_filtered = df[selected_items] if selected_items else pd.DataFrame()
        if not df_filtered.empty and len(df_filtered.columns) > 1:
            alpha = pg.cronbach_alpha(data=df_filtered)[0]
        else:
            alpha = None

        return alpha

    @staticmethod
    def compute_split_half_rel(
        df: pd.DataFrame,
        dataset: str,
        construct: str,
        user_id_col: str,
        measurement_idx_col: str,
        method: str = "individual_halves",
        correlation_method: str = "pearson",
    ) -> Optional[float]:
        """
        Computes split-half reliability for a construct across users in a dataset.

        - Splits user data into two halves based on the specified method.
        - Calculates the correlation between the two halves using the specified correlation method.
        - Applies the Spearman-Brown formula to estimate reliability.

        Args:
            df: DataFrame containing the dataset.
            dataset: Identifier for the dataset being analyzed (used in error messages and output).
            construct: Column name representing the construct to analyze.
            user_id_col: Column name for unique user IDs (e.g., "joint_user_id").
            measurement_idx_col: Column name representing measurement indices for each user.
            method: Method for splitting data into halves ("odd_even" or "individual_halves"). Default is "individual_halves".
            correlation_method: Method for calculating correlation ("pearson" or "spearman"). Default is "pearson".

        Returns:
            Optional[float]: Estimated split-half reliability or NaN if insufficient data.
        """
        if construct not in df.columns:  # This may happen
            print(
                f"The construct '{construct}' is not a column in the DataFrame for dataset {dataset}."
            )

        first_half_means, second_half_means = [], []
        for user_id, group in df.groupby(user_id_col):
            group = group.sort_values(by=measurement_idx_col)

            if method == "odd_even":
                odd_mean = group[group[measurement_idx_col] % 2 == 1][construct].mean()
                even_mean = group[group[measurement_idx_col] % 2 == 0][construct].mean()
                first_half_means.append(odd_mean)
                second_half_means.append(even_mean)

            elif method == "individual_halves":
                half = len(group) // 2
                first_half_means.append(group.iloc[:half][construct].mean())
                second_half_means.append(group.iloc[half:][construct].mean())

            else:
                raise ValueError(
                    f"Unknown method '{method}'. Use 'odd_even' or 'individual_halves'."
                )

        first_half_series, second_half_series = pd.Series(first_half_means), pd.Series(
            second_half_means
        )
        valid_idx = (~first_half_series.isna()) & (~second_half_series.isna())
        first_half_series, second_half_series = (
            first_half_series[valid_idx],
            second_half_series[valid_idx],
        )

        if len(first_half_series) < 2:
            print(f"Not enough data to compute correlation for dataset '{dataset}'.")
            return np.nan

        correlation = first_half_series.corr(
            second_half_series, method=correlation_method
        )
        reliability = (
            2 * correlation / (1 + correlation) if correlation is not None else np.nan
        )

        return reliability

    def create_wb_items_table(
        self,
        dataset: str,
        m_sd_df: pd.DataFrame,
        decimals: int,
        store: bool,
        base_filename: str,
        icc1: pd.Series = None,
        icc2: pd.Series = None,
        rel: pd.Series = None,
        bp_corr: pd.DataFrame = None,
        wp_corr: pd.DataFrame = None,
        trait_corr: pd.DataFrame = None,
    ) -> None:
        """
        Creates a classical item descriptives table in APA style.

        Args:
            dataset: Dataset name for saving the table.
            m_sd_df: DataFrame with M (SD) statistics indexed by item names.
            decimals: Number of decimals to round in the item table.
            store: If true, table is stored
            base_filename: Base filename for saving the table that is equal for all datasets.
            icc1: Optional Series of ICC1 values indexed by item names.
            icc2: Optional Series of ICC2 values indexed by item names.
            rel: Optional Series of reliability values indexed by item names.
            bp_corr: Optional DataFrame of between-person correlations.
            wp_corr: Optional DataFrame of within-person correlations.
            trait_corr: Optional DataFrame of trait correlations.
        """
        unified_index = m_sd_df.index
        table = m_sd_df.reindex(unified_index)

        for col, name in zip([rel, icc1, icc2], ["Rel", "ICC1", "ICC2"]):
            if col is not None:
                table = table.join(col.reindex(unified_index).rename(name), how="left")

        # Combine state correlation matrices
        if bp_corr is not None or wp_corr is not None:
            corr_values = (
                (bp_corr if bp_corr is not None else wp_corr)
                .reindex(index=unified_index, columns=unified_index)
                .values
            )

            if bp_corr is not None and wp_corr is not None:
                lower_mask = np.tril(np.ones(corr_values.shape, dtype=bool), k=-1)
                corr_values[lower_mask] = wp_corr.reindex(
                    index=unified_index, columns=unified_index
                ).values[lower_mask]

            combined_corr = pd.DataFrame(
                corr_values, index=unified_index, columns=unified_index
            )
            table = table.join(combined_corr, how="left", rsuffix="_corr")

        # Add trait correlations (upper triangular only)
        if trait_corr is not None:
            trait_corr = trait_corr.reindex(index=unified_index, columns=unified_index)
            trait_corr.values[np.tril_indices(len(unified_index), k=-1)] = np.nan
            table = table.join(trait_corr, how="left", rsuffix="_trait_corr")

        # Drop empty columns and format table
        table = format_df(
            df=table.dropna(axis=1, how="all"), capitalize=False, decimals=decimals
        )
        excluded_cols = ["M", "SD"]
        for col in table.columns:
            if col not in excluded_cols:
                table[col] = table[col].astype(str).apply(remove_leading_zero)

        if store:
            file_path = os.path.join(
                self.desc_results_base_path, f"{dataset}_{base_filename}"
            )
            self.data_saver.save_excel(table, file_path)

    def analyze_country_level_vars(self, decimals: int = 2) -> pd.DataFrame:
        """
        Summarizes country-level variables from the full dataset.

        This method selects all columns prefixed with `mac_` and computes descriptive
        statistics across the full dataset and the CoCoESM subset. For each variable, it reports:
        - Number of unique values
        - Minimum value
        - Maximum value

        Args:
            decimals: Number of decimals to round in the df. Defaults to 2.

        Returns:
            pd.DataFrame: A summary table with a MultiIndex column header.
                          The first level indicates the dataset ("all" or "cocoesm"),
                          and the second level indicates the statistic ("num_unique", "min", "max").
        """
        mac_data_cols = [col for col in self.full_data.columns if col.startswith("mac_")]
        df_all = self.full_data[mac_data_cols].copy()
        df_cocoesm = df_all.loc[self.full_data.index.str.startswith("cocoesm")]

        # Summary helper function
        def summarize(df):
            summary = {}
            for col in df.columns:
                non_na = df[col].dropna()
                summary[col] = {
                    "num_unique": non_na.nunique(),
                    "min": non_na.min(),
                    "max": non_na.max()
                }
            return pd.DataFrame(summary).T  # Transpose to have variables as rows

        summary_all = summarize(df_all)
        summary_cocoesm = summarize(df_cocoesm)

        combined = pd.concat(
            [summary_all, summary_cocoesm],
            axis=1,
            keys=["all", "cocoesm"]
        )
        combined = combined.round(decimals)
        combined.index = apply_name_mapping(list(combined.index), name_mapping=self.name_mapping, prefix=True)

        return combined

    @staticmethod
    def create_age_gender_descriptives(dataset: str, data: pd.DataFrame,
                                       decimals = 2) -> dict[str, float]:
        """
        Computes mean and standard deviation of age, and gender proportions for a given dataset.

        Filters the provided DataFrame to rows where the index begins with the given dataset identifier.
        Then calculates:
        - Mean and standard deviation of the 'pl_age' column
        - Proportion of entries with gender indicated as male, female, or other

        Args:
            dataset: The dataset identifier to filter rows by matching the beginning of the index.
            data: The complete DataFrame containing participant data with age and gender information.
            decimals: The number of decimal places the results are rounded. Defaults to 2.

        Returns:
            dict[str, float]: A dictionary containing the mean age, standard deviation of age, and gender proportions
                              with keys 'mean_age', 'std_age', 'prop_male', 'prop_female', 'prop_other'.
        """
        subset = data[data.index.str.startswith(dataset)]
        mean_age = np.round(subset["pl_age"].mean(), decimals)
        std_age = np.round(subset["pl_age"].std(), decimals)

        # Gender logic
        gcols = ["pl_gender_male", "pl_gender_female", "pl_gender_other"]
        genders = subset[gcols].fillna(0).astype(float)
        valid_rows = genders.sum(axis=1) == 1
        valid = genders[valid_rows]

        total = int(valid.shape[0])

        if total == 0:
            props = {"male": 0.0, "female": 0.0, "other": 0.0}

        else:
            sums = valid.sum()  # each column sum = number of people in that category
            props = (sums / total * 100).round(decimals)
            props = {
                "male": float(props["pl_gender_male"]),
                "female": float(props["pl_gender_female"]),
                "other": float(props["pl_gender_other"]),
            }

        return {
            "mean_age": float(mean_age),
            "std_age": float(std_age),
            "prop_male": props["male"],
            "prop_female": props["female"],
            "prop_other": props["other"],
        }
