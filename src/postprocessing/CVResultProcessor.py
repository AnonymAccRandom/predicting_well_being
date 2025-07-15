import os

import numpy as np
import pandas as pd

from src.utils.DataLoader import DataLoader
from src.utils.DataSaver import DataSaver
from src.utils.utilfuncs import (
    defaultdict_to_dict,
    create_defaultdict,
    rearrange_path_parts,
    NestedDict,
)


class CVResultProcessor:
    """
    Processes cross-validation (CV) results for machine learning analysis.

    This class extracts, formats, and saves CV results into structured tables, suitable for reporting and analysis.
    The tables include:
        - Mean (M) and Standard Deviation (SD) of metrics for different models and feature combinations.
        - MultiIndex structures to organize results by models and metrics.
        - Configurable formatting, including feature combination mapping and visual separation between models.

    Attributes:
        cfg_postprocessing (dict): Configuration dictionary for postprocessing, defining mappings and parameters.
        data_loader (DataLoader): Utility for loading data (e.g., JSON, pickle files).
        data_saver (DataSaver): Utility for saving data in various formats.
        feature_combo_name_mapping (dict): Combined mapping of feature combinations for main and supplementary analyses.
        samples_to_include_name_mapping (dict): Mapping of sample subsets (e.g., "all", "selected").
        crit_name_mapping (dict): Mapping of criteria names.
        model_name_mapping (dict): Mapping of model names.
        metric_name_mapping (dict): Mapping of metric names.
    """

    def __init__(self, cfg_postprocessing: NestedDict) -> None:
        """
        Initializes the CVResultProcessor with configuration data.

        Args:
            cfg_postprocessing: Dictionary containing postprocessing configurations, including mappings for feature
                                combinations, models, metrics, and result table settings.
        """
        self.cfg_postprocessing = cfg_postprocessing

        self.data_loader = DataLoader()
        self.data_saver = DataSaver()

        self.feature_combo_name_mapping_main = self.cfg_postprocessing["general"][
            "feature_combinations"
        ]["name_mapping"]["main"]
        self.feature_combo_name_mapping_supp = self.cfg_postprocessing["general"][
            "feature_combinations"
        ]["name_mapping"]["supp"]
        self.feature_combo_name_mapping = {
            **self.feature_combo_name_mapping_main,
            **self.feature_combo_name_mapping_supp,
        }
        self.samples_to_include_name_mapping = self.cfg_postprocessing["general"][
            "samples_to_include"
        ]["name_mapping"]
        self.crit_name_mapping = self.cfg_postprocessing["general"]["crits"][
            "name_mapping"
        ]
        self.model_name_mapping = self.cfg_postprocessing["general"]["models"][
            "name_mapping"
        ]
        self.metric_name_mapping = self.cfg_postprocessing["general"]["metrics"][
            "name_mapping"
        ]

        self.result_table_cfg = self.cfg_postprocessing["condense_cv_results"][
            "result_table"
        ]
        self.table_cat_name_mapping = self.result_table_cfg["mapping"]

    @property
    def cat_mapping(self) -> dict[str, list[str]]:
        """
        Creates a dictionary mapping analysis parameters to their possible values.

        The categories include:
            - "feature_combination": Feature combinations from main and supplementary mappings.
            - "samples_to_include": Sample subsets (e.g., "all", "selected").
            - "crit": Criteria names (e.g., "wb_state")
            - "model": Model names (e.g., "randomforestregressor")

        Returns:
            dict: A dictionary where keys are category names (e.g., "feature_combination") and values are lists of
                  possible values for each category.
        """
        return {
            "feature_combination": list(self.feature_combo_name_mapping.keys()),
            "samples_to_include": list(self.samples_to_include_name_mapping.keys()),
            "crit": list(self.crit_name_mapping.keys()),
            "model": list(self.model_name_mapping.keys()),
        }

    def extract_cv_results(
        self,
        base_dir: str,
        metrics: list[str],
        cv_results_filename: str,
        negate_mse: bool = True,
        decimals: int = 3,
        add_other_sds: bool = False,
    ) -> NestedDict:
        """
        Extracts required metrics from `proc_cv_results.json` files within a nested directory structure.

        This method traverses the directory tree starting from `base_dir`, locates JSON files matching
        the given filename, and extracts summary statistics (mean and standard deviation) for the specified metrics.

        - Adjusts for negative MSE values when `negate_mse` is True.
        - Organizes results into a nested dictionary based on criteria, samples, feature combinations, and models.

        Args:
            base_dir: Base directory to search for CV results.
            metrics: List of metric names to extract (e.g., "neg_mean_squared_error").
            cv_results_filename: Name of the JSON file to read in each directory.
            negate_mse: If True, converts negative MSE values to positive. Defaults to True.
            decimals: Number of decimal places to round the extracted metrics. Defaults to 3.
            add_other_sds: If we add SDs across folds within imps / within folds across imps to the result dict.

        Returns:
            dict: A nested dictionary containing extracted metrics with the structure:
                  {crit: {samples_to_include: {feature_combination: {model: {metric: {"M": float, "SD": float}}}}}}.
        """
        result_dct = create_defaultdict(n_nesting=5, default_factory=dict)

        for root, _, files in os.walk(base_dir):
            if cv_results_filename in files:
                print(root)
                # These analyses are not included in the tables
                if "pl_srmc_control" in root or "srmc_control" in root:
                    continue

                (
                    rearranged_key,
                    crit,
                    samples_to_include,
                    feature_combination,
                    model,
                ) = rearrange_path_parts(
                    root,
                    base_dir,
                    min_depth=4,
                    order_mapping={
                        "crit": 1,
                        "samples_to_include": 2,
                        "feature_combination": 3,
                        "model": 4,
                    },
                    cat_values=self.cat_mapping,
                )

                cv_results_summary = self.data_loader.read_json(
                    os.path.join(root, cv_results_filename)
                )

                for metric in metrics:
                    m_metric = cv_results_summary["m"][metric]
                    sd_metric = cv_results_summary["sd_across_folds_imps"][metric]

                    # Correct MSE values if required, as these are negative values in scikit-learn
                    if negate_mse and metric == "neg_mean_squared_error":
                        m_metric *= -1

                    result_stats = {
                        "M": f"{m_metric:.{decimals}f}",
                        "SD": f"{sd_metric:.{decimals}f}",
                    }

                    if add_other_sds:
                        result_stats.update(
                            {
                                "SD_across_folds_within_imps": f"{cv_results_summary['sd_across_folds_within_imps'][metric]:.{decimals}f}",
                                "SD_within_folds_across_imps": f"{cv_results_summary['sd_within_folds_across_imps'][metric]:.{decimals}f}",
                            }
                        )

                    result_dct[crit][samples_to_include][feature_combination][model][
                        metric
                    ] = result_stats

        return defaultdict_to_dict(result_dct)

    @staticmethod
    def calculate_main_nnse_diff(
        cv_results: NestedDict,
        metric: str,
        decimals: int = 3,
    ) -> tuple[float, float]:
        """
        Calculates the average delta for a given metric between the analysis with and without nnse.

        Args:
            cv_results: A nested dictionary containing cross-validation results.
            metric: The performance metric to compare (e.g., 'pearson', 'r2', 'spearman').
        Returns:
            tuple[float, float]: Tuple with the M and SD of the differences between the main and nnse analyses.
        """
        differences = []

        paired_keys = [
            (key, key + "_nnse") for key in cv_results if key + "_nnse" in cv_results
        ]

        for base_key, nnse_key in paired_keys:
            for model in cv_results[base_key]:
                if (
                    model in cv_results[nnse_key]
                    and metric in cv_results[base_key][model]
                ):
                    mean_base = cv_results[base_key][model][metric]["M"]
                    mean_nnse = cv_results[nnse_key][model][metric]["M"]

                    diff = float(mean_base) - float(mean_nnse)
                    differences.append(diff)

        mean_diff = np.round(np.mean(differences), decimals)
        std_diff = np.round(np.std(differences), decimals)

        return mean_diff, std_diff

    @staticmethod
    def calculate_perc_increase(
        cv_results: NestedDict,
        metric: str,
        model: str,
        feature_combos_to_compare: tuple[str, str],
        decimals: int = 3,
    ) -> float:
        """
        Computes the percentage increase in performance between two feature combinations.

        Args:
            cv_results: A nested dictionary containing cross-validation results.
            metric: The performance metric to compare (e.g., 'pearson', 'r2', 'spearman').
            model: The model to compare.
            feature_combos_to_compare: A tuple containing two keys to compare.
            decimals: Number of decimal places to round the result.

        Returns:
            float: The percentage increase in performance.
        """
        key1, key2 = feature_combos_to_compare

        if key1 not in cv_results or key2 not in cv_results:
            raise ValueError("Both feature keys must be present in cv_results.")

        if model in cv_results[key2] and metric in cv_results[key1][model]:
            mean_key1 = float(cv_results[key1][model][metric]["M"])
            mean_key2 = float(cv_results[key2][model][metric]["M"])

            min_val, max_val = sorted([mean_key1, mean_key2])
            ratio = max_val / min_val

            return np.round(ratio, decimals)

    def create_cv_results_table(
        self,
        data: NestedDict,
        crit: str,
        samples_to_include: str,
        output_dir: str,
        include_empty_col_between_models: bool = True,
        supp_analysis: bool = False,
    ) -> None:
        """
        Generates and saves a cross-validation (CV) results table as an Excel file.

        The table organizes results by:
            - Rows: Feature combinations.
            - Columns: A MultiIndex with:
                - Level 1: Models (e.g., ENR, RFR).
                - Level 2: Metrics (e.g., Pearson's r, R², Spearman's rho, MSE).
                - Level 3: Statistics (e.g., Mean (M), Standard Deviation (SD)).

        The function filters and maps feature combinations, aggregates statistics, and ensures
        a consistent structure for display.

        Args:
            data: Nested dictionary of CV results with the structure:
                  {feature_combination: {model: {metric: {'M': float, 'SD': float}}}}.
            crit: Criterion used for the table (e.g., "wb_state").
            samples_to_include: Subset of samples (e.g., "all").
            output_dir: Directory for saving the Excel file.
            include_empty_col_between_models: Whether to include an empty column between models for visual separation.
            supp_analysis: Whether to include only the supplementary (nnse) analysis. Defaults to False.
                This may be the analysis without personality or without neuroticism and self-esteem
        """
        if supp_analysis:
            feature_combo_mapping = self.feature_combo_name_mapping_supp
            result_str = self.result_table_cfg["result_strs"]["supp"]
        else:
            feature_combo_mapping = self.feature_combo_name_mapping_main
            result_str = self.result_table_cfg["result_strs"]["main"]

        rows = [
            {
                self.table_cat_name_mapping[
                    "feature_combination"
                ]: feature_combo_mapping[feature_combination],
                self.table_cat_name_mapping["model"]: self.model_name_mapping.get(
                    model, model
                ),
                self.table_cat_name_mapping["metric"]: self.metric_name_mapping.get(
                    metric, metric
                ),
                "M (SD)": stats.get("M (SD)", "N/A"),
            }
            for feature_combination, models in data.items()
            if feature_combination in feature_combo_mapping
            for model, metrics in models.items()
            for metric, stats in metrics.items()
        ]

        df = pd.DataFrame(rows)
        df_pivot = df.pivot(
            index=self.table_cat_name_mapping["feature_combination"],
            columns=[
                self.table_cat_name_mapping["model"],
                self.table_cat_name_mapping["metric"],
            ],
            values="M (SD)",
        )

        # Custom order for metrics
        metric_order = list(self.metric_name_mapping.values())
        n_metrics = len(metric_order)
        #df_pivot.columns = pd.MultiIndex.from_tuples(
        #    sorted(
        #        df_pivot.columns,
        #        key=lambda col: (
        #            col[0],
        #            metric_order.index(col[1])
        #            if col[1] in metric_order
        #            else len(metric_order),
        #        ),
        #    )
        #)

        if include_empty_col_between_models:
            empty_col = pd.Series([np.nan] * len(df_pivot), name=(" ", " "))
            df_pivot = pd.concat(
                [df_pivot.iloc[:, :n_metrics], empty_col, df_pivot.iloc[:, n_metrics:]],
                axis=1,
            )

        custom_order = [feature_combo_mapping[k] for k in feature_combo_mapping]
        df_pivot = df_pivot.reindex(custom_order, fill_value="N/A")

        if self.result_table_cfg["store"]:
            output_path = os.path.join(
                output_dir,
                f"{self.result_table_cfg['file_base_name']}_{crit}_{samples_to_include}_{result_str}.xlsx",
            )
            self.data_saver.save_excel(df_pivot, output_path)

        print(f"Processed {crit}_{samples_to_include}_{result_str}")
