import os
import re
from typing import Union

import pandas as pd

from src.postprocessing.CVResultProcessor import CVResultProcessor
from src.postprocessing.DescriptiveStatistics import DescriptiveStatistics
from src.postprocessing.InvarianceTester import InvarianceTester
from src.postprocessing.LinearRegressor import LinearRegressor
from src.postprocessing.ResultPlotter import ResultPlotter
from src.postprocessing.SampleMissingsAnalyzer import SampleMissingsAnalyzer
from src.postprocessing.ShapProcessor import ShapProcessor
from src.postprocessing.SignificanceTesting import SignificanceTesting
from src.postprocessing.SuppFileCreator import SuppFileCreator
from src.utils.DataLoader import DataLoader
from src.utils.DataSaver import DataSaver
from src.utils.Logger import Logger
from src.utils.SanityChecker import SanityChecker
from src.utils.utilfuncs import merge_M_SD_in_dct, NestedDict, create_defaultdict


class Postprocessor:
    """
    Executes postprocessing steps for analyzing machine learning results and generating insights.

    Responsibilities include:
    - Conducting tests of significance to compare prediction results.
    - Calculating and displaying descriptive statistics.
    - Generating plots for results, SHAP values, SHAP interaction values and pred vs. true parity.
    - Preparing supplemental files for reports

    Attributes:
        cfg_preprocessing (NestedDict): Yaml config specifying details on preprocessing (e.g., scales, items).
        cfg_analysis (NestedDict): Yaml config specifying details on the ML analysis (e.g., CV, models).
        cfg_postprocessing (NestedDict): Yaml config specifying details on postprocessing (e.g., tables, plots).
        name_mapping (NestedDict): Mapping of feature names for presentation purposes.
        data_base_path: Path to the preprocessed data directory.
        base_result_path: Path to the base directory for storing results.
        cv_shap_results_path: Path to the main result directory containing cross-validation results and SHAP values.
        methods_to_apply: List of postprocessing methods to apply.
        datasets: List of datasets included in the analysis.
        meta_vars: List of meta variables to exclude from SHAP processing.
        data_loader: Instance of `DataLoader` for loading data.
        data_saver: Instance of `DataSaver` for saving data.
        raw_results_filenames: Filenames for raw analysis output.
        processed_results_filenames: Filenames for processed postprocessing results.
        full_data: The full dataset loaded from preprocessed data.
        logger: Instance of `Logger` for logging.
        descriptives_creator: Instance of `DescriptiveStatistics` for creating descriptive statistics.
        cv_result_processor: Instance of `CVResultProcessor` for processing cross-validation results.
        significance_testing: Instance of `SignificanceTesting` for conducting statistical tests.
        shap_processor: Instance of `ShapProcessor` for processing SHAP values.
        plotter: Instance of `ResultPlotter` for generating plots.
        supp_file_creator: Instance of `SuppFileCreator` for creating supplemental files.
        sanity_checker: Instance of `SanityChecker` for validating preprocessing and results.
    """

    def __init__(
        self,
        cfg_preprocessing: NestedDict,
        cfg_analysis: NestedDict,
        cfg_postprocessing: NestedDict,
        name_mapping: NestedDict,
    ):
        """
        Initializes the Postprocessor with configuration settings, paths, and analysis components.

        Args:
            cfg_preprocessing: Configuration dictionary for preprocessing settings,
                               including paths to data and logs.
            cfg_analysis: Configuration dictionary for analysis settings, such as cross-validation
                          and imputation parameters.
            cfg_postprocessing: Configuration dictionary for postprocessing settings,
                                including result paths, methods to apply, and filenames.
            name_mapping: Dictionary mapping feature combination keys to human-readable names.
        """
        self.cfg_preprocessing = cfg_preprocessing
        self.cfg_analysis = cfg_analysis
        self.cfg_postprocessing = cfg_postprocessing
        self.name_mapping = name_mapping

        self.data_base_path = self.cfg_preprocessing["general"][
            "path_to_preprocessed_data"
        ]

        result_paths_cfg = self.cfg_postprocessing["general"]["data_paths"]
        self.base_result_path = result_paths_cfg["base_path"]
        self.cv_shap_results_path = os.path.join(
            self.base_result_path, result_paths_cfg["main_results"]
        )

        self.methods_to_apply = self.cfg_postprocessing["methods_to_apply"]
        self.datasets = self.cfg_preprocessing["general"]["datasets_to_be_included"]
        self.meta_vars = [
            self.cfg_analysis["cv"]["id_grouping_col"],
            self.cfg_analysis["imputation"]["country_grouping_col"],
            self.cfg_analysis["imputation"]["years_col"],
        ]

        self.data_loader = DataLoader()
        self.data_saver = DataSaver()
        self.raw_results_filenames = self.cfg_analysis["output_filenames"]
        self.processed_results_filenames = self.cfg_postprocessing["general"][
            "processed_filenames"
        ]

        self.full_data = self.data_loader.read_pkl(
            os.path.join(
                self.data_base_path,
                self.cfg_preprocessing["general"]["full_data_filename"],
            )
        )

        self.logger = Logger(
            log_dir=self.cfg_preprocessing["general"]["log_dir"],
            log_file=self.cfg_preprocessing["general"]["log_name"],
        )

        self.descriptives_creator = DescriptiveStatistics(
            cfg_preprocessing=self.cfg_preprocessing,
            cfg_analysis=self.cfg_analysis,
            cfg_postprocessing=self.cfg_postprocessing,
            name_mapping=name_mapping,
            base_result_path=self.base_result_path,
            full_data=self.full_data,
        )

        self.sample_missings_analyzer = SampleMissingsAnalyzer(
            full_data=self.full_data,
            data_saver=self.data_saver,
            name_mapping=self.name_mapping,
            cfg_postprocessing=self.cfg_postprocessing,
        )

        self.cv_result_processor = CVResultProcessor(
            cfg_postprocessing=self.cfg_postprocessing,
        )

        self.significance_testing = SignificanceTesting(
            base_result_path=self.base_result_path,
            cfg_postprocessing=self.cfg_postprocessing,
        )

        self.shap_processor = ShapProcessor(
            cfg_postprocessing=self.cfg_postprocessing,
            base_result_path=self.base_result_path,
            cv_shap_results_path=self.cv_shap_results_path,
            processed_results_filenames=self.processed_results_filenames,
            name_mapping=self.name_mapping,
            meta_vars=self.meta_vars,
        )

        self.plotter = ResultPlotter(
            cfg_postprocessing=self.cfg_postprocessing,
            base_result_path=self.base_result_path,
        )

        self.supp_file_creator = SuppFileCreator(
            cfg_postprocessing=self.cfg_postprocessing,
            name_mapping=self.name_mapping,
            meta_vars=self.meta_vars,
        )

        self.sanity_checker = SanityChecker(
            logger=self.logger,
            cfg_preprocessing=self.cfg_preprocessing,
            cfg_postprocessing=self.cfg_postprocessing,
            plotter=self.plotter,
        )
        self.invariance_tester = InvarianceTester(
            cfg_preprocessing=self.cfg_preprocessing,
            cfg_postprocessing=self.cfg_postprocessing,
            name_mapping=self.name_mapping,
        )

    def apply_methods(self) -> None:
        """
        Executes the methods specified in the cfg dynamically.

        Raises:
            ValueError: If a specified method does not exist in the class.
        """
        for method_name in self.cfg_postprocessing["methods_to_apply"]:
            if not hasattr(self, method_name):
                raise ValueError(f"Method '{method_name}' is not implemented.")

            print(f">>>Executing postprocessing method: {method_name}<<<")
            getattr(self, method_name)()

    def condense_cv_results(self) -> None:
        """
        Summarizes the CV results for all analysis and stores the results in tables.

        Methodology
        - Loads the specific cv_results file for each analysis from its directories
        - Stores all cv_results in one json file
        - Computes the mean and standard deviation of the differences between the main and the nnse analyses
        - Computes the percentage increase in performance for given feature combinations
        - Creates summary tables in Excel
        """
        # Get all results
        cv_results_filename = self.cfg_postprocessing["general"]["processed_filenames"][
            "cv_results_summarized"
        ]
        cfg_condense_results = self.cfg_postprocessing["condense_cv_results"]

        cv_results_dct = self.cv_result_processor.extract_cv_results(
            base_dir=self.cv_shap_results_path,
            metrics=cfg_condense_results["metrics"],
            cv_results_filename=cv_results_filename,
            negate_mse=cfg_condense_results["negate_mse"],
            decimals=cfg_condense_results["decimals"],
            add_other_sds=cfg_condense_results["add_other_sds"],
        )
        if cfg_condense_results["all_results"]["store"]:
            all_result_path = os.path.join(
                self.cv_shap_results_path,
                cfg_condense_results["all_results"]["filename"],
            )
            self.data_saver.save_json(cv_results_dct, all_result_path)

        # Compute mean and standard deviation of the differences between main and nnse analyses
        if cfg_condense_results["main_nnse_diffs"]["run"]:
            main_nnse_diff_cfg = cfg_condense_results["main_nnse_diffs"]
            main_nnse_diff_dct = create_defaultdict(3, dict)
            for crit in main_nnse_diff_cfg["crits"]:
                for samples_to_include in main_nnse_diff_cfg["samples_to_include"]:
                    for metric in main_nnse_diff_cfg["metrics"]:
                        (
                            mean_diff,
                            std_diff,
                        ) = self.cv_result_processor.calculate_main_nnse_diff(
                            cv_results=cv_results_dct[crit][samples_to_include],
                            metric=metric,
                        )
                        main_nnse_diff_dct[crit][samples_to_include][metric] = {
                            "M": mean_diff,
                            "SD": std_diff,
                        }
            if cfg_condense_results["main_nnse_diffs"]["store"]:
                nnse_diff_path = os.path.join(
                    self.cv_shap_results_path,
                    cfg_condense_results["main_nnse_diffs"]["filename"],
                )
                self.data_saver.save_json(main_nnse_diff_dct, nnse_diff_path)

        if cfg_condense_results["perf_increase"]["calculate"]:

            perf_increase_cfg = cfg_condense_results["perf_increase"]
            perf_increase_dct = create_defaultdict(5, dict)

            for crit in perf_increase_cfg["crits"]:
                for samples_to_include in perf_increase_cfg["samples_to_include"]:
                    for metric in perf_increase_cfg["metrics"]:
                        for model in perf_increase_cfg["models"]:
                            for fc_to_compare in perf_increase_cfg["fc_to_compare"]:
                                perc_increase = (
                                    self.cv_result_processor.calculate_perc_increase(
                                        cv_results=cv_results_dct[crit][
                                            samples_to_include
                                        ],
                                        metric=metric,
                                        model=model,
                                        feature_combos_to_compare=fc_to_compare,
                                        decimals=perf_increase_cfg["decimals"],
                                    )
                                )
                                fc_to_compare_str = ", ".join(fc_to_compare)
                                perf_increase_dct[crit][samples_to_include][metric][
                                    model
                                ][fc_to_compare_str] = {"% increase": perc_increase}
            if perf_increase_cfg["store"]:
                perf_inc_path = os.path.join(
                    self.cv_shap_results_path, perf_increase_cfg["filename"]
                )
                self.data_saver.save_json(perf_increase_dct, perf_inc_path)

        # Creates Excel tables
        for crit, crit_vals in cv_results_dct.items():
            for samples_to_include, samples_to_include_vals in crit_vals.items():
                for analysis in ["main", "nnse", "npers", "altimp"]:
                    if analysis and samples_to_include == "control":
                        continue

                    data_for_table = merge_M_SD_in_dct(samples_to_include_vals)

                    self.cv_result_processor.create_cv_results_table(
                        crit=crit,
                        samples_to_include=samples_to_include,
                        data=data_for_table,
                        output_dir=self.cv_shap_results_path,
                        analysis=analysis,
                        include_empty_col_between_models=True,
                    )

    def prediction_sanity_checks(self) -> None:
        """Sanity checks the predictions vs. the true values for selected analysis and plots the results."""
        full_data = self.data_loader.read_pkl(
            os.path.join(
                self.cfg_preprocessing["general"]["path_to_preprocessed_data"],
                self.cfg_preprocessing["general"]["full_data_filename"],
            )
        )
        self.sanity_checker.sanity_check_mac(full_data)
        self.sanity_checker.sanity_check_pred_vs_true(
            full_data, groupby="country_group"
        )

    def create_descriptives(self) -> None:
        """Creates tables containing descriptives (e.g., M, SD, correlations, reliability) for the datasets."""
        desc_cfg = self.cfg_postprocessing["create_descriptives"]
        var_table_cfg = desc_cfg["m_sd_table"]

        country_stats = self.descriptives_creator.analyze_country_level_vars()
        if desc_cfg["country_vars"]["store"]:
            file_path_country = os.path.join(
                self.descriptives_creator.desc_results_base_path,
                desc_cfg["country_vars"]["filename"],
            )
            self.data_saver.save_excel(country_stats, file_path_country)

        self.descriptives_creator.create_m_sd_var_table(
            vars_to_include=var_table_cfg["vars_to_include"],
            binary_stats_to_calc=var_table_cfg["bin_agg_lst"],
            continuous_stats_to_calc=var_table_cfg["cont_agg_dct"],
            table_decimals=var_table_cfg["decimals"],
            store_table=var_table_cfg["store"],
            filename=var_table_cfg["filename"],
            store_index=var_table_cfg["store_index"],
        )

        rel_dct = {}
        soc_dem_dct = {}

        # Dataset specific descriptives calculation (e.g., wb-outcomes, sociodemographcis)
        for dataset in self.datasets:
            soc_dem_dct[dataset] = (
                self.descriptives_creator.create_age_gender_descriptives(
                    dataset=dataset, data=self.full_data.copy()
                )
            )

            traits_base_filename = self.cfg_postprocessing["create_descriptives"][
                "traits_base_filename"
            ]
            path_to_trait_df = os.path.join(
                self.data_base_path, f"{traits_base_filename}_{dataset}"
            )
            trait_df = (
                self.data_loader.read_pkl(path_to_trait_df)
                if os.path.exists(path_to_trait_df)
                else None
            )

            states_base_filename = self.cfg_postprocessing["create_descriptives"][
                "states_base_filename"
            ]
            path_to_state_df = os.path.join(
                self.data_base_path, f"{states_base_filename}_{dataset}"
            )
            state_df = (
                self.data_loader.read_pkl(path_to_state_df)
                if os.path.exists(path_to_state_df)
                else None
            )
            esm_id_col = self.cfg_preprocessing["general"]["esm_id_col"][dataset]
            esm_tp_col = self.cfg_preprocessing["general"]["esm_timestamp_col"][dataset]

            rel_dct[dataset] = self.descriptives_creator.compute_rel(
                state_df=state_df,
                trait_df=trait_df,
                dataset=dataset,
                decimals=desc_cfg["rel"]["decimals"],
            )

            wb_items_dct = self.descriptives_creator.create_wb_items_stats_per_dataset(
                dataset=dataset,
                state_df=state_df,
                trait_df=trait_df,
                esm_id_col=esm_id_col,
                esm_tp_col=esm_tp_col,
            )

            self.descriptives_creator.create_wb_items_table(
                m_sd_df=wb_items_dct["m_sd"],
                decimals=desc_cfg["wb_items"]["decimals"],
                store=desc_cfg["wb_items"]["store"],
                base_filename=desc_cfg["wb_items"]["filename"],
                dataset=dataset,
                icc1=wb_items_dct["icc1"],
                bp_corr=wb_items_dct["bp_corr"],
                wp_corr=wb_items_dct["wp_corr"],
                trait_corr=wb_items_dct["trait_corr"],
            )

        if desc_cfg["soc_dem"]["store"]:
            file_path_socdem = os.path.join(
                self.descriptives_creator.desc_results_base_path,
                desc_cfg["soc_dem"]["filename"],
            )
            self.data_saver.save_json(soc_dem_dct, file_path_socdem)

        if desc_cfg["rel"]["store"]:
            file_path_rel = os.path.join(
                self.descriptives_creator.desc_results_base_path,
                desc_cfg["rel"]["filename"],
            )
            self.data_saver.save_json(rel_dct, file_path_rel)

    def conduct_significance_tests(self) -> None:
        """Conducts significance tests to compare models and compare predictor classes."""

        self.significance_testing.significance_testing()

    def test_measurement_invariance(self) -> None:
        """Tests measurement invariance across countries and datasets"""
        self.invariance_tester.load_data()

        mi_datasets = self.invariance_tester.test_invariance_across_datasets()
        self._save_nested_results(mi_datasets, "../results/mi_across_datasets")

        mi_countries = self.invariance_tester.test_invariance_across_countries()
        self._save_nested_results(mi_countries, "../results/mi_across_countries")

    def analyze_samples_missings(self) -> None:
        """Analyzes differences between missings and non-missings and between datasets."""

        self.sample_missings_analyzer.compare_missings(  # This takes some time
            method="pairwise"
        )
        self.sample_missings_analyzer.compare_samples()

    def create_cv_results_plots(self) -> None:
        """Creates a bar plot summarizing CV results for the analyses specified."""

        all_results_file_path = os.path.join(
            self.cv_shap_results_path,
            self.cfg_postprocessing["condense_cv_results"]["all_results"]["filename"],
        )
        all_cv_results_dct = self.data_loader.read_json(all_results_file_path)

        if all_cv_results_dct:
            self.plotter.plot_cv_results_plots_wrapper(
                cv_results_dct=all_cv_results_dct,
                rel=None,
            )

    def create_shap_plots(self) -> None:
        """Creates SHAP beeswarm plots for all analyses specified in the cfg."""

        self.plotter.plot_shap_beeswarm_plots(
            prepare_shap_data_func=self.shap_processor.prepare_shap_data,
            prepare_shap_ia_data_func=self.shap_processor.prepare_shap_ia_data,
        )

    def calculate_exp_lin_models(self) -> None:
        """Calculates explanatory linear models with the x best features for selected analyses."""

        linear_regressor_cfg = self.cfg_postprocessing["calculate_exp_lin_models"]

        for feature_combination in self.cfg_postprocessing["general"][
            "feature_combinations"
        ]["name_mapping"]["main"].keys():
            for samples_to_include in linear_regressor_cfg["samples_to_include"]:
                for crit in linear_regressor_cfg["crits"]:
                    for model_for_features in linear_regressor_cfg[
                        "model_for_features"
                    ]:
                        linear_regressor = LinearRegressor(
                            cfg_preprocessing=self.cfg_preprocessing,
                            cfg_analysis=self.cfg_analysis,
                            cfg_postprocessing=self.cfg_postprocessing,
                            name_mapping=self.name_mapping,
                            cv_shap_results_path=self.cv_shap_results_path,
                            df=self.full_data.copy(),
                            feature_combination=feature_combination,
                            crit=crit,
                            samples_to_include=samples_to_include,
                            model_for_features=model_for_features,
                            meta_vars=self.meta_vars,
                        )

                        linear_regressor.get_regression_data()
                        lin_model = linear_regressor.compute_regression_models()

                        linear_regressor.create_coefficients_table(
                            model=lin_model,
                            feature_combination=feature_combination,
                            output_dir=self.cv_shap_results_path,
                        )

    def create_lin_model_coefs_supp(self) -> None:
        """Creates a new dir containing JSON files with the coefficients of the linear models for each analysis."""

        filename = self.processed_results_filenames["lin_model_coefs_summarized"]
        output_dir = os.path.join(
            self.base_result_path,
            self.cfg_postprocessing["create_supp_files"]["lin_coefs_output_filename"],
        )

        self.supp_file_creator.create_mirrored_dir_with_files(
            base_dir=self.cv_shap_results_path,
            file_name=filename,
            output_base_dir=output_dir,
        )

    def create_shap_values_supp(self) -> None:
        """Creates a new dir containing JSON files with the shap_values for each analysis."""

        filename = self.processed_results_filenames["shap_values_summarized"]
        output_dir = os.path.join(
            self.base_result_path,
            self.cfg_postprocessing["create_supp_files"]["shap_output_filename"],
        )

        self.supp_file_creator.create_mirrored_dir_with_files(
            base_dir=self.cv_shap_results_path,
            file_name=filename,
            output_base_dir=output_dir,
        )

    def create_shap_ia_values_supp(self) -> None:
        """Creates a new dir containing JSON files with the shap_ia_values for some selected analysis."""

        filename = self.processed_results_filenames["shap_ia_values_summarized"]
        output_dir = os.path.join(
            self.base_result_path,
            self.cfg_postprocessing["create_supp_files"]["shap_ia_output_filename"],
        )

        input_dir = os.path.join(
            self.base_result_path,
            self.cfg_postprocessing["general"]["data_paths"]["ia_values"],
        )
        self.supp_file_creator.create_mirrored_dir_with_files(
            base_dir=input_dir,
            file_name=filename,
            output_base_dir=output_dir,
        )

    def _save_nested_results(  # keep name for compatibility
        self,
        results: dict[str, Union[pd.DataFrame, dict]],
        base_result_dir: str | os.PathLike,
        *,
        flatten: bool = True,  # True → filenames; False → folders
        file_ext: str = "xlsx",  # "xlsx", "csv", …
        sep: str = "__",  # joiner for flat filenames
    ) -> None:
        """
        Saves a nested dictionary of DataFrames to disk in either flat or hierarchical structure.

        This method recursively traverses a dictionary whose leaves are pandas DataFrames and writes them
        to `base_result_dir` using the specified file extension. The output can be structured as either:
        - Flat: files named using joined keys (e.g., "group__construct.xlsx")
        - Nested: directories reflecting the dictionary hierarchy

        Args:
            results: A nested dictionary where leaves are pd.DataFrame objects.
            base_result_dir: Base directory to save files or folders into.
            flatten: If True, all files are saved in the base directory with concatenated filenames.
                     If False, the dictionary structure is mirrored as subdirectories.
            file_ext: File format to write ("xlsx" or "csv").
            sep: Separator used to join keys in filenames when `flatten` is True.
        """

        base_dir = os.fspath(base_result_dir)
        os.makedirs(base_dir, exist_ok=True)

        def _safe(key: str) -> str:
            """
            Converts a string into a filesystem-safe format by replacing unsafe characters.

            Replaces all characters except letters, digits, underscores, hyphens, and periods
            with underscores to ensure compatibility with file and directory names.

            Args:
                key: The string to sanitize for use in paths.

            Returns:
                str: A cleaned string safe for use as a filename or folder name.
            """
            return re.sub(r"[^\w\-\.]", "_", str(key))

        def _write(df: pd.DataFrame, path: str) -> None:
            """
            Writes a DataFrame to disk using the specified file format.

            Supports "xlsx" and "csv" formats. Delegates the actual writing to the appropriate
            method on `self.data_saver`.

            Args:
                df: The DataFrame to save.
                path: Full file path (without extension stripping).

            Returns:
                None
            """
            if file_ext == "xlsx":
                self.data_saver.save_excel(df, path)
            elif file_ext == "csv":
                self.data_saver.save_csv(df, path)
            else:
                raise ValueError(
                    f"Unsupported file_ext '{file_ext}' for pd.DataFrame as input. "
                    "Add a writer for it in _save_nested_results()."
                )

        def _walk_flat(subtree: dict, key_chain: list[str]) -> None:
            """
            Recursively traverses a nested dictionary and writes files using flat naming.

            Keys are joined with the configured separator to form filenames. All files are saved
            directly in the base result directory.

            Args:
                subtree: Current level of the nested dictionary.
                key_chain: List of keys traversed so far, used to build the filename.
            """
            for key, val in subtree.items():
                new_chain = key_chain + [_safe(key)]
                if isinstance(val, pd.DataFrame):
                    filename = sep.join(new_chain) + f".{file_ext.lstrip('.')}"
                    _write(val, os.path.join(base_dir, filename))
                elif isinstance(val, dict):
                    _walk_flat(val, new_chain)
                else:
                    raise TypeError(
                        f"Unsupported type at {'/'.join(new_chain)}: "
                        f"{type(val).__name__}"
                    )

        def _walk_nested(subtree: dict, current_path: str) -> None:
            """
            Recursively traverses a nested dictionary and saves files in corresponding subfolders.

            Creates subdirectories reflecting the dictionary structure and writes each DataFrame
            into its appropriate folder.

            Args:
                subtree: Current level of the nested dictionary.
                current_path: Path where files or subfolders should be written.
            """
            for key, val in subtree.items():
                safe_key = _safe(key)
                next_path = os.path.join(current_path, safe_key)
                if isinstance(val, pd.DataFrame):
                    os.makedirs(current_path, exist_ok=True)
                    _write(val, next_path + f".{file_ext.lstrip('.')}")
                elif isinstance(val, dict):
                    os.makedirs(next_path, exist_ok=True)
                    _walk_nested(val, next_path)
                else:
                    raise TypeError(
                        f"Unsupported type at {next_path}: {type(val).__name__}"
                    )

        # start traversal
        if flatten:
            _walk_flat(results, [])
        else:
            _walk_nested(results, base_dir)
