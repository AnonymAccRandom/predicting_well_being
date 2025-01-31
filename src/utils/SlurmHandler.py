import argparse
import os

from src.utils.utilfuncs import NestedDict


class SlurmHandler:
    """
    Handles computation tasks for machine learning analyses on clusters using the SLURM scheduler.

    This class provides utility methods to:
    - Parse SLURM-related arguments.
    - Update configuration files (`cfg_analysis`) with SLURM-specific parameters.
    - Allocate computing cores for various tasks.
    - Perform sanity checks for configurations before running cluster jobs.
    - Construct output paths for analysis results.
    """

    def get_slurm_vars(self) -> argparse.Namespace:
        """
        Parses SLURM-specific arguments (str, bool, int) from the command line or a slurm script.

        Returns:
            argparse.Namespace: Parsed SLURM arguments as a namespace object.
        """
        parser = argparse.ArgumentParser(description="CoCo WB ML - OH")

        parser.add_argument(
            "--prediction_model", type=str, help="elasticnet or randomforestregressor"
        )
        parser.add_argument("--crit", type=str, help="state or trait wb/pa/na")
        parser.add_argument(
            "--feature_combination",
            type=str,
            help="Feature combinations defined in the PreReg",
        )
        parser.add_argument(
            "--samples_to_include", type=str, help="all, selected, control"
        )
        parser.add_argument("--output_path", type=str, help="Output file path.")

        parser.add_argument(
            "--comp_shap_ia_values",
            type=self.str2bool,
            help="Calculate IA values for RFR, Bool",
        )
        parser.add_argument(
            "--parallelize_imputation_runs",
            type=self.str2bool,
            help="Parallelize imputed datasets, Bool",
        )
        parser.add_argument(
            "--parallelize_inner_cv",
            type=self.str2bool,
            help="Parallelize inner CV, Bool",
        )
        parser.add_argument(
            "--parallelize_shap_ia_values",
            type=self.str2bool,
            help="Parallelize SHAP IA values, Bool",
        )
        parser.add_argument(
            "--parallelize_shap",
            type=self.str2bool,
            help="Parallelize SHAP calculations, Bool",
        )
        parser.add_argument(
            "--split_reps",
            type=self.str2bool,
            help="Split repetitions into separate jobs, Bool",
        )

        # We decided not to use MPI for final analysis, so this is always false (defined in the cfg)
        # parser.add_argument("--use_mpi", type=self.str2bool, help="Use mpi4py, Bool")

        parser.add_argument(
            "--rep",
            type=int,
            default=None,
            help="Repetition number, used when split_reps is true",
        )

        args = parser.parse_args()

        return args

    @staticmethod
    def str2bool(v: str) -> bool:
        """
        Converts a string representation of truth to a boolean value.

        Valid true values: 'yes', 'true', 't', 'y', '1'.
        Valid false values: 'no', 'false', 'f', 'n', '0'.

        Args:
            v: A string representing a boolean value.

        Returns:
            bool: The boolean equivalent of the given input.

        Raises:
            ValueError: If the input string does not represent a valid boolean value.
        """
        if isinstance(v, bool):
            return v

        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True

        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False

        else:
            raise ValueError("Boolean value expected.")

    @staticmethod
    def update_cfg_with_slurm_vars(
        cfg_analysis: NestedDict, args: argparse.Namespace
    ) -> NestedDict:
        """
        Updates the current YAML configuration (`cfg_analysis`) with SLURM-provided arguments.

        It updates the following sections of the configuration with the corresponding variables:
        - cfg_analysis[arg_name]
        - cfg_analysis["params"][arg_name]
        - cfg_analysis["parallelize"][arg_name]
        - cfg_analysis["shap_ia_values"][arg_name]
        - cfg_analysis["output_path"]

        Args:
            cfg_analysis: A dictionary containing the old YAML configuration containing analysis params.
            args: The SLURM arguments parsed from the command line.

        Returns:
            NestedDict: The updated analysis YAML configuration with SLURM parameters.
        """
        args_dict = vars(args)
        print(args_dict)

        for arg_name, arg_value in args_dict.items():
            # Skip 'rep' as it's not part of cfg_analysis
            if arg_name == "rep" or arg_value is None:
                continue

            if arg_name in ["split_reps"]:
                cfg_analysis[arg_name] = arg_value

            elif arg_name in cfg_analysis["params"]:
                cfg_analysis["params"][arg_name] = arg_value

            elif arg_name in cfg_analysis["parallelize"]:
                cfg_analysis["parallelize"][arg_name] = arg_value

            elif arg_name in cfg_analysis["shap_ia_values"]:
                cfg_analysis["shap_ia_values"][arg_name] = arg_value

            elif arg_name == "output_path":
                cfg_analysis["output_path"] = arg_value

            else:
                print(
                    f"Warning: Argument {arg_name} not recognized. Skipping update for this argument."
                )

        return cfg_analysis

    @staticmethod
    def allocate_cores(cfg_analysis: NestedDict, total_cores: int) -> NestedDict:
        """
        Allocates available CPUs for various tasks based on the cfg, the SLURM configuration, and available cores.

        Args:
            cfg_analysis: Yaml config specifying details on the ML analysis (e.g., CV, models).
            total_cores: The total number of CPU cores available for the analysis.

        Returns:
            NestedDict: The updated YAML configuration with core allocations.
        """
        print("total_cores in func:", total_cores)

        if cfg_analysis["parallelize"]["parallelize_inner_cv"]:
            cfg_analysis["parallelize"]["inner_cv_n_jobs"] = total_cores
        else:
            cfg_analysis["parallelize"]["inner_cv_n_jobs"] = 1

        if cfg_analysis["parallelize"]["parallelize_imputation_runs"]:
            cfg_analysis["parallelize"]["imputation_runs_n_jobs"] = total_cores
        else:
            cfg_analysis["parallelize"]["imputation_runs_n_jobs"] = 1

        if cfg_analysis["parallelize"]["parallelize_shap"]:
            cfg_analysis["parallelize"]["shap_n_jobs"] = total_cores
        else:
            cfg_analysis["parallelize"]["shap_n_jobs"] = 1

        if cfg_analysis["parallelize"]["parallelize_shap_ia_values"]:
            cfg_analysis["parallelize"]["shap_ia_values_n_jobs"] = total_cores
        else:
            cfg_analysis["parallelize"]["shap_ia_values_n_jobs"] = 1

        return cfg_analysis

    @staticmethod
    def sanity_checks_cfg_cluster(cfg_analysis: NestedDict) -> NestedDict:
        """
        Ensures that certain parameters in the configuration are set correctly for cluster computation.

        For example, for testing certain analysis settings, I may run the CV procedure with num_cv=3. If I forget to re-adjust
        this setting in the config, this function does this automatically when running on the cluster.

        Args:
            cfg_analysis: Yaml config specifying details on the ML analysis (e.g., CV, models).

        Returns:
            cfg_analysis: Dict, containing the updated YAML cfg_analysis.
        """
        if "store_analysis_results" not in cfg_analysis["methods_to_apply"]:
            cfg_analysis["methods_to_apply"].append("store_analysis_results")

        # Hard-coded, but valid here
        cfg_analysis["cv"]["num_inner_cv"] = 10
        cfg_analysis["cv"]["num_outer_cv"] = 10
        cfg_analysis["cv"]["num_reps"] = 10
        cfg_analysis["imputation"]["num_imputations"] = 5
        cfg_analysis["imputation"]["max_iter"] = 40

        return cfg_analysis

    @staticmethod
    def construct_local_output_path(cfg_analysis: NestedDict) -> str:
        """
        Constructs the local output path for storing analysis results.

        Note: On the cluster, we construct and pass the path using the SLURM script.

        Args:
            cfg_analysis: Yaml config specifying details on the ML analysis (e.g., CV, models).

        Returns:
            str: The constructed local output path.
        """
        base_output_dir = cfg_analysis["output_path"]

        crit = cfg_analysis["params"]["crit"]
        feature_combination = cfg_analysis["params"]["feature_combination"]
        samples_to_include = cfg_analysis["params"]["samples_to_include"]
        prediction_model = cfg_analysis["params"]["prediction_model"]

        local_output_path = os.path.normpath(
            os.path.join(
                base_output_dir,
                feature_combination,
                samples_to_include,
                crit,
                prediction_model,
            )
        )

        if not os.path.exists(local_output_path):
            os.makedirs(local_output_path, exist_ok=True)

        return local_output_path
