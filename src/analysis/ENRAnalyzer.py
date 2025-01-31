import os
import pickle
from collections import defaultdict
from typing import Optional

import pandas as pd
from sklearn.linear_model import ElasticNet

from src.analysis.BaseMLAnalyzer import BaseMLAnalyzer
from src.utils.utilfuncs import defaultdict_to_dict, NestedDict


class ENRAnalyzer(BaseMLAnalyzer):
    """
    Implements functionality for linear models such as ElasticNet.

    This class extends the `BaseMLAnalyzer` to provide specific implementations
    for linear models, particularly ElasticNet. It handles tasks like coefficient
    aggregation across cross-validation loops while leveraging the functionality
    defined in the base class.

    Additional Attributes (for other Attributes, see BaseMLAnalyzer):
        model (ElasticNet): The linear model used for prediction and analysis.
    """

    def __init__(
        self,
        cfg_analysis: NestedDict,
        output_dir: str,
        df: pd.DataFrame,
        rep: Optional[int],
        rank: Optional[int],
    ) -> None:
        """
        Initializes the ENRAnalyzer instance with the specified configuration and data.

        Args:
            cfg_analysis: YAML configuration object specifying the analysis parameters.
            output_dir: Directory where the analysis results will be stored.
            df: Input DataFrame containing features and labels for the analysis.
            rep: Repetition index for cross-validation splits.
            rank: Rank identifier for multi-node parallelism.
        """
        super().__init__(cfg_analysis, output_dir, df, rep, rank)
        self.model = ElasticNet(random_state=self.cfg_analysis["random_state"])

    def get_average_coefficients(self) -> None:
        """
        Computes the average coefficients across all outer CV loops.

        This method:
        - Reads the best-performing models for each repetition and outer fold from stored files.
        - Extracts and aggregates the coefficients from these models.
        - Organizes the coefficients into a nested dictionary structure and stores them in
          the `self.lin_model_coefs` attribute.

        The results are sorted by the absolute magnitude of coefficients for interpretability.
        Requires that `self.best_models` has been properly populated during the analysis.
        """

        if not self.split_reps:
            if self.rank == 0:
                meta_vars_in_df = [
                    col for col in self.meta_vars if col in self.X.columns
                ]
                feature_names = self.X.columns.drop(meta_vars_in_df).tolist()
                coefs_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

                for rep in range(self.num_reps):
                    best_models_file = os.path.join(
                        self.spec_output_path, f"best_models_rep_{rep}.pkl"
                    )

                    if os.path.exists(best_models_file):
                        with open(best_models_file, "rb") as f:
                            best_models_rep = pickle.load(f)

                        for outer_fold_idx, outer_fold in enumerate(best_models_rep):
                            for imputation_idx, model in enumerate(outer_fold):
                                coefs_sub_dict = dict(zip(feature_names, model.coef_))

                                sorted_coefs_sub_dict = dict(
                                    sorted(
                                        coefs_sub_dict.items(),
                                        key=lambda item: abs(item[1]),
                                        reverse=True,
                                    )
                                )

                                coefs_dict[f"rep_{rep}"][
                                    f"outer_fold_{outer_fold_idx}"
                                ][
                                    f"imputation_{imputation_idx}"
                                ] = sorted_coefs_sub_dict
                    else:
                        self.logger.log(
                            f"WARNING: Best models file for rep {rep} not found"
                        )

                regular_dict = defaultdict_to_dict(coefs_dict)
                self.lin_model_coefs = regular_dict
