import os
from typing import Callable, Union, Collection, Optional, Any

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import shap
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter
from numpy import ndarray
from shap import Explanation

from src.utils.utilfuncs import NestedDict


class ResultPlotter:
    """
    Handles the visualization of processed data, focusing solely on the plotting tasks.

    This class is responsible for:
    - Setting up base plots and subplots.
    - Plotting specific types of data, such as SHAP beeswarm plots and cross-validation (CV) results.
    - Adjusting formatting for readability and presentation.
    - Saving plots to a structured directory based on the configuration.

    The plots created by this class include:
    - **SHAP Beeswarm Plots**: Visualizations of feature importance for model interpretation.
    - **CV Results Plots**: Visualizations of prediction results from cross-validation.
    - **Pred-True Parity Plots**: Plots comparing predicted vs. true values for selected analyses.

    Args:
        cfg_postprocessing: A nested dictionary containing all postprocessing configurations,
                            including SHAP plots and CV results plot settings.
        base_result_path: The base directory path where results and plots should be stored.

    Attributes:
        cfg_postprocessing (NestedDict): Stores the full postprocessing configuration.
        cfg_cv_results_plot (NestedDict): Configuration specific to CV results plots.
        cfg_shap_plot (NestedDict): Configuration specific to SHAP beeswarm plots.
        plot_base_dir (str): Path where all generated plots will be saved.
    """

    def __init__(self, cfg_postprocessing: NestedDict, base_result_path: str) -> None:
        self.cfg_postprocessing = cfg_postprocessing

        self.cfg_cv_results_plot = self.cfg_postprocessing["create_cv_results_plots"]
        self.cfg_shap_plot = self.cfg_postprocessing["create_shap_plots"]

        self.plot_base_dir = os.path.join(
            base_result_path, self.cfg_postprocessing["general"]["data_paths"]["plots"]
        )

    def plot_cv_results_plots_wrapper(
        self, cv_results_dct: NestedDict, rel: Optional[float] = None
    ) -> None:
        """
        Wraps the `plot_cv_results_plots` function to generate and plot cross-validation results.

        This method:
        1. Iterates over all combinations of metrics, criteria (`crit`), and samples to include.
        2. Creates a new base plot for each combination.
        3. Retrieves the specific data needed for plotting, including:
            - Metric columns filtered for the given combination.
            - Reference values (e.g., baseline or comparison values).
            - Margins for comparison to reference values.
        4. Invokes the `plot_cv_results_plots` function to generate the bar plots for the given combination.

        Args:
            cv_results_dct: Nested dictionary containing cross-validation results for multiple combinations
                            of criteria, feature combinations, samples to include, and models.
                            Each entry corresponds to mean/standard deviation values for a given metric.
            rel: Optional reliability of the specified criterion. If `None`, reliability is not included in the plots.
        """
        feature_combos = [
            feature_combination_lst
            for feature_combination_lst in self.cfg_cv_results_plot[
                "col_assignment"
            ].values()
        ]

        for metric in self.cfg_cv_results_plot["metrics"]:
            for crit in self.cfg_cv_results_plot["crits"]:
                for samples_to_include in self.cfg_cv_results_plot[
                    "samples_to_include"
                ]:
                    fig, axes = self.create_grid(
                        num_cols=self.cfg_cv_results_plot["figure_params"]["num_cols"],
                        num_rows=self.cfg_cv_results_plot["figure_params"]["num_rows"],
                        figsize=(
                            self.cfg_cv_results_plot["figure_params"]["width"],
                            self.cfg_cv_results_plot["figure_params"]["height"],
                        ),
                        empty_cells=[
                            tuple(cell)
                            for cell in self.cfg_cv_results_plot["figure_params"][
                                "empty_cells"
                            ]
                        ],
                    )

                    filtered_metric_cols = self.prepare_cv_results_plot_data(
                        cv_results_dct=cv_results_dct,
                        crit=crit,
                        samples_to_include=samples_to_include,
                        metric=metric,
                        feature_combinations=feature_combos,
                    )
                    ref_samples_to_include = self.cfg_cv_results_plot["ref_dct"][
                        "samples_to_include"
                    ]
                    ref_feature_combo = self.cfg_cv_results_plot["ref_dct"][
                        "feature_combo"
                    ]

                    ref_dct = self.get_refs(
                        cv_results_dct=cv_results_dct,
                        crit=crit,
                        metric=metric,
                        samples_to_include=ref_samples_to_include,
                        ref_feature_combo=ref_feature_combo,
                    )
                    margin_dct = self.get_margins(
                        cv_results_dct=cv_results_dct,
                        ref_dct=ref_dct,
                        metric=metric,
                        crit=crit,
                        samples_to_include=ref_samples_to_include,
                        ref_feature_combo=ref_feature_combo,
                    )

                    # Create plots for a given metric - crit - samples_to_include combination
                    self.plot_cv_results_plots(
                        feature_combinations=feature_combos,
                        crit=crit,
                        samples_to_include=samples_to_include,
                        titles=self.cfg_cv_results_plot["titles"],
                        filtered_metric_col=filtered_metric_cols,
                        margin_dct=margin_dct,
                        ref_dct=ref_dct,
                        fig=fig,
                        axes=axes,
                        models=self.cfg_cv_results_plot["models"],
                        color_dct=self.cfg_cv_results_plot["color_dct"],
                        fontsizes=self.cfg_cv_results_plot["fontsizes"],
                        figure_params=self.cfg_cv_results_plot["figure_params"],
                        metric=metric,
                        rel=rel,
                    )

    def prepare_cv_results_plot_data(
        self,
        cv_results_dct: NestedDict,
        crit: str,
        metric: str,
        samples_to_include: str,
        feature_combinations: list[list[str]],
    ) -> list[NestedDict]:
        """
        Prepares the filtered data for CV results plotting.

        This function:
        - Filters the data to include only the specified `crit` and `samples_to_include`.
        - Extracts the relevant metrics for each subplot location based on the feature_combinations and models.

        Args:
            cv_results_dct: Nested dictionary containing CV results for a given crit, feature_combo, samples_to_include, model, and metric.
            crit: The criterion being processed (e.g., 'na_state').
            samples_to_include: The sample type to include ("all", "selected", or "combo").
            metric: The metric to plot (e.g., 'spearman', 'pearson', 'r2').
            feature_combinations: List of feature groups to consider for filtering.

        Returns:
            list[NestedDict]: A list where each element contains filtered data for one feature group.
        """
        if samples_to_include not in ["all", "selected", "combo"]:
            raise ValueError(
                "Invalid value for samples_to_include. Must be one of ['all', 'selected', 'combo']."
            )

        filtered_metrics_cols = []

        for i, group in enumerate(feature_combinations):
            # For "combo", the first column uses "selected" samples, and the others use "all"
            sublist_sample = (
                "selected" if samples_to_include == "combo" and i == 0 else "all"
            )

            print(sublist_sample)
            filtered_metrics = self.filter_cv_results_data(
                cv_results_dct=cv_results_dct,
                crit=crit,
                samples_to_include=sublist_sample,
                metric=metric,
            )

            filtered_metric_col = {
                f"{feature_combo}_{model}": model_vals
                for feature_combo, feature_combo_vals in filtered_metrics.items()
                for model, model_vals in feature_combo_vals.items()
                if feature_combo in group
            }

            filtered_metrics_cols.append(filtered_metric_col)

        return filtered_metrics_cols

    @staticmethod
    def filter_cv_results_data(
        cv_results_dct: NestedDict, crit: str, metric: str, samples_to_include: str
    ) -> NestedDict:
        """
        Filters the cv_results_dct for a specific criterion, samples_to_include, and metric

        Args:
            cv_results_dct: Nested dictionary containing CV results for crit, feature_combo, samples_to_include, model, and metric.
            crit: The criterion to filter by (e.g., 'na_state').
            metric: The metric to plot (e.g., 'spearman', 'pearson', 'r2').
            samples_to_include: The sample type to filter by (e.g., 'all' or 'selected').

        Returns:
            NestedDict: Filtered dictionary (given metric, samples_to_include, crit)
                with keys for feature_combinations, models.
        """
        if crit not in cv_results_dct:
            raise KeyError(
                f"Criterion '{crit}' not found in the provided CV results dictionary."
            )

        crit_results = cv_results_dct[crit]

        if samples_to_include not in crit_results:
            raise KeyError(
                f"Samples_to_include '{samples_to_include}' not found for criterion '{crit}'."
            )

        samples_results = crit_results[samples_to_include]

        filtered_results = {
            feature_comb: {
                model: {k: float(v) for k, v in metrics.get(metric, {}).items()}
                for model, metrics in models_metrics.items()
            }
            for feature_comb, models_metrics in samples_results.items()
        }

        return filtered_results

    def plot_cv_results_plots(
        self,
        feature_combinations: list[list[str]],
        crit: str,
        samples_to_include: str,
        titles: list[str],
        filtered_metric_col: list[NestedDict],
        margin_dct: dict[str, dict[str, float]],
        ref_dct: dict[str, dict[str, float]],
        fig: Figure,
        axes: ndarray[Axes],
        models: list[str],
        color_dct: NestedDict,
        fontsizes: dict[str, int],
        figure_params: dict[Any],
        metric: str,
        rel: float = None,
    ) -> None:
        """
        Creates bar plots summarizing cross-validation (CV) results for a given analysis.

        This method visualizes CV results for the different analysis levels (e.g., one-level, two-level, three-level).
        Each column in the plot grid represents a different analysis level, and each row corresponds
        to a specific feature combination.

        Methodology:
            - Iterates through columns (analysis levels) and rows (feature combinations) in the subplot grid.
            - Creates either standard bar plots or incremental bar plots based on the column index.
            - Configures axes, titles, and formatting based on the provided `figure_params` and `fontsizes`.
            - Adds a legend to the plot to indicate models and feature combinations.
            - Draws a joint y-axis across all subplots for consistent scaling.
            - Saves the plot to a specified location or displays it interactively, depending on the `store_plots` flag.

        Args:
            feature_combinations: A nested list of feature combinations for each column (e.g., pl)
            crit: The criterion used in the analysis (e.g., "wb_state").
            samples_to_include:: The sample inclusion criteria for the analysis (e.g., "all").
            titles: Titles for the columns, representing analysis levels (e.g., one-level).
            filtered_metric_col: A list of dicts containing the filtered metric for each feature combination.
            margin_dct: Dictionary with incremental values compare to a reference for each prediction level.
            ref_dct: Dictionary with reference values for each prediction level.
            fig: The matplotlib Figure object used for the plot grid.
            axes: An array of Axes objects for the subplots.
            models: A list of model names to include in the bar plots.
            color_dct: A dictionary mapping models and features to specific colors.
            fontsizes: A dictionary specifying font sizes for various plot elements (e.g., tick labels, legends, titles).
            figure_params: A dictionary containing parameters for the figure layout (e.g., x-axis limits, title positions).
            metric: The performance metric to visualize (e.g., "r2", "r").
            rel: The reliability of the criterion, if we include this.

        """
        feature_combo_name_mapping = self.cfg_postprocessing["general"][
            "feature_combinations"
        ]["name_mapping"]["main"]

        for col_idx, (group, title, metrics) in enumerate(
            zip(feature_combinations, titles, filtered_metric_col)
        ):
            for row_idx, feature_combo in enumerate(group):
                ax = axes[row_idx, col_idx]
                single_category_metrics = {
                    f"{feature_combo}_{model}": metrics[f"{feature_combo}_{model}"]
                    for model in models
                    if f"{feature_combo}_{model}" in metrics
                }

                if col_idx == 0:
                    self.plot_bar_plot(
                        ax=ax,
                        row_idx=row_idx,
                        data=single_category_metrics,
                        feature_combo=feature_combo,
                        models=models,
                        color_dct=color_dct,
                        feature_combo_mapping=feature_combo_name_mapping,
                        metric=metric,
                        fontsizes=fontsizes,
                        bar_width=figure_params["bar_width"],
                        rel=rel,
                    )

                elif col_idx == 1 or col_idx == 2 and row_idx < 2:
                    self.plot_incremental_bar_plot(
                        ax=ax,
                        row_idx=row_idx,
                        data=single_category_metrics,
                        pl_margin_dct=margin_dct,
                        feature_combo=feature_combo,
                        feature_combo_mapping=feature_combo_name_mapping,
                        models=models,
                        color_dct=color_dct,
                        metric=metric,
                        fontsizes=fontsizes,
                        bar_width=figure_params["bar_width"],
                        rel=rel,
                        pl_reference_dct=ref_dct,
                    )

                ax.set_xlim(figure_params["x_min"], figure_params["x_max"])
                ax.set_title(
                    title if row_idx == 0 else "",
                    fontsize=fontsizes["tick_params"],
                    pad=figure_params["title_pad"],
                    fontweight="bold",
                    loc="center",
                )
                ax.title.set_position(figure_params["title_pos"])
                ax.set_xlim(figure_params["x_min"], figure_params["x_max"])
                ax.tick_params(axis="y", labelsize=fontsizes["tick_params"])

                for spine in ["top", "right", "left"]:
                    ax.spines[spine].set_visible(False)
                if row_idx < len(group) - 1:
                    ax.tick_params(axis="x", bottom=False, labelbottom=False)
                    ax.spines["bottom"].set_visible(False)

        self.add_cv_results_plot_legend(
            fig=fig,
            legend_cfg=self.cfg_cv_results_plot["legend"],
            legend_fontsize=fontsizes["legend"],
            color_dct=color_dct,
            rel=True if rel else False,
        )

        self.manually_add_x_label(
            fig=fig,
            metric=metric,
            position=figure_params["manual_x_label"]["pos"],
            fontsize=fontsizes["label"],
        )

        y_line_cfg = figure_params["y_line"]
        for x, y in zip(y_line_cfg["x_pos"], y_line_cfg["y_pos"]):
            line = mlines.Line2D(
                xdata=[x, x],
                ydata=y,
                transform=fig.transFigure,
                color=y_line_cfg["color"],
                linestyle=y_line_cfg["linestyle"],
                linewidth=y_line_cfg["linewidth"],
            )
            fig.add_artist(line)

        plt.tight_layout(rect=figure_params["tight_layout"])

        if self.cfg_cv_results_plot["store_params"]["store"]:
            self.store_plot(
                plot_name=self.cfg_cv_results_plot["store_params"]["name"],
                plot_format=self.cfg_cv_results_plot["store_params"]["format"],
                dpi=self.cfg_cv_results_plot["store_params"]["dpi"],
                crit=crit,
                samples_to_include=samples_to_include,
                model=None,
            )

        else:
            plt.show()

    def add_cv_results_plot_legend(
        self,
        fig: Figure,
        legend_cfg: dict[str, Any],
        legend_fontsize: int,
        color_dct: dict,
        ref_feature_combo: str = "pl",
        rel: bool = False,
    ) -> None:
        """
        Adds a legend to the lower-right corner of the cross-validation results plot.

        This function enhances the plot by:
        - Adding a legend to describe the bar locations corresponding to different models.
        - Adding a legend to describe the colors corresponding to feature groups.
        - Optionally, adding a reliability indicator (if `rel` is provided).

        Args:
            fig: The Matplotlib `Figure` object to which the legend will be added.
            legend_cfg: Configuration dictionary for the legend. Contains settings for:
                - Model legend (e.g., whether to add it, colors, and labels for ElasticNet and RandomForest).
                - Feature combination legend (e.g., whether to add it, colors, and labels for reference and other combos).
                - General legend properties such as position, column count, and alignment.
            legend_fontsize: Font size for the legend text.
            color_dct: A dictionary mapping feature group names to their corresponding colors.
            ref_feature_combo: The reference feature combination key for the plot (default: "pl").
            rel: Optional reliability value to be added to the legend as a line. If `None`, the reliability line is omitted.

        """
        legends_to_plot = []

        if legend_cfg["model_legend"]["add"]:
            model_legend_cfg = legend_cfg["model_legend"]
            elasticnet_patch = mpatches.Patch(
                facecolor=model_legend_cfg["enr_color"],
                edgecolor="none",
                label=model_legend_cfg["enr_label"],
            )
            randomforest_patch = mpatches.Patch(
                facecolor=model_legend_cfg["rfr_color"],
                edgecolor="none",
                label=model_legend_cfg["rfr_label"],
            )
            legends_to_plot.extend([elasticnet_patch, randomforest_patch])

        if legend_cfg["feature_combo_legend"]["add"]:
            feature_combo_legend_cfg = legend_cfg["feature_combo_legend"]
            personal_patch = mpatches.Patch(
                facecolor=color_dct[ref_feature_combo],
                edgecolor="none",
                label=feature_combo_legend_cfg["ref_label"],
            )
            other_patch = mpatches.Patch(
                facecolor=color_dct["other"],
                edgecolor="none",
                label=feature_combo_legend_cfg["other_label"],
            )
            legends_to_plot.extend([personal_patch, other_patch])

        if rel:
            rel_cfg = self.cfg_cv_results_plot["rel"]
            reliability_line = mlines.Line2D(
                [],
                [],
                color=rel_cfg["color"],
                linestyle=rel_cfg["linestyle"],
                linewidth=rel_cfg["linewidth"],
                label=f"{rel_cfg['base_label']}{rel}",
            )
            legends_to_plot.append(reliability_line)

        # Add the custom legend to the figure
        fig.legend(
            handles=legends_to_plot,
            loc=legend_cfg["legend_loc"],
            bbox_to_anchor=legend_cfg["legend_pos"],
            ncol=legend_cfg["ncol"],
            frameon=False,
            title="",
            fontsize=legend_fontsize,
        )

    def plot_bar_plot(
        self,
        ax: Axes,
        data: dict[str, dict[str, float]],
        feature_combo: str,
        models: list[str],
        color_dct: dict[str, Union[str, dict[str, float]]],
        feature_combo_mapping: dict[str, str],
        fontsizes: dict[str, int],
        metric: str,
        bar_width: float,
        row_idx: int,
        ref_feature_combo: str = "pl",
        rel: float = None,
    ) -> None:
        """
        Creates a horizontal bar plot with error bars for a given feature group and its associated models.

        The function plots bars for each model associated with a feature combination. It uses distinct colors
        for reference and non-reference feature combinations and applies different saturation levels to
        differentiate models. An optional vertical line can be added to represent a threshold (`rel`).

        Args:
            ax: The Matplotlib `Axes` object to draw the plot on.
            data: A dictionary where keys are feature-model combinations (e.g., "pl_elasticnet") and values are
                  dictionaries containing metrics (e.g., mean and standard deviation).
            feature_combo: The feature combination key to plot (e.g., "pl", "srmc").
            models: A list of model names to include in the plot (e.g., ["elasticnet", "randomforestregressor"]).
            color_dct: A dictionary specifying colors and bar saturations for models and feature categories.
            feature_combo_mapping: A mapping dictionary to format feature combination labels for display.
            fontsizes: A dictionary specifying font sizes for plot elements.
            metric: The name of the metric being plotted (e.g., "MSE").
            bar_width: The width of each bar in the plot.
            row_idx: The index of the row in the subplot grid, used for formatting purposes.
            ref_feature_combo: The reference feature combination key for determining the bar color (default: "pl").
            rel: Optional reliability threshold to be shown as a vertical line. If `None`, no line is added.
        """
        for i, model in enumerate(models):
            if feature_combo == ref_feature_combo:
                base_color = color_dct[ref_feature_combo]
            else:
                base_color = color_dct["other"]
            alpha = color_dct["bar_saturation"][model]

            # ENR above, RFR below
            y_position_to_plot = 0 + (1 - 2 * i) * (bar_width / 2)
            model_key = f"{feature_combo}_{model}"
            value = data[model_key][self.cfg_cv_results_plot["m_metric"]]
            error = data[model_key][self.cfg_cv_results_plot["sd_metric"]]

            ax.barh(
                y_position_to_plot,
                value,
                xerr=error,
                height=bar_width,
                color=base_color,
                align=self.cfg_cv_results_plot["figure_params"]["bar_align"],
                capsize=self.cfg_cv_results_plot["figure_params"]["bar_capsize"],
                edgecolor=None,
                alpha=alpha,
            )

        self.format_bar_plot(
            ax=ax,
            row_idx=row_idx,
            y_position=y_position_to_plot,
            bar_width=bar_width,
            feature_combo_mapping=feature_combo_mapping,
            feature_combo=feature_combo,
            metric=metric,
            fontsizes=fontsizes,
            rel=rel,
        )

    def plot_incremental_bar_plot(
        self,
        ax: Axes,
        data: dict[str, dict[str, float]],
        pl_margin_dct: dict[str, dict[str, float]],
        pl_reference_dct: dict[str, dict[str, float]],
        feature_combo: str,
        models: list[str],
        color_dct: NestedDict,
        feature_combo_mapping: dict[str, str],
        fontsizes: dict[str, int],
        metric: str,
        row_idx: int,
        bar_width: float,
        ref_feature_combo: str = "pl",
        rel: float = None,
    ) -> None:
        """
        Plots an incremental bar plot, splitting each bar into the reference "pl" base value and the incremental effect.

        This function visualizes how each feature combination contributes to the metric of interest in splitting
        the bars into two components:
        - **Base Value**: The performance of the reference feature combination (`pl`).
        - **Incremental Effect**: The additional contribution of the current feature combination.

        It also uses different colors to distinguish between the base and incremental parts of the bars.
        An optional vertical line (`rel`) can be added to indicate a threshold.

        Args:
            ax: The Matplotlib `Axes` object to draw the plot on.
            data: A dictionary containing metrics for each feature-model combination.
            pl_margin_dct: A dictionary containing incremental metric values for each feature-model combination.
            pl_reference_dct: A dictionary containing base metric values for the reference feature combination.
            feature_combo: The current feature combination being plotted (e.g., "pl", "srmc").
            models: A list of model names to include in the plot (e.g., ["elasticnet", "randomforestregressor"]).
            color_dct: A dictionary specifying colors and bar saturations for models and feature categories.
            feature_combo_mapping: A mapping dictionary to format feature combination labels for display.
            fontsizes: A dictionary specifying font sizes for plot elements (e.g., axis labels, titles).
            metric: The name of the metric being plotted (e.g., "MSE").
            row_idx: The index of the row in the subplot grid, used for formatting purposes.
            bar_width: The width of each bar in the plot.
            ref_feature_combo: The reference feature combination key for determining reference bars (default: "pl").
            rel: Optional reliability threshold to be shown as a vertical line. If `None`, no line is added.

        """
        m_metric = self.cfg_cv_results_plot["m_metric"]
        sd_metric = self.cfg_cv_results_plot["sd_metric"]

        for i, model in enumerate(models):
            base_value = pl_reference_dct[f"{ref_feature_combo}_{model}"][m_metric]
            increment = pl_margin_dct[f"{feature_combo}_{model}"][
                f"incremental_{m_metric}"
            ]
            error = data[f"{feature_combo}_{model}"][sd_metric]

            y_position_to_plot = 0 + (1 - 2 * i) * (bar_width / 2)
            alpha = color_dct["bar_saturation"][model]
            other_feat_color = color_dct["other"]
            ref_color = color_dct[ref_feature_combo]

            if increment > 0:
                ax.barh(
                    y_position_to_plot,
                    base_value,
                    height=bar_width,
                    color=ref_color,
                    align=self.cfg_cv_results_plot["figure_params"]["bar_align"],
                    edgecolor=None,
                    alpha=alpha,
                )
                ax.barh(
                    y_position_to_plot,
                    increment,
                    left=base_value,
                    height=bar_width,
                    xerr=error,
                    color=other_feat_color,
                    align=self.cfg_cv_results_plot["figure_params"]["bar_align"],
                    edgecolor=None,
                    alpha=alpha,
                    capsize=self.cfg_cv_results_plot["figure_params"]["bar_capsize"],
                )

            else:
                # If there is no incremental performance, only plot the bar for the ref
                real_value = base_value + increment
                ax.barh(
                    y_position_to_plot,
                    real_value,
                    height=bar_width,
                    color=ref_color,
                    xerr=error,
                    align=self.cfg_cv_results_plot["figure_params"]["bar_align"],
                    edgecolor=None,
                    alpha=alpha,
                    capsize=self.cfg_cv_results_plot["figure_params"]["bar_capsize"],
                )

            self.format_bar_plot(
                ax=ax,
                row_idx=row_idx,
                y_position=y_position_to_plot,
                bar_width=bar_width,
                feature_combo_mapping=feature_combo_mapping,
                feature_combo=feature_combo,
                metric=metric,
                fontsizes=fontsizes,
                rel=rel,
            )

    def format_bar_plot(
        self,
        ax: Axes,
        row_idx: int,
        y_position: float,
        bar_width: float,
        feature_combo_mapping: dict[str, str],
        feature_combo: str,
        fontsizes: dict[str, int],
        metric: str,
        rel: float = None,
    ) -> None:
        """
        Formats a bar plot by adjusting y-ticks, x-axis labels, and adding an optional reference line.

        This function:
        - Sets y-ticks and their labels based on feature combinations.
        - Formats the x-axis, including tick labels and axis labels based on the metric.
        - Optionally adds a vertical reference line (`rel`) to indicate a threshold or baseline.
        - Hides unnecessary plot spines and adjusts tick parameters for aesthetics.

        Args:
            ax: The Matplotlib `Axes` object to format.
            row_idx: The row index of the subplot in a grid. Used to determine if the x-axis label is added.
            y_position: The y-coordinate of the bar to set y-ticks for.
            bar_width: The width of the bars, used for positioning y-ticks.
            feature_combo_mapping: A dictionary mapping feature combination keys to display labels.
                                   Example: {"pl": "Primary Features", "srmc": "Secondary Features"}.
            feature_combo: The key for the current feature combination being plotted (e.g., "pl").
            fontsizes: A dictionary containing font sizes for various plot elements.
            metric: The metric being plotted (e.g., "spearman", "pearson", "r2").
            rel: Optional float for a vertical reference line to indicate a threshold or baseline. If `None`, no line is added.
        """
        y_arr = np.array([y_position])
        ax.set_yticks(y_arr + bar_width / 2)
        xlabels_format_cfg = self.cfg_cv_results_plot["format_bar_plot"]["xlabels"]

        feature_combo_formatted = self.line_break_strings(
            feature_combo_mapping[feature_combo],
            max_char_on_line=xlabels_format_cfg["max_char_on_line"],
            balance=xlabels_format_cfg["balance"],
            split_strng=xlabels_format_cfg["split_strng"],
            force_split_strng=xlabels_format_cfg["force_split_strng"],
        )

        n_samples = self.cfg_cv_results_plot["n_samples"][feature_combo]
        feature_combos_str_format_n = self.add_n_to_title(
            feature_combo_formatted, n_samples
        )

        axes_cfg = self.cfg_cv_results_plot["format_bar_plot"]["axes"]
        ax.set_yticklabels(
            [feature_combos_str_format_n],
            fontsize=fontsizes["tick_params"],
            horizontalalignment=axes_cfg["ylabels"]["hor_align"],
        )

        ax.spines["left"].set_visible(False)
        xticks_decimals = self.cfg_cv_results_plot["format_bar_plot"]["axes"]["xticks"][
            "decimals"
        ]
        ax.xaxis.set_major_formatter(
            FuncFormatter(lambda x, _: self.format_metric(x, xticks_decimals))
        )
        ax.tick_params(axis="x", labelsize=fontsizes["tick_params"])
        ax.tick_params(
            axis="y",
            which=axes_cfg["yticks"]["which"],
            pad=axes_cfg["yticks"]["pad"],
            length=axes_cfg["yticks"]["length"],
        )

        if row_idx == 3:
            xlabel_pad = axes_cfg["xlabels"]["pad"]
            if metric == "spearman":
                ax.set_xlabel(
                    r"$\rho$", fontsize=fontsizes["label"], labelpad=xlabel_pad
                )
            elif metric == "pearson":
                ax.set_xlabel(r"$r$", fontsize=fontsizes["label"], labelpad=xlabel_pad)
            elif metric == "r2":
                ax.set_xlabel(
                    r"$R^2$", fontsize=fontsizes["label"], labelpad=xlabel_pad
                )

        if rel:
            rel_cfg = self.cfg_cv_results_plot["rel"]
            ax.axvline(
                x=rel,
                color=rel_cfg["color"],
                linestyle=rel_cfg["linestyle"],
                linewidth=rel_cfg["linewidth"],
                label=f"{rel_cfg['base_label']}{rel:.2f}",
            )

    @staticmethod
    def manually_add_x_label(
        fig: Figure,
        metric: str,
        position: tuple[float, float],
        fontsize: int,
    ) -> None:
        """
        Adds a correlation metric as a text annotation on the global figure.

        Args:
            fig: The Matplotlib figure object where the text should be added.
            metric: The correlation metric, one of 'spearman', 'pearson', or 'r2'.
            position: A tuple (x, y) specifying the position of the text in axis coordinates.
            fontsize: Fontsize for the label
        """
        metric_symbols = {"spearman": r"$\rho$", "pearson": r"$r$", "r2": r"$R^2$"}
        symbol = metric_symbols[metric]

        fig.text(
            x=position[0],
            y=position[1],
            s=symbol,
            fontsize=fontsize,
            verticalalignment="top",
            horizontalalignment="right",
        )

    @staticmethod
    def format_metric(metric: float, xticks_decimals: int = 2) -> Union[str, int]:
        """
        Formats a given metric according to APA style:
            - No leading zero for correlations or R² values (e.g., .85 instead of 0.85).
            - Rounded to the specified number of decimal places.

        Args:
            metric: The metric to be formatted (e.g., a correlation or R² value).
            xticks_decimals: The number of decimal places for rounding. Defaults to 2.

        Returns:
            Union[str, int]: A string representation of the formatted metric or 0 for exact zero.
        """
        if not isinstance(metric, (float, int)):
            raise ValueError("The metric must be a numeric value.")

        if metric == 0:
            return 0

        return f"{metric:.{xticks_decimals}f}".lstrip("0").replace("-.", "-0.")

    @staticmethod
    def create_grid(
        num_rows: int,
        num_cols: int,
        figsize: tuple[int] = (15, 20),
        empty_cells: Collection[tuple[int]] = None,
    ) -> tuple[Figure, ndarray[Axes]]:
        """
        Creates a flexible grid of subplots with customizable empty cells.

        Args:
            num_rows: Number of rows in the grid.
            num_cols: Number of columns in the grid.
            figsize: Figure size for the entire grid (default is (15, 20)).
            empty_cells: List of (row, col) tuples where cells should be empty.

        Returns:
            fig: Matplotlib figure object for the grid.
            axes: Array of axes objects for individual plots.
        """
        fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize, squeeze=False)

        if empty_cells:
            for row, col in empty_cells:
                fig.delaxes(axes[row, col])  # Remove the axis for empty cells

        return fig, axes

    def get_margins(
        self,
        cv_results_dct: dict[str, dict[str, float]],
        ref_dct: dict[str, dict[str, float]],
        crit: str,
        samples_to_include: str,
        metric: str,
        ref_feature_combo: str = "pl",
    ) -> dict[str, dict[str, float]]:
        """
        Computes the incremental performance difference for all combinations involving 'pl' (e.g., 'pl_srmc')
        compared to the base 'pl' metric for each model.

        Methodology
            - extracts the base metric and the combined metrics to compute the increment
            - Loop over each key in the filtered_metrics that contains ref but is not ref (e.g., pl_srmc vs. pl)
            - stores the increment and the standard deviation of the combined metric in a dict
            - we need the sd of the combined analysis for the plot

        Args:
            cv_results_dct: Dictionary containing metrics for each model and feature combination.
            ref_dct: Dict containing the reference values for both models (e.g., pl_elasticnet).
            crit: e.g., wb_state
            samples_to_include: e.g., all
            metric: The metric to compute the incremental performance difference for.
            ref_feature_combo: The reference feature combination to compare the incremental performance to (default is 'pl').

        Returns:
            dict: Dictionary containing incremental performance differences for each 'pl' combination and model.
        """
        margin_dict = {}
        m_metric = self.cfg_cv_results_plot["m_metric"]
        sd_metric = self.cfg_cv_results_plot["sd_metric"]
        ref = f"{ref_feature_combo}_"

        filtered_metrics = self.filter_cv_results_data(
            cv_results_dct=cv_results_dct,
            crit=crit,
            samples_to_include=samples_to_include,
            metric=metric,
        )

        for feature_combo, feature_combo_vals in filtered_metrics.items():
            for model, model_vals in feature_combo_vals.items():
                if (
                    feature_combo.startswith(ref)
                    and feature_combo not in ref_dct
                    or feature_combo.startswith("all_in")
                ):
                    ref_key = f"{ref}{model}"

                    if ref_key in ref_dct:
                        incremental_difference = (
                            model_vals[m_metric] - ref_dct[ref_key][m_metric]
                        )
                        margin_dict[f"{feature_combo}_{model}"] = {
                            f"incremental_{m_metric}": incremental_difference,
                            sd_metric: model_vals[sd_metric],
                        }

        return margin_dict

    @staticmethod
    def get_refs(
        cv_results_dct: NestedDict,
        crit: str,
        metric: str,
        samples_to_include: str,
        ref_feature_combo: str = "pl",
    ) -> NestedDict:
        """
        Extracts reference values for the two- and three-level analysis in cross-validation result plots.

        This function retrieves the metric values (e.g., mean and standard deviation) for the reference
        feature combination (`ref_feature_combo`) across two models: ElasticNet and RandomForestRegressor.

        Args:
            cv_results_dct: A nested dictionary containing cross-validation results for multiple feature combinations,
                            criteria, and sample subsets. Each entry corresponds to a metric (e.g., mean, standard deviation).
            crit: The criterion key for the analysis (e.g., "wb_state").
            metric: The specific metric to extract (e.g., "m_metric", "sd_metric").
            samples_to_include: The subset of samples to include in the analysis (e.g., "all").
            ref_feature_combo: The reference feature combination key (default: "pl").

        Returns:
            NestedDict: A dictionary containing the reference values for both models (ElasticNet and RandomForestRegressor).
                        Example:
                        {
                            "pl_elasticnet": {"m_metric": 0.8, "sd_metric": 0.05},
                            "pl_randomforestregressor": {"m_metric": 0.85, "sd_metric": 0.06}
                        }
        """
        enr_subdct = cv_results_dct[crit][samples_to_include][ref_feature_combo][
            "elasticnet"
        ][metric]
        rfr_subdct = cv_results_dct[crit][samples_to_include][ref_feature_combo][
            "randomforestregressor"
        ][metric]

        return {
            f"{ref_feature_combo}_elasticnet": {
                stat: float(val) for stat, val in enr_subdct.items()
            },
            f"{ref_feature_combo}_randomforestregressor": {
                stat: float(val) for stat, val in rfr_subdct.items()
            },
        }

    def plot_shap_beeswarm_plots(
        self,
        prepare_shap_data_func: Callable,
        prepare_shap_ia_data_func: Callable = None,
    ) -> None:
        """
            Generates SHAP beeswarm plots for different predictor combinations arranged in a grid layout.

            This method visualizes SHAP values, which indicate the contribution of individual features
            to model predictions. It supports plotting:
            - Main SHAP values for specified predictor combinations across models and datasets.
            - Interaction SHAP values (optional) for Random Forest models if configured.

            The plots are organized in a grid defined by the `col_assignment` configuration.
            Subplots for missing data or empty cells are skipped, and customizable color maps
            and layout adjustments are applied.

                Method Details:
        - **Grid Layout and Assignment**:
            - The `col_assignment` configuration determines the layout of predictor combinations
              in the grid, assigning each combination to a specific row and column.
            - Interaction SHAP values, if enabled, are positioned at a specified location
              within the grid.

        - **Data Retrieval**:
            - The `prepare_shap_data_func` retrieves SHAP values for the current combination
              of criterion, dataset, and model.
            - If interaction SHAP values are enabled, the `prepare_shap_ia_data_func` is invoked
              to retrieve the interaction SHAP values.

        - **Plot Customization**:
            - Custom color maps and font sizes are applied to each plot.
            - Titles are added to specific rows and columns to indicate the type of SHAP values
              being visualized (main or interaction values).
            - Subplot adjustments (e.g., spacing, margins) are made using the `subplot_adjustments` configuration.

        - **Storage and Display**:
            - Plots are either saved to a file or displayed, based on the configuration in
              `store_params`. The file format and resolution are also configurable.

            Args:
                prepare_shap_data_func: A callable function that retrieves the SHAP values
                                        for a given combination of criterion, dataset, and model.
                                        Expected to return a dictionary of SHAP values for all
                                        predictor combinations in the current configuration.
                prepare_shap_ia_data_func: A callable function that retrieves
                                           interaction SHAP values for Random Forest models.
                                           If provided, interaction SHAP values are added to
                                           specific subplots based on configuration.

            Returns:
                None: The function generates and either saves or displays the SHAP beeswarm plots.
        """
        col_assignment = self.cfg_shap_plot["col_assignment"]
        positions = {}

        for col_idx, (col_name, column) in enumerate(col_assignment.items()):
            for row_idx, predictor_combination in enumerate(column):
                positions[(row_idx, col_idx)] = predictor_combination

        colors = self.cfg_postprocessing["general"]["global_plot_params"][
            "custom_cmap_colors"
        ]
        cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)

        cfg_ia_values = self.cfg_shap_plot["ia_values"]

        if self.cfg_shap_plot["analysis"] == "main":
            mapping_str = "main"
        else:
            mapping_str = "supp"
        feature_combo_name_mapping = self.cfg_postprocessing["general"][
            "feature_combinations"
        ]["name_mapping"][mapping_str]
        if cfg_ia_values["add"]:
            feature_combo_name_mapping = {
                **feature_combo_name_mapping,
                **self.cfg_postprocessing["general"]["feature_combinations"][
                    "name_mapping"
                ]["ia_values"],
            }

        num_to_display = self.cfg_shap_plot["num_to_display"]
        beeswarm_figure_params = self.cfg_shap_plot["figure_params"]
        beeswarm_subplot_adj = beeswarm_figure_params["subplot_adjustments"]
        beeswarm_fontsizes = self.cfg_shap_plot["fontsizes"]
        beeswarm_title_params = self.cfg_shap_plot["titles"]
        ia_data_current = None

        for crit in self.cfg_shap_plot["crits"]:
            for samples_to_include in self.cfg_shap_plot["samples_to_include"]:
                for model in self.cfg_shap_plot["models"]:
                    print(f"### Plot combination: {samples_to_include}_{crit}_{model}")
                    data_current = prepare_shap_data_func(
                        crit_to_plot=crit,
                        samples_to_include=samples_to_include,
                        model_to_plot=model,
                        col_assignment=col_assignment,
                    )
                    if cfg_ia_values["add"] and model == "randomforestregressor":
                        ia_data_current = prepare_shap_ia_data_func(
                            crit_to_plot=crit,
                            samples_to_include=samples_to_include,
                            model_to_plot=model,
                            feature_combination_to_plot=cfg_ia_values[
                                "feature_combination"
                            ],
                        )
                        ia_position = tuple(cfg_ia_values["position"])
                        positions[ia_position] = next(iter(ia_data_current))

                    figure_params = self.cfg_shap_plot["figure_params"]
                    empty_cells = (
                        [self.cfg_shap_plot["figure_params"]["empty_cells"][0]]
                        if cfg_ia_values["add"]
                        else self.cfg_shap_plot["figure_params"]["empty_cells"]
                    )
                    fig, axes = self.create_grid(
                        num_rows=figure_params["num_rows"],
                        num_cols=figure_params["num_cols"],
                        figsize=(figure_params["width"], figure_params["height"]),
                        empty_cells=empty_cells,
                    )

                    for (row, col), predictor_combination in positions.items():
                        if predictor_combination in data_current:
                            shap_values = data_current[predictor_combination]
                            self.plot_shap_beeswarm(
                                shap_values=shap_values,
                                ax=axes[row, col],
                                row=row,
                                beeswarm_fontsizes=beeswarm_fontsizes,
                                beeswarm_figure_params=beeswarm_figure_params,
                                num_to_display=num_to_display,
                                cmap=cmap,
                                feature_combo_name_mapping=feature_combo_name_mapping,
                                predictor_combination=predictor_combination,
                                title_line_dct=beeswarm_title_params["line_dct"],
                            )

                            if row == 0:
                                first_row_heading = beeswarm_title_params[
                                    "shap_values"
                                ][col]
                                shap_values_pos_cfg = beeswarm_title_params["position"][
                                    "shap_values"
                                ]

                                axes[row, col].text(
                                    x=shap_values_pos_cfg["x_pos"],
                                    y=shap_values_pos_cfg["y_pos"],
                                    s=first_row_heading,
                                    fontsize=beeswarm_fontsizes["main_title"],
                                    fontweight=beeswarm_title_params["fontweight"],
                                    ha=shap_values_pos_cfg["ha"],
                                    va=shap_values_pos_cfg["va"],
                                    transform=axes[
                                        row, col
                                    ].transAxes,  # Use axes-relative coordinates
                                )
                        else:
                            if ia_data_current:
                                if predictor_combination in ia_data_current:
                                    shap_ia_data = ia_data_current[
                                        predictor_combination
                                    ]
                                    self.plot_shap_beeswarm(
                                        shap_values=shap_ia_data,
                                        ax=axes[row, col],
                                        row=row,
                                        beeswarm_fontsizes=beeswarm_fontsizes,
                                        beeswarm_figure_params=beeswarm_figure_params,
                                        num_to_display=num_to_display,
                                        cmap=cmap,
                                        feature_combo_name_mapping=feature_combo_name_mapping,
                                        predictor_combination=predictor_combination,
                                        ia_values=True,
                                        title_line_dct=beeswarm_title_params[
                                            "line_dct"
                                        ],
                                    )

                                    ia_heading = beeswarm_title_params[
                                        "shap_ia_values"
                                    ][0]
                                    shap_ia_values_pos_cfg = beeswarm_title_params[
                                        "position"
                                    ]["shap_ia_values"]
                                    axes[row, col].text(
                                        x=shap_ia_values_pos_cfg["x_pos"],
                                        y=shap_ia_values_pos_cfg["y_pos"],
                                        s=ia_heading,
                                        fontsize=beeswarm_fontsizes["main_title"],
                                        fontweight=beeswarm_title_params["fontweight"],
                                        ha=shap_ia_values_pos_cfg["ha"],
                                        va=shap_ia_values_pos_cfg["va"],
                                        transform=axes[row, col].transAxes,
                                    )
                            else:
                                print(
                                    f"Predictor combination '{predictor_combination}' not found in shap data or shap ia data"
                                )

                    plt.subplots_adjust(
                        top=beeswarm_subplot_adj["top"],
                        left=beeswarm_subplot_adj["left"],
                        wspace=beeswarm_subplot_adj["wspace"],
                        hspace=beeswarm_subplot_adj["hspace"],
                        right=beeswarm_subplot_adj["right"],
                    )

                    if self.cfg_shap_plot["store_params"]["store"]:
                        self.store_plot(
                            plot_name=f"beeswarm_{crit}_{samples_to_include}_{model}",
                            crit=crit,
                            samples_to_include=samples_to_include,
                            plot_format=self.cfg_shap_plot["store_params"]["format"],
                            dpi=self.cfg_shap_plot["store_params"]["dpi"],
                            model=model,
                            mapping_str=mapping_str,
                        )
                    else:
                        plt.show()

    @staticmethod
    def add_n_to_title(input_string: str, n_samples: int) -> str:
        """
        Formats a string to include the number of samples, with 'n' in italics, and places it on a new line.

        Args:
            input_string (str): The main title or string to format.
            n_samples (int): The number of samples to include in the title.

        Returns:
            str: A formatted string in the format: "<input_string>\n*n* = <formatted_number>",
                 where 'n' is italicized, and large numbers are formatted with commas.
        """
        formatted_n = f"$n$ = {n_samples:,}"
        formatted_title = f"{input_string}\n{formatted_n}"

        return formatted_title

    def plot_shap_beeswarm(
        self,
        shap_values: shap.Explanation,
        ax,
        row,
        beeswarm_fontsizes: dict,
        beeswarm_figure_params: dict,
        num_to_display: int,
        cmap,
        feature_combo_name_mapping,
        predictor_combination,
        title_line_dct: dict = None,
        ia_values: bool = False,
    ):
        """
        Generates a SHAP beeswarm plot for a given predictor combination.

        This method visualizes the SHAP values for a set of features, optionally including interaction values,
        formatted with custom labels, colors, and font sizes. Titles are dynamically adjusted to accommodate
        line breaks and additional information such as the number of samples.

        Method Details:
            - If `ia_values` is `True`, the method calls `get_ia_data` to retrieve interaction SHAP values and formatted data.
            - Titles are dynamically adjusted based on the number of lines, and sample sizes are optionally included.
            - The x-axis limits can be fixed based on the configuration in `beeswarm_figure_params`.

        Args:
        shap_values: A `shap.Explanation` object containing SHAP values, feature names, and feature data.
        ax: The Matplotlib `Axes` object where the plot will be drawn.
        row: The row index of the subplot in the grid. Used for dynamic title adjustments.
        beeswarm_fontsizes: A dictionary specifying font sizes for the title, ticks, and labels.
        beeswarm_figure_params: A dictionary containing figure-level parameters such as max characters for
                                y-ticks, x-axis limits, and whether to fix x-axis limits.
        num_to_display: The maximum number of features to display in the plot.
        cmap: A Matplotlib colormap object used for coloring the beeswarm plot.
        feature_combo_name_mapping: A dictionary mapping predictor combination keys to display names.
                                    Example: {"pl": "Primary Features", "srmc": "Secondary Features"}
        predictor_combination: The key for the predictor combination being plotted.
        title_line_dct: (Optional) A dictionary specifying the maximum number of title lines for each row.
        ia_values: If `True`, interaction SHAP values are plotted. Defaults to `False`.
        """
        plt.sca(ax)

        if ia_values:
            (
                shap_values_to_plot,
                data_to_plot,
                feature_names_formatted,
            ) = self.get_ia_data(shap_values, num_to_display)

        else:
            max_char_on_line = beeswarm_figure_params["max_char_on_line_y_ticks"]
            feature_names_formatted = [
                self.line_break_strings(
                    strng=feature_name,
                    max_char_on_line=max_char_on_line,
                    balance=True,
                    split_strng=None,
                )
                for feature_name in shap_values.feature_names
            ]
            shap_values_to_plot = shap_values.values
            data_to_plot = shap_values.data

        shap.summary_plot(
            shap_values_to_plot,
            data_to_plot,
            feature_names=feature_names_formatted,
            max_display=num_to_display,
            show=False,
            plot_size=None,
            color_bar=False,
            cmap=cmap,
        )

        if ia_values:
            split_strng = self.cfg_shap_plot["ia_values"]["title_split_strng"]  # ":"
        else:
            split_strng = self.cfg_shap_plot["titles"]["split_strng"]  # "+"

        formatted_title = self.line_break_strings(
            strng=feature_combo_name_mapping[predictor_combination],
            max_char_on_line=self.cfg_shap_plot["titles"]["max_char_on_line"],
            split_strng=split_strng,
        )

        if self.cfg_shap_plot["titles"]["add_n"]:
            n_samples = shap_values.shape[0]
            formatted_title = self.add_n_to_title(formatted_title, n_samples)

        max_lines_in_row = title_line_dct[f"row_{row}"]
        current_title_line_count = formatted_title.count("\n") + 1
        y_position = 1.0 + 0.05 * (max_lines_in_row - current_title_line_count)

        ax.set_title(
            formatted_title, fontsize=beeswarm_fontsizes["title"], y=y_position
        )

        ax.tick_params(
            axis="both", which="major", labelsize=beeswarm_fontsizes["tick_params"]
        )
        if ia_values:
            ax.set_xlabel(
                self.cfg_shap_plot["ia_values"]["xlabel"],
                fontsize=beeswarm_fontsizes["x_label"],
            )
        ax.xaxis.label.set_size(beeswarm_fontsizes["x_label"])
        ax.yaxis.label.set_size(beeswarm_fontsizes["y_label"])

        if self.cfg_shap_plot["figure_params"]["fix_x_lim"]:
            ax.set_xlim(tuple(self.cfg_shap_plot["figure_params"]["x_lim"]))

    @staticmethod
    def get_ia_data(
        shap_values: Explanation, num_to_display: int, ia_strng: str = "IA"
    ) -> tuple:
        """
        Extracts and formats the top interaction features and corresponding SHAP values and data for visualization.

        Args:
            shap_values: A SHAP explanation object containing:
                - `values`: SHAP interaction values.
                - `data`: Original feature data corresponding to SHAP values.
            num_to_display: The number of top interaction features to extract.
            ia_strng: IA abbreviation that will be displayed in the plot

        Returns:
            dict: A tuple containing:
                - "feature_names": Formatted feature names for the top interaction features.
                - "shap_values": SHAP values for the top interaction features.
                - "data": Original feature data for the top interaction features.
        """
        feature_names_formatted = [
            f"{ia_strng}{num_ia}" for num_ia in range(1, num_to_display + 1)
        ]

        mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
        sorted_indices = np.argsort(mean_abs_shap)[::-1][:num_to_display]

        shap_values_to_plot = shap_values.values[:, sorted_indices]
        data_to_plot = shap_values.data[:, sorted_indices]

        return shap_values_to_plot, data_to_plot, feature_names_formatted

    def plot_pred_true_parity(
        self,
        sample_data: NestedDict,
        samples_to_include: str,
        crit: str,
        model: str,
        store_plot: bool,
        filename: Optional[str] = None,
    ) -> None:
        """
        Creates a parity plot of predicted vs. true values for all samples.

        This plot helps analyze unexpected patterns in prediction results by comparing predicted
        and true values for various sample subsets. It shows a scatter plot for each sample.

        Args:
            sample_data: Nested dictionary containing predicted and true values for each sample.
            samples_to_include: Label for the subset of samples included in the analysis (e.g., "all").
            crit: Criterion used for analysis (e.g., "wb_state").
            model: Model used for predictions (e.g., "randomforestregressor").
            store_plot: Whether to save the plot to a file.
            filename: Filename for saving the plot, if `store_plot` is True. Defaults to None.
        """
        colors_raw = self.cfg_postprocessing["general"]["global_plot_params"][
            "custom_cmap_colors"
        ]
        cfg_pred_true_plot = self.cfg_postprocessing["sanity_check_pred_vs_true"][
            "plot"
        ]

        width = cfg_pred_true_plot["figure"]["width"]
        height = cfg_pred_true_plot["figure"]["height"]
        fig, ax = plt.subplots(figsize=(width, height))
        samples = list(sample_data.keys())
        num_samples = len(samples)
        colors = [colors_raw[i % len(colors_raw)] for i in range(num_samples)]

        for i, sample_name in enumerate(samples):
            color = colors[i % 10]
            pred_values = sample_data[sample_name]["pred"]
            true_values = sample_data[sample_name]["true"]
            ax.scatter(true_values, pred_values, color=color, label=sample_name)

        min_val = min(
            [min(sample_data[s]["true"] + sample_data[s]["pred"]) for s in samples]
        )
        max_val = max(
            [max(sample_data[s]["true"] + sample_data[s]["pred"]) for s in samples]
        )
        ax.plot([min_val, max_val], [min_val, max_val], "k--", label="r = 1")

        ax.set_xlabel(cfg_pred_true_plot["xlabel"])
        ax.set_ylabel(cfg_pred_true_plot["ylabel"])
        ax.set_title(
            f'{cfg_pred_true_plot["base_title"]} - {samples_to_include} - {crit} - {model}'
        )
        ax.legend()

        if store_plot:
            self.store_plot(
                plot_name=filename,
                crit=crit,
                samples_to_include=samples_to_include,
                model=model,
            )
        else:
            plt.show()

    def line_break_strings(
        self,
        strng: str,
        max_char_on_line: int,
        split_strng: str = None,
        balance: bool = False,
        force_split_strng: bool = False,
    ) -> str:
        """
        Recursively breaks a string into multiple lines to fit within a maximum character limit per line.

        This function is particularly useful for creating a balanced layout in visualizations or textual outputs.
        It provides several customization options to control where and how lines are split.

        Key functionality:
        - Splits the string at the last space within the limit, unless a specific splitting string (`split_strng`) is provided.
        - If `force_split_strng` is True, it ensures splitting at `split_strng` for the last line, even if the line length exceeds the limit.
        - If `balance` is True, the function adjusts splitting positions to minimize the difference in length between lines.
        - The splitting process is performed recursively until the entire string is broken into appropriately sized lines.

        Args:
            strng: The input string to be broken into lines.
            max_char_on_line: Maximum number of characters allowed per line.
            split_strng: A specific string to split on, if provided (e.g., a delimiter or word boundary).
            balance: If True, attempts to balance line lengths for better readability by fine-tuning the split positions.
            force_split_strng: If True, enforces splitting at `split_strng` for the last line, regardless of line length.

        Returns:
            str: The input string formatted with line breaks to fit the specified constraints.
        """
        if len(strng) <= max_char_on_line:
            if force_split_strng and split_strng and split_strng in strng:
                split_pos = strng.rfind(split_strng)
                if split_pos != -1:
                    break_pos = split_pos + len(split_strng)
                    first_line = strng[:break_pos].rstrip()
                    remainder = strng[break_pos:].lstrip()
                    if remainder:
                        return first_line + "\n" + remainder
                    else:
                        return first_line
            return strng  # Return as-is if no forced split is needed

        substring = strng[: max_char_on_line + 1]

        if split_strng:
            if force_split_strng:
                # Always attempt to split at split_strng
                split_pos = substring.rfind(split_strng)
                if split_pos != -1:
                    break_pos = split_pos + len(split_strng)
                else:
                    # If split_strng not found in substring, fallback to space or max_char_on_line
                    split_pos = substring.rfind(" ")
                    break_pos = split_pos if split_pos != -1 else max_char_on_line
            else:
                # If not force_split_strng: prefer split_strng, then space, then max_char_on_line
                split_pos = substring.rfind(split_strng)
                if split_pos != -1:
                    break_pos = split_pos + len(split_strng)
                else:
                    split_pos = substring.rfind(" ")
                    break_pos = split_pos if split_pos != -1 else max_char_on_line
        else:
            # If no split_strng provided, split at the last space or max_char_on_line
            split_pos = substring.rfind(" ")
            break_pos = split_pos if split_pos != -1 else max_char_on_line

        if balance:
            first_line = strng[:break_pos].rstrip()
            remainder = strng[break_pos:].lstrip()

            if remainder:
                current_diff = abs(len(first_line) - len(remainder))

                best_pos = break_pos
                best_diff = current_diff

                search_range = 10
                start_search = max(break_pos - search_range, 1)
                end_search = min(break_pos + search_range, len(strng) - 1)

                for candidate_pos in range(start_search, end_search + 1):
                    if candidate_pos != break_pos and candidate_pos < len(strng):
                        if strng[candidate_pos] == " " or (
                            split_strng
                            and strng[
                                candidate_pos - len(split_strng) + 1 : candidate_pos + 1
                            ]
                            == split_strng
                        ):
                            test_first = strng[:candidate_pos].rstrip()
                            test_rem = strng[candidate_pos:].lstrip()
                            new_diff = abs(len(test_first) - len(test_rem))
                            if new_diff < best_diff:
                                best_diff = new_diff
                                best_pos = candidate_pos

                break_pos = best_pos
                first_line = strng[:break_pos].rstrip()
                remainder = strng[break_pos:].lstrip()
        else:
            first_line = strng[:break_pos].rstrip()
            remainder = strng[break_pos:].lstrip()

        next_force_split_strng = True

        return (
            first_line
            + "\n"
            + self.line_break_strings(
                remainder,
                max_char_on_line,
                split_strng,
                balance,
                force_split_strng=next_force_split_strng,
            )
        )

    def store_plot(
        self,
        plot_name: str,
        plot_format: str = "png",
        dpi: int = 450,
        feature_combination: str = None,
        samples_to_include: str = None,
        crit: str = None,
        model: str = None,
        mapping_str: str = None,
    ):
        """
        Stores plots in a given directory in a certain format and resolution.
            - If we have single plots for specific model/crit/... combinations, we store the plots in the respective dirs
            - If we have a single summarizing plots across everything, we save it in the base folder

        Args:
            plot_name: Name of the plot type, will be used in the filename (e.g. "beeswarm")
            plot_format: Format for the stored image (e.g. "png")
            dpi: Resolution of the plot in dots per inch (e.g., 450)
            feature_combination: e.g., "pl_srmc"
            samples_to_include: e.g., "all"
            crit: e.g., "state_wb"
            model: e.g., "randomforestregressor"
            mapping_str: e.g., "main" or "supp", to differentiate, if necessary
        """
        plot_path = self.create_plot_path(
            feature_combination=feature_combination,
            samples_to_include=samples_to_include,
            crit=crit,
            model=model,
            mapping_str=mapping_str
        )
        os.makedirs(plot_path, exist_ok=True)
        filename = f"{plot_name}.{plot_format}"
        file_path = os.path.join(plot_path, filename)

        print("store plot in:", file_path)
        plt.savefig(
            file_path,
            format=plot_format,
            dpi=dpi,
        )
        plt.close()

    def create_plot_path(
        self,
        samples_to_include: Optional[str] = None,
        crit: Optional[str] = None,
        model: Optional[str] = None,
        feature_combination: Optional[str] = None,
        mapping_str: Optional[str] = None,
    ) -> str:
        """
        Constructs the file path where plots should be stored based on input parameters.

        This method dynamically builds a file path by combining the provided components
        (`samples_to_include`, `crit`, `model`, `feature_combination`). Only non-None components
        are included in the final path. The base directory for the plots is specified by
        `self.plot_base_dir`.

        Args:
            samples_to_include: Specifies the subset of samples to include in the plot.
            crit: Criterion used for generating the plot.
            model: Name of the model associated with the plot.
            feature_combination: The feature configuration used in the plot.

        Returns:
            str: The normalized file path where the plot should be stored.
        """
        path_components = [None, None, None, None, None ]

        for path_idx, var in enumerate(
            [samples_to_include, crit, model, feature_combination, mapping_str]
        ):
            if var is not None:
                path_components[path_idx] = var
        filtered_path_components = [comp for comp in path_components if comp]

        return os.path.normpath(
            os.path.join(self.plot_base_dir, *filtered_path_components)
        )
