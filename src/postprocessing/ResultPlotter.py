import json
import os
from typing import Callable, Union, Collection, Optional, Any

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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

    def apply_figure_style(self) -> None:
        """
        Sets the font and the PDF font type for all following plots, if specified in the cfg.

        With `pdf.fonttype: 42`, text is embedded as TrueType and stays editable in the PDF
        (matplotlib's default, Type 3, does not). Mathtext (e.g., italic r and n) uses the same font.
        """
        style_cfg = self.cfg_postprocessing["general"].get("figure_style", {})
        if not style_cfg.get("apply", False):
            return

        font = style_cfg["font_family"]
        plt.rcParams.update(
            {
                "font.family": "sans-serif",
                "font.sans-serif": [font],
                "pdf.fonttype": style_cfg["pdf_fonttype"],
                "ps.fonttype": style_cfg["pdf_fonttype"],
                "mathtext.fontset": "custom",
                "mathtext.rm": font,
                "mathtext.it": f"{font}:italic",
                "mathtext.bf": f"{font}:bold",
            }
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
        self.apply_figure_style()

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
                    if samples_to_include == "combo":
                        ref_samples_to_include = "all"
                    else:
                        ref_samples_to_include = samples_to_include

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
            if samples_to_include == "combo":
                sublist_sample = "selected" if i == 0 else "all"
            else:
                sublist_sample = samples_to_include

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
    def get_individual_performances(
        feature_combos: list[str],
        samples_to_include: str,
        crit: str,
        base_dir: str = "../results/sig_tests_1312",
    ) -> NestedDict:
        """
        Loads individual Pearson values from CV result JSON files across multiple models and feature combinations.

        This function:
        - Iterates over specified feature combinations.
        - For each feature combo, loads JSON files from both model subdirectories (elasticnet and randomforestregressor).
        - Extracts all Pearson correlation values from each file and aggregates them by model.

        Note: Only used if plot_type is "box" in the cfg (fold-level distributions instead of bars)

        Args:
            feature_combos: List of feature combination names to process.
            samples_to_include: The sample type used in the directory path (e.g., "all", "selected").
            crit: The criterion used in the directory path (e.g., "wb_state").
            base_dir: Root directory where the model folders are located.

        Returns:
            dict[str, dict[str, list[float]]]: Nested dictionary with structure:
                {
                    "<feature_combo>": {
                        "elasticnet": [...pearson values...],
                        "randomforestregressor": [...pearson values...]
                    },
                    ...
                }
        """
        result = {}

        for feature_combo in feature_combos:
            result[feature_combo] = {}
            for model in ["elasticnet", "randomforestregressor"]:
                model_path = os.path.join(
                    base_dir, feature_combo, samples_to_include, crit, model
                )
                if not os.path.exists(model_path):
                    continue

                pearson_values_all = []

                for filename in os.listdir(model_path):
                    if not filename.startswith(
                        "cv_results_rep_"
                    ) or not filename.endswith(".json"):
                        continue

                    file_path = os.path.join(model_path, filename)
                    with open(file_path, "r") as f:
                        data = json.load(f)

                    pearson_values_imp = [
                        imp_vals["pearson"]
                        for fold_vals in data.values()
                        for imp_vals in fold_vals.values()
                        if "pearson" in imp_vals
                    ]

                    pearson_values_all.extend(pearson_values_imp)

                result[feature_combo][model] = pearson_values_all

        return result

    def filter_cv_results_data(
        self,
        cv_results_dct: NestedDict,
        crit: str,
        metric: str,
        samples_to_include: str,
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

        individual_results = self.get_individual_performances(
            feature_combos=list(filtered_results.keys()),
            samples_to_include=samples_to_include,
            crit=crit,
        )

        merged_results = {
            feature: {
                model: {
                    **filtered_results[feature][model],
                    "vals": individual_results[feature][model],
                }
                for model in filtered_results.get(feature, {})
                if model in individual_results.get(feature, {})
            }
            for feature in filtered_results
            if feature in individual_results
        }

        return merged_results

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
        if not y_line_cfg.get("auto", False):
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

        if y_line_cfg.get("auto", False):
            # One vertical line per column at x = 0, spanning the subplots of that column (positions after layout)
            for col_idx, group in enumerate(feature_combinations):
                col_axes = [axes[row_idx, col_idx] for row_idx in range(len(group))]
                x_fig = fig.transFigure.inverted().transform(
                    col_axes[0].transData.transform((0, 0))
                )[0]
                y_bottom = min(ax.get_position().y0 for ax in col_axes)
                y_top = max(ax.get_position().y1 for ax in col_axes)
                fig.add_artist(
                    mlines.Line2D(
                        xdata=[x_fig, x_fig],
                        ydata=[y_bottom, y_top],
                        transform=fig.transFigure,
                        color=y_line_cfg["color"],
                        linestyle=y_line_cfg["linestyle"],
                        linewidth=y_line_cfg["linewidth"],
                    )
                )
            for col_idx, group in enumerate(feature_combinations):
                for row_idx in range(len(group)):
                    # x-axis line starts at zero, together with the vertical line
                    axes[row_idx, col_idx].spines["bottom"].set_bounds(
                        0, figure_params["x_max"]
                    )

        ref_line_cfg = self.cfg_cv_results_plot.get("box_params", {}).get(
            "ref_line", {}
        )
        if (
            self.cfg_cv_results_plot.get("plot_type", "bar") == "box"
            and ref_line_cfg["add"]
            and ref_line_cfg["span"] == "column"
        ):
            # Person-only mean (all datasets) as a line across the whole column, per model (panels b and c)
            m_metric = self.cfg_cv_results_plot["m_metric"]
            for col_idx, group in enumerate(feature_combinations):
                if col_idx == 0:
                    continue
                col_axes = [axes[row_idx, col_idx] for row_idx in range(len(group))]
                # The line lies behind the axes, so their background must be open
                for ax in col_axes:
                    ax.set_facecolor("none")
                y_bottom = min(ax.get_position().y0 for ax in col_axes)
                y_top = max(ax.get_position().y1 for ax in col_axes)
                for model in ref_line_cfg.get("models", models):
                    ref_value = ref_dct[f"pl_{model}"][m_metric]
                    x_fig = fig.transFigure.inverted().transform(
                        col_axes[0].transData.transform((ref_value, 0))
                    )[0]
                    fig.add_artist(
                        mlines.Line2D(
                            xdata=[x_fig, x_fig],
                            ydata=[y_bottom, y_top],
                            transform=fig.transFigure,
                            color=color_dct["pl"],
                            alpha=(
                                color_dct["bar_saturation"][model]
                                if len(ref_line_cfg.get("models", models)) > 1
                                else 1
                            ),
                            linestyle=ref_line_cfg["linestyle"],
                            linewidth=ref_line_cfg["linewidth"],
                            zorder=0,
                        )
                    )

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

            ref_line_cfg = self.cfg_cv_results_plot.get("box_params", {}).get(
                "ref_line", {}
            )
            if self.cfg_cv_results_plot.get(
                "plot_type", "bar"
            ) == "box" and ref_line_cfg["add"]:
                ref_line = mlines.Line2D(
                    [],
                    [],
                    color=color_dct[ref_feature_combo],
                    linestyle=ref_line_cfg["linestyle"],
                    linewidth=ref_line_cfg["linewidth"],
                    label=ref_line_cfg["legend_label"],
                )
                legends_to_plot.append(ref_line)

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
            add_individual_data_points: If individual data points should be added in the plot as a scatter plot.
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

            if self.cfg_cv_results_plot.get("plot_type", "bar") == "box":
                self._plot_fold_distribution(
                    ax=ax,
                    values=data[model_key]["vals"],
                    mean=value,
                    y_center=y_position_to_plot,
                    bar_width=bar_width,
                    color=base_color,
                    alpha=alpha,
                    metric=metric,
                    seed_offset=i,
                    point_color=self.cfg_cv_results_plot["box_params"]
                    .get("point_colors", {})
                    .get(
                        ref_feature_combo
                        if feature_combo == ref_feature_combo
                        else "other"
                    ),
                )
            else:
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

            self._annotate_model_label(
                ax=ax,
                y_pos=y_position_to_plot,
                bar_len=value,
                fontweight="bold",
                model=model,
                model_label_map={"elasticnet": "ENR", "randomforestregressor": "RFR"},
                fontsizes=fontsizes,  # make sure fontsizes["bar_label"] exists or rely on default 12
                x_offset_frac=0.05,  # tweak if you want a tad more / less padding
                outside=self.cfg_cv_results_plot.get("plot_type", "bar") == "box",
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
            add_n=self.cfg_cv_results_plot["figure_params"]["add_n"],
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
            try:  # bit hacky but ok
                increment = pl_margin_dct[f"{feature_combo}_{model}"][
                    f"incremental_{m_metric}"
                ]
                error = data[f"{feature_combo}_{model}"][sd_metric]
            except KeyError:
                print("No all_in data for selected")
                continue

            y_position_to_plot = 0 + (1 - 2 * i) * (bar_width / 2)
            alpha = color_dct["bar_saturation"][model]
            other_feat_color = color_dct["other"]
            ref_color = color_dct[ref_feature_combo]

            if self.cfg_cv_results_plot.get("plot_type", "bar") == "box":
                # Distribution of the combined model; the bar is split at the person-only mean (all datasets)
                model_key = f"{feature_combo}_{model}"
                box_cfg = self.cfg_cv_results_plot["box_params"]
                self._plot_fold_distribution(
                    ax=ax,
                    values=data[model_key]["vals"],
                    mean=data[model_key][m_metric],
                    y_center=y_position_to_plot,
                    bar_width=bar_width,
                    color=other_feat_color,
                    alpha=alpha,
                    metric=metric,
                    seed_offset=i,
                    point_color=box_cfg.get("point_colors", {}).get("other"),
                    # split the bar at the person-only mean (blue / green) or draw it in green only
                    base_value=base_value if box_cfg.get("split_bar", True) else None,
                    base_color=ref_color,
                )
                ref_line_cfg = box_cfg["ref_line"]
                if ref_line_cfg["add"] and ref_line_cfg["span"] == "bar":
                    half_len = bar_width * ref_line_cfg["length_frac"] / 2
                    ax.vlines(
                        x=base_value,
                        ymin=y_position_to_plot - half_len,
                        ymax=y_position_to_plot + half_len,
                        color=ref_color,
                        linewidth=ref_line_cfg["linewidth"],
                        zorder=6,
                    )
                self._annotate_model_label(
                    ax=ax,
                    y_pos=y_position_to_plot,
                    bar_len=base_value,
                    fontweight="bold",
                    model=model,
                    model_label_map={
                        "elasticnet": "ENR",
                        "randomforestregressor": "RFR",
                    },
                    fontsizes=fontsizes,
                    x_offset_frac=0.05,
                    outside=True,
                )

            elif increment > 0:
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

                self._annotate_model_label(
                    ax=ax,
                    y_pos=y_position_to_plot,
                    bar_len=base_value,
                    fontweight="bold",
                    model=model,
                    model_label_map={
                        "elasticnet": "ENR",
                        "randomforestregressor": "RFR",
                    },
                    fontsizes=fontsizes,  # make sure fontsizes["bar_label"] exists or rely on default 12
                    x_offset_frac=0.05,  # tweak if you want a tad more / less padding
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

                self._annotate_model_label(
                    ax=ax,
                    y_pos=y_position_to_plot,
                    bar_len=base_value,
                    fontweight="bold",
                    model=model,
                    model_label_map={
                        "elasticnet": "ENR",
                        "randomforestregressor": "RFR",
                    },
                    fontsizes=fontsizes,  # make sure fontsizes["bar_label"] exists or rely on default 12
                    x_offset_frac=0.05,  # tweak if you want a tad more / less padding
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
                add_n=self.cfg_cv_results_plot["figure_params"]["add_n"]
            )

    @staticmethod
    def _plot_violin_points_vertical(
        ax: Axes,
        values: list[float],
        y_center: float,
        bar_width: float,
        color: str = "black",
        alpha: float = 0.25,
        size: float = 8,
        max_width_frac: float = 0.18,
        bw: float | None = None,
    ) -> None:
        """
        Draws a violin-shaped cloud of points around a horizontal bar by compressing only vertically.

        This method:
        - Keeps x-positions equal to the given values (no horizontal compression).
        - Estimates a 1D KDE over x to get local density per point.
        - Sets each point's vertical half-width as `bar_width * max_width_frac * normalized_density`.
        - Jitters y uniformly within that half-width around `y_center`, yielding a violin envelope that stays
          close to the error bar.

        Args:
            ax: The Matplotlib Axes to plot on.
            values: The individual metric values (x-axis positions).
            y_center: Vertical center of the corresponding bar.
            bar_width: Height of the bar for scaling the violin thickness.
            color: Point color.
            alpha: Point transparency.
            size: Marker size.
            max_width_frac: Maximum vertical half-width as a fraction of `bar_width`.
            bw: Optional KDE bandwidth override; if None, uses a rule-of-thumb.
        """
        x = np.asarray(values, float)

        if x.size == 0:
            return

        if x.size == 1:
            ax.scatter(
                x,
                np.full(1, y_center),
                c=color,
                s=size,
                alpha=alpha,
                linewidths=0,
                zorder=4,
            )
            return

        std = float(np.std(x))
        n = x.size
        if bw is None or not np.isfinite(bw) or bw <= 0:
            bw = 1.06 * std * (n ** (-1 / 5)) if std > 0 else 1e-3

        diffs = (x[:, None] - x[None, :]) / bw
        dens = np.exp(-0.5 * diffs**2).sum(axis=1) / (np.sqrt(2 * np.pi) * bw * n)
        dmax = dens.max() if np.isfinite(dens).all() and dens.max() > 0 else 1.0
        dens = dens / dmax

        half_width = (bar_width * max_width_frac) * dens
        y = y_center + (np.random.rand(n) * 2 - 1) * half_width

        ax.scatter(x, y, c=color, s=size, alpha=alpha, linewidths=0, zorder=4)

    def _plot_fold_distribution(
        self,
        ax: Axes,
        values: list[float],
        mean: float,
        y_center: float,
        bar_width: float,
        color: str,
        alpha: float,
        metric: str,
        seed_offset: int = 0,
        point_color: Optional[str] = None,
        base_value: Optional[float] = None,
        base_color: Optional[str] = None,
    ) -> None:
        """
        Draws the distribution of the outer-fold performance values of one model as a horizontal box plot
        with the individual values overlaid as jittered points. Optionally, a bar from zero to the mean is
        drawn below (`add_bar` in the cfg) and/or the mean is marked (`mean_marker`).
        If `base_value` is given (two- and three-level panels), the bar is split as in the bar version:
        from zero to the reference (person-only) mean in `base_color`, the increment in `color`.

        Box plot definition (matplotlib defaults, set explicitly here):
        - centre line: median
        - box: 25th to 75th percentile
        - whiskers: most extreme value within 1.5 x IQR from the box
        - no separate outlier symbols, because all individual values are drawn as points

        Args:
            ax: The Matplotlib Axes to plot on.
            values: Outer-fold performance values of the model (10 reps x 10 outer folds x 5 imputations = 500).
            mean: Mean performance across all values, drawn as a marker.
            y_center: Vertical center of the box.
            bar_width: Vertical space available per model, used to scale the box height and the jitter.
            color: Color of the bar or, without a bar, of the box face.
            alpha: Transparency of the bar or box face (differentiates ENR and RFR).
            metric: The metric plotted. The fold-level values are Pearson correlations only.
            seed_offset: Added to the seed so that ENR and RFR get different, but reproducible, jitter.
            point_color: Fill color of the points; defaults to `point_color` in the cfg.
            base_value: Mean of the reference model (person-only predictors), splits the bar.
            base_color: Color of the bar segment up to `base_value`.
        """
        if metric != "pearson":
            raise ValueError(
                "plot_type 'box' needs fold-level values, which are only loaded for metric 'pearson'"
            )

        box_cfg = self.cfg_cv_results_plot["box_params"]
        x = np.asarray(values, float)

        # Set the final x-limits already here, because the model labels are positioned relative to them
        figure_params = self.cfg_cv_results_plot["figure_params"]
        ax.set_xlim(figure_params["x_min"], figure_params["x_max"])

        # Optional bar from zero to the mean below the box plot (the box face is then left open)
        add_bar = box_cfg.get("add_bar", False)
        if add_bar:
            bar_kwargs = {
                "height": bar_width * box_cfg.get("bar_height_frac", 1),
                "alpha": alpha * box_cfg.get("bar_alpha_factor", 1),
                "align": figure_params["bar_align"],
                "edgecolor": None,
                "zorder": 2,
            }
            if base_value is None:
                ax.barh(y_center, mean, color=color, **bar_kwargs)
            elif mean > base_value:
                ax.barh(y_center, base_value, color=base_color, **bar_kwargs)
                ax.barh(
                    y_center,
                    mean - base_value,
                    left=base_value,
                    color=color,
                    **bar_kwargs,
                )
            else:
                # No increment over the reference: only the reference color, as in the bar version
                ax.barh(y_center, mean, color=base_color, **bar_kwargs)

        box_height = bar_width * box_cfg["box_height_frac"]

        ax.boxplot(
            [x],
            positions=[y_center],
            vert=False,
            widths=box_height,
            whis=box_cfg["whis"],
            showfliers=False,
            patch_artist=True,
            boxprops={
                "facecolor": "none" if add_bar else color,
                "alpha": 1 if add_bar else alpha,
                "edgecolor": "black",
                "linewidth": box_cfg["linewidth"],
            },
            medianprops={"color": "black", "linewidth": box_cfg["linewidth"]},
            whiskerprops={"color": "black", "linewidth": box_cfg["linewidth"]},
            capprops={"color": "black", "linewidth": box_cfg["linewidth"]},
            manage_ticks=False,
            zorder=5,  # above the points, so the black box outline stays visible
        )

        rng = np.random.default_rng(box_cfg["seed"] + seed_offset)
        jitter = (rng.random(x.size) * 2 - 1) * box_height / 2 * box_cfg["jitter_frac"]
        ax.scatter(
            x,
            y_center + jitter,
            color=point_color or box_cfg["point_color"],
            s=box_cfg["point_size"],
            alpha=box_cfg["point_alpha"],
            edgecolors=box_cfg.get("point_edgecolor", "none"),
            linewidths=box_cfg.get("point_edgewidth", 0),
            zorder=4,
        )

        mean_cfg = box_cfg["mean_marker"]
        if mean_cfg["add"]:
            ax.scatter(
                [mean],
                [y_center],
                marker=mean_cfg["marker"],
                s=mean_cfg["size"],
                facecolor=mean_cfg["facecolor"],
                edgecolor="black",
                linewidths=box_cfg["linewidth"],
                zorder=5,
            )

    @staticmethod
    def _annotate_model_label(
        ax: Axes,
        *,
        y_pos: float,
        bar_len: float,
        fontweight: str = "normal",
        model: str,
        model_label_map: dict[str, str],
        fontsizes: dict[str, int],
        x_offset_frac: float = 0.01,
        extra_text_kwargs: dict[str, Any] | None = None,
        outside: bool = False,
    ) -> None:
        """
        Adds a model label to the left edge of a horizontal bar in a bar plot.

        The method determines whether the bar is positive or negative and places the label accordingly,
        offset slightly from the y-axis using a fraction of the axis range. The label text is retrieved
        from a mapping dictionary and styled with optional parameters.
        If `outside` is True (box plots), the label is placed left of the y-axis instead, so that it
        does not overlap the distribution.

        Args:
            ax: The matplotlib Axes object where the text will be added.
            y_pos: Vertical position for label placement (center of the bar).
            bar_len: Length of the bar to determine alignment direction.
            fontweight: Font weight for the label (e.g., "normal", "bold").
            model: Model key to annotate (used for label lookup).
            model_label_map: Mapping from model keys to human-readable labels.
            fontsizes: Dictionary containing font sizes (expects key "bar_label").
            x_offset_frac: Horizontal offset from axis, as a fraction of the x-axis range.
            extra_text_kwargs: Additional keyword arguments passed to `ax.text`.
        """
        if extra_text_kwargs is None:
            extra_text_kwargs = {}

        # Left padding in data units
        x_span = ax.get_xlim()[1] - ax.get_xlim()[0]
        x_offset = x_offset_frac * x_span

        # Text just right of the y-axis (or just left for negative bars or if outside)
        if outside:
            x_text = ax.get_xlim()[0] - x_offset
            ha = "right"
        elif bar_len >= 0:
            x_text = ax.get_xlim()[0] + x_offset
            ha = "left"
        else:
            x_text = ax.get_xlim()[0] - x_offset
            ha = "right"

        ax.text(
            x_text,
            y_pos,
            model_label_map.get(model, model),
            va="center",
            ha=ha,
            fontsize=fontsizes.get("bar_label", 12),  # bump default up a bit
            fontweight=fontweight,
            color="black",
            clip_on=False,  # avoid being clipped by axes limits
            **extra_text_kwargs,
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
        add_n: bool = True,
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
            add_n: If the sample sizes are added in the y-axes ticks
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

        if add_n:
            feature_combo_formatted = self.add_n_to_title(
                feature_combo_formatted, n_samples
            )

        axes_cfg = self.cfg_cv_results_plot["format_bar_plot"]["axes"]
        ax.set_yticklabels(
            [feature_combo_formatted],
            fontsize=fontsizes["tick_params"],
            horizontalalignment=axes_cfg["ylabels"]["hor_align"],
        )

        ax.spines["left"].set_visible(False)
        xticks_cfg = self.cfg_cv_results_plot["format_bar_plot"]["axes"]["xticks"]
        xticks_decimals = xticks_cfg["decimals"]
        leading_zero = xticks_cfg.get("leading_zero", False)
        ax.xaxis.set_major_formatter(
            FuncFormatter(
                lambda x, _: self.format_metric(x, xticks_decimals, leading_zero)
            )
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
    def format_metric(
        metric: float, xticks_decimals: int = 2, leading_zero: bool = False
    ) -> Union[str, int]:
        """
        Formats a given metric according to APA style (default) or journal style:
            - APA: No leading zero for correlations or R² values (e.g., .85 instead of 0.85).
            - Journal (leading_zero=True): Leading zero kept (e.g., 0.85).
            - Rounded to the specified number of decimal places.

        Args:
            metric: The metric to be formatted (e.g., a correlation or R² value).
            xticks_decimals: The number of decimal places for rounding. Defaults to 2.
            leading_zero: If True, keeps the leading zero (journal style). Defaults to False (APA).

        Returns:
            Union[str, int]: A string representation of the formatted metric or 0 for exact zero.
        """
        if not isinstance(metric, (float, int)):
            raise ValueError("The metric must be a numeric value.")

        if metric == 0:
            return 0

        if leading_zero:
            return f"{metric:.{xticks_decimals}f}"

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
        self.apply_figure_style()

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
                        # Without "bottom" in the cfg, matplotlib's default is used
                        bottom=beeswarm_subplot_adj.get("bottom"),
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

        # shap draws the points with a fixed marker size, which is too large at the final print size
        point_size = beeswarm_figure_params.get("point_size")
        if point_size:
            for collection in ax.collections:
                collection.set_sizes([point_size])

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
        group_by: str = "sample",
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
            group_by: Aggregation level the predictions are grouped by (e.g., dataset of country)
        """
        cfg_pred_true_plot = self.cfg_postprocessing["sanity_check_pred_vs_true"][
            "plot"
        ]

        width = cfg_pred_true_plot["figure"]["width"]
        height = cfg_pred_true_plot["figure"]["height"]
        fig, ax = plt.subplots(figsize=(width, height))
        samples = list(sample_data.keys())

        if group_by == "country_year":
            for i, sample_name in enumerate(samples):
                pred_values = sample_data[sample_name]["pred"]
                true_values = sample_data[sample_name]["true"]
                ax.scatter(
                    true_values,
                    pred_values,
                    label=sample_name,
                )

        if group_by == "sample":
            for i, sample_name in enumerate(samples):
                pred_values = sample_data[sample_name]["pred"]
                true_values = sample_data[sample_name]["true"]
                ax.scatter(
                    true_values,
                    pred_values,
                    label=sample_name,
                )

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
            mapping_str=mapping_str,
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
            mapping_str: Further identifier, optional.

        Returns:
            str: The normalized file path where the plot should be stored.
        """
        path_components = [None, None, None, None, None]

        for path_idx, var in enumerate(
            [samples_to_include, crit, model, feature_combination, mapping_str]
        ):
            if var is not None:
                path_components[path_idx] = var
        filtered_path_components = [comp for comp in path_components if comp]

        return os.path.normpath(
            os.path.join(self.plot_base_dir, *filtered_path_components)
        )

    def plot_shap_sanity_histograms(
        self,
        comparison_df: pd.DataFrame,
        shap_values_by_feature: dict[str, dict[str, np.ndarray]],
        feature_combination: str,
        num_features: int,
        rank_by: str = "selected",
        bin_width: float = 0.01,
        feature_label_col: str = "Feature",
        alpha: float = 0.45,
        figsize: tuple[float, float] | None = None,
        store: bool = False,
        plot_name: str | None = None,
        plot_format: str = "png",
        dpi: int = 450,
        crit: str | None = None,
        model: str | None = None,
    ) -> None:
        """
        Plots signed SHAP-value histograms for selected-vs-all sanity checks.


        Note: Not used in final code

        For a given feature combination, the function selects the top `num_features`
        according to the mean absolute SHAP value in either the selected or all analysis.
        For each selected feature, it overlays semi-transparent histograms of signed SHAP
        values from the selected and all analyses.

        Args:
            comparison_df: DataFrame created in shap_sanity_checks. Must contain
                Feature, Mean_abs_SHAP_selected, Mean_abs_SHAP_all, and
                Pearson_r_selected_vs_all.
            shap_values_by_feature: Dictionary with raw signed SHAP values:
                {
                    feature_name: {
                        "selected": np.ndarray,
                        "all": np.ndarray,
                    }
                }
            feature_combination: Feature-combination label, e.g., "pl_srmc".
            num_features: Number of top features to plot.
            rank_by: Whether to rank top features by "selected" or "all" mean absolute SHAP.
            bin_width: Width of the SHAP-value histogram bins.
            feature_label_col: Column to use for feature labels. Defaults to "Feature".
            alpha: Histogram transparency.
            figsize: Optional figure size. If None, determined from num_features.
            store: Whether to save the plot.
            plot_name: Optional filename stem. If None, generated automatically.
            plot_format: File format for saving.
            dpi: Plot resolution.
            crit: Optional criterion for structured storage path.
            model: Optional model for structured storage path.
        """
        if rank_by not in ["selected", "all"]:
            raise ValueError("rank_by must be either 'selected' or 'all'.")

        importance_col = f"Mean_abs_SHAP_{rank_by}"

        required_cols = {
            "Feature",
            "Mean_abs_SHAP_selected",
            "Mean_abs_SHAP_all",
            "Pearson_r_selected_vs_all",
        }
        missing_cols = required_cols.difference(comparison_df.columns)
        if missing_cols:
            raise KeyError(f"comparison_df is missing columns: {missing_cols}")

        if importance_col not in comparison_df.columns:
            raise KeyError(f"Column '{importance_col}' not found in comparison_df.")

        label_col = (
            feature_label_col
            if feature_label_col in comparison_df.columns
            else "Feature"
        )

        top_features_df = (
            comparison_df.dropna(subset=[importance_col])
            .sort_values(by=importance_col, ascending=False)
            .head(num_features)
            .reset_index(drop=True)
        )

        if top_features_df.empty:
            print(f"No features available for {feature_combination}.")
            return

        missing_features = [
            feature
            for feature in top_features_df["Feature"]
            if feature not in shap_values_by_feature
        ]
        if missing_features:
            raise KeyError(
                f"Raw SHAP values missing for features: {missing_features[:10]}"
            )

        if figsize is None:
            figsize = (10, max(2.2 * len(top_features_df), 4))

        fig, axes = plt.subplots(
            nrows=len(top_features_df),
            ncols=1,
            figsize=figsize,
            sharex=True,
            squeeze=False,
        )
        axes = axes.flatten()

        for ax, (_, row) in zip(axes, top_features_df.iterrows()):
            feature = row["Feature"]
            feature_label = str(row[label_col])

            selected_vals = np.asarray(
                shap_values_by_feature[feature]["selected"],
                dtype=float,
            )
            all_vals = np.asarray(
                shap_values_by_feature[feature]["all"],
                dtype=float,
            )

            selected_vals = selected_vals[np.isfinite(selected_vals)]
            all_vals = all_vals[np.isfinite(all_vals)]

            if selected_vals.size == 0 and all_vals.size == 0:
                continue

            min_val = np.nanmin(
                [
                    selected_vals.min() if selected_vals.size > 0 else np.nan,
                    all_vals.min() if all_vals.size > 0 else np.nan,
                ]
            )
            max_val = np.nanmax(
                [
                    selected_vals.max() if selected_vals.size > 0 else np.nan,
                    all_vals.max() if all_vals.size > 0 else np.nan,
                ]
            )

            if min_val == max_val:
                min_val -= bin_width
                max_val += bin_width

            bins = np.arange(min_val, max_val + bin_width, bin_width)
            if len(bins) < 2:
                bins = np.array([min_val, max_val + bin_width])

            ax.hist(
                all_vals,
                bins=bins,
                alpha=alpha,
                label="All analysis",
            )
            ax.hist(
                selected_vals,
                bins=bins,
                alpha=alpha,
                label="Selected analysis",
            )

            ax.axvline(np.mean(all_vals), linestyle=":", linewidth=1)
            ax.axvline(np.mean(selected_vals), linestyle="--", linewidth=1)

            formatted_label = self.line_break_strings(
                strng=feature_label,
                max_char_on_line=45,
                balance=True,
            )

            ax.text(
                0.01,
                0.92,
                formatted_label,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=10,
                fontweight="bold",
            )

            ax.text(
                0.99,
                0.92,
                (
                    f"r = {row['Pearson_r_selected_vs_all']:.2f}\n"
                    f"|SHAP| {rank_by} = {row[importance_col]:.3f}"
                ),
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=9,
            )

            ax.set_ylabel("Count")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        axes[-1].set_xlabel("Signed SHAP value")

        mean_corr = pd.to_numeric(
            comparison_df["Pearson_r_selected_vs_all"],
            errors="coerce",
        ).mean()

        selected_patch = mpatches.Patch(alpha=alpha, label="Selected analysis")
        all_patch = mpatches.Patch(alpha=alpha, label="All analysis")
        mean_corr_line = mlines.Line2D(
            [],
            [],
            linestyle="none",
            label=f"Mean r = {mean_corr:.2f}",
        )

        fig.legend(
            handles=[selected_patch, all_patch, mean_corr_line],
            loc="lower center",
            ncol=3,
            frameon=False,
        )

        fig.suptitle(
            f"{feature_combination}: selected vs. all SHAP distributions",
            fontsize=12,
            fontweight="bold",
        )

        plt.tight_layout(rect=(0, 0.06, 1, 0.96))

        if store:
            if plot_name is None:
                plot_name = (
                    f"shap_sanity_histograms_{feature_combination}"
                    f"_top{num_features}_by_{rank_by}"
                )

            self.store_plot(
                plot_name=plot_name,
                plot_format=plot_format,
                dpi=dpi,
                feature_combination=feature_combination,
                samples_to_include="selected_vs_all",
                crit=crit,
                model=model,
            )
        else:
            plt.show()

    def plot_country_level_variability(
        self,
        df: pd.DataFrame,
        samples_to_include: str,
        columns: list[str] | None = None,
        country_id_col: str | None = None,
        scale_boundaries: dict[str, dict[str, Any]] | None = None,
        country_var_groups: dict[str, list[str]] | None = None,
        n_cols: int = 5,
        panel_width: float = 3.3,
        panel_height: float = 4,
        x_padding_frac: float = 0.05,
        store: bool = False,
        plot_name_base: str = "country_level_variability",
        plot_format: str = "png",
        dpi: int = 600,
    ) -> None:
        """
        Creates country-level variability plots.

        If `reference_data.use` is true in the config:
        - violin plots are based on all available reference data
        - boxplots are based on all available reference data
        - all reference data points can be plotted
        - analytic-data values are highlighted directly from `df_plot[col]`
        - the in-panel annotation only shows the percentage of reference range
          covered by the analytic-data range
        """
        country_vars_cfg = self.cfg_postprocessing["create_descriptives"][
            "country_vars"
        ]

        store = country_vars_cfg.get("store_plot", store)
        plot_name_base = country_vars_cfg.get("plot_filename", plot_name_base)
        n_cols = country_vars_cfg.get("n_cols", n_cols)
        n_rows = country_vars_cfg.get("n_rows", None)
        single_plot_source = country_vars_cfg.get("single_plot_source", "countries")

        reference_cfg = country_vars_cfg.get("reference_data", {})
        reference_df = None

        if reference_cfg.get("use", False):
            reference_df = self._load_country_reference_data(reference_cfg)

        if columns is None:
            columns = list(df.columns)

        if country_id_col is not None:
            columns = [col for col in columns if col != country_id_col]

        if scale_boundaries is None:
            scale_boundaries = country_vars_cfg.get("scale_boundaries", {})

        if country_var_groups is None:
            country_var_groups = {
                "psycho_political": country_vars_cfg.get("psycho_political", []),
                "socio_economic": country_vars_cfg.get("socio_economic", []),
                "health_environmental": country_vars_cfg.get(
                    "health_environmental", []
                ),
            }

        cols_to_use = columns.copy()

        if country_id_col is not None and country_id_col in df.columns:
            df_country = (
                df[[country_id_col] + cols_to_use]
                .drop_duplicates(subset=[country_id_col])
                .copy()
            )
        else:
            df_country = df[cols_to_use].drop_duplicates().copy()

        df_person = df[cols_to_use].copy()

        plot_top_n = country_vars_cfg.get("plot_top_n", None)

        if plot_top_n is not None:
            top_n_vars = country_vars_cfg.get("top_n_to_plot", [])

            if not top_n_vars:
                raise KeyError(
                    "`plot_top_n` is set, but `top_n_to_plot` is missing or empty."
                )

            vars_to_plot = top_n_vars[:plot_top_n]

            if len(top_n_vars) < plot_top_n:
                print(
                    f"Warning: `plot_top_n` is set to {plot_top_n}, but only "
                    f"{len(top_n_vars)} variables are listed in `top_n_to_plot`."
                )

            selected_cols, missing_vars = self._resolve_country_plot_columns(
                available_columns=cols_to_use,
                requested_variables=vars_to_plot,
            )

            if missing_vars:
                print(
                    "Warning: The following configured variables could not be matched "
                    f"to analytic dataframe columns and will be skipped: {missing_vars}"
                )

            if not selected_cols:
                raise ValueError(
                    "No columns from `top_n_to_plot` could be matched to the analytic dataframe."
                )

            if single_plot_source == "countries":
                df_plot = df_country
                distribution_label = (
                    f"Across countries: Top {len(selected_cols)} selected variables"
                )
                violin_color = "#4C78A8"
                show_points = True

            elif single_plot_source == "persons":
                df_plot = df_person
                distribution_label = (
                    f"Across persons: Top {len(selected_cols)} selected variables"
                )
                violin_color = "#F58518"
                show_points = False

            else:
                raise ValueError(
                    "`single_plot_source` must be either 'countries' or 'persons'."
                )

            self._plot_country_variability_overview(
                df_plot=df_plot,
                columns=selected_cols,
                scale_boundaries=scale_boundaries,
                n_cols=n_cols,
                n_rows=n_rows,
                panel_width=panel_width,
                panel_height=panel_height,
                x_padding_frac=x_padding_frac,
                distribution_label=distribution_label,
                violin_color=violin_color,
                show_points=show_points,
                store=store,
                plot_name=plot_name_base,
                plot_format=plot_format,
                dpi=dpi,
                reference_df=reference_df,
                reference_cfg=reference_cfg,
            )

            return

        grouped_columns = self._get_country_var_grouped_columns(
            columns=cols_to_use,
            country_var_groups=country_var_groups,
        )

        for group_name, group_cols in grouped_columns.items():
            if not group_cols:
                continue

            self._plot_country_variability_overview(
                df_plot=df_country,
                columns=group_cols,
                scale_boundaries=scale_boundaries,
                n_cols=n_cols,
                n_rows=n_rows,
                panel_width=panel_width,
                panel_height=panel_height,
                x_padding_frac=x_padding_frac,
                distribution_label=f"Across countries: {group_name.replace('_', ' ').title()}",
                violin_color="#4C78A8",
                show_points=True,
                store=store,
                plot_name=f"{plot_name_base}_{group_name}_countries",
                plot_format=plot_format,
                dpi=dpi,
                reference_df=reference_df,
                reference_cfg=reference_cfg,
            )

            self._plot_country_variability_overview(
                df_plot=df_person,
                columns=group_cols,
                scale_boundaries=scale_boundaries,
                n_cols=n_cols,
                n_rows=n_rows,
                panel_width=panel_width,
                panel_height=panel_height,
                x_padding_frac=x_padding_frac,
                distribution_label=f"Across persons: {group_name.replace('_', ' ').title()}",
                violin_color="#F58518",
                show_points=False,
                store=store,
                plot_name=f"{plot_name_base}_{group_name}_persons",
                plot_format=plot_format,
                dpi=dpi,
                reference_df=reference_df,
                reference_cfg=reference_cfg,
            )

    def _plot_country_variability_overview(
        self,
        df_plot: pd.DataFrame,
        columns: list[str],
        scale_boundaries: dict[str, dict[str, Any]],
        n_cols: int,
        n_rows: int | None,
        panel_width: float,
        panel_height: float,
        x_padding_frac: float,
        distribution_label: str,
        violin_color: str,
        show_points: bool,
        store: bool,
        plot_name: str,
        plot_format: str,
        dpi: int,
        reference_df: pd.DataFrame | None = None,
        reference_cfg: dict[str, Any] | None = None,
    ) -> None:
        """
        Helper that creates one multi-panel overview figure.

        If reference_df is provided:
        - reference values are used for the violin and boxplot
        - analytic values from df_plot[col] are plotted as highlighted points
        - the highlighted orange range bar is computed from the same visible highlighted values
        - the only in-panel annotation is the percentage of reference range covered
          by the analytic-data range
        """
        n_vars = len(columns)

        if n_vars == 0:
            print(f"No variables available for plot '{plot_name}'.")
            return

        if n_rows is None:
            n_rows = int(np.ceil(n_vars / n_cols))

        if n_rows * n_cols < n_vars:
            raise ValueError(
                f"Configured grid too small for plot '{plot_name}': "
                f"{n_rows} rows x {n_cols} cols = {n_rows * n_cols} panels, "
                f"but {n_vars} variables need to be plotted."
            )

        if reference_cfg is None:
            reference_cfg = {}

        country_vars_cfg = self.cfg_postprocessing["create_descriptives"][
            "country_vars"
        ]

        show_legend = country_vars_cfg.get("show_legend", False)
        show_suptitle = country_vars_cfg.get("show_suptitle", False)

        plot_percentile = country_vars_cfg.get("plot_percentile", False)
        percentile = country_vars_cfg.get("percentile", 0.90)

        show_configured_min_max_points_default = country_vars_cfg.get(
            "show_configured_min_max_points",
            False,
        )

        use_reference_data = reference_df is not None and reference_cfg.get(
            "use", False
        )

        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(n_cols * panel_width, n_rows * panel_height),
            squeeze=False,
        )

        axes_flat = axes.flatten()
        rng = np.random.default_rng(42)

        any_configured_extrema = False
        any_reference_distribution = False
        any_highlighted_points = False

        all_points_color = reference_cfg.get("all_points_color", "black")
        all_points_alpha = reference_cfg.get("all_points_alpha", 0.18)
        all_points_size = reference_cfg.get("all_points_size", 10)
        all_points_jitter_width = reference_cfg.get("all_points_jitter_width", 0.13)

        highlight_color = reference_cfg.get("highlight_color", "#D55E00")
        highlight_alpha = reference_cfg.get("highlight_alpha", 0.95)
        highlight_size = reference_cfg.get("highlight_size", 32)
        highlight_jitter_width = reference_cfg.get("highlight_jitter_width", 0.17)

        highlight_range_y = reference_cfg.get(
            "highlight_range_y",
            country_vars_cfg.get("highlight_range_y", -0.50),
        )
        highlight_range_linewidth = reference_cfg.get(
            "highlight_range_linewidth",
            country_vars_cfg.get("highlight_range_linewidth", 4),
        )
        highlight_range_alpha = reference_cfg.get(
            "highlight_range_alpha",
            country_vars_cfg.get("highlight_range_alpha", 0.65),
        )

        y_min = country_vars_cfg.get("country_variability_y_min", -0.58)
        y_max = country_vars_cfg.get("country_variability_y_max", 0.60)

        coverage_label = country_vars_cfg.get("range_coverage_label", "range covered")
        coverage_decimals = country_vars_cfg.get("range_coverage_decimals", 1)
        coverage_fontsize = country_vars_cfg.get("range_coverage_fontsize", 9.5)

        subplot_title_fontsize = country_vars_cfg.get("subplot_title_fontsize", 12)
        subplot_title_fontweight = country_vars_cfg.get(
            "subplot_title_fontweight",
            "bold",
        )
        subplot_title_pad = country_vars_cfg.get("subplot_title_pad", 14)

        row_hspace = country_vars_cfg.get("row_hspace", 0.65)
        tight_layout_h_pad = country_vars_cfg.get("tight_layout_h_pad", 3.0)
        tight_layout_w_pad = country_vars_cfg.get("tight_layout_w_pad", 1.5)

        display_name_mapping = country_vars_cfg.get("display_name_mapping", {})

        for ax_idx, col in enumerate(columns):
            ax = axes_flat[ax_idx]

            reference_var_df = None
            reference_col = None
            values_source = "analytic"

            if use_reference_data:
                reference_col = self._resolve_reference_variable_column(
                    plot_col=col,
                    reference_columns=list(reference_df.columns),
                    variable_mapping=reference_cfg.get("variable_mapping", {}),
                )

                if reference_col is not None:
                    reference_var_df = self._prepare_reference_variable_df(
                        reference_df=reference_df,
                        reference_col=reference_col,
                        reference_cfg=reference_cfg,
                    )

                    values = pd.to_numeric(
                        reference_var_df[reference_col],
                        errors="coerce",
                    ).dropna()

                    if not values.empty:
                        values_source = "reference"
                        any_reference_distribution = True
                    else:
                        values = pd.to_numeric(df_plot[col], errors="coerce").dropna()
                        print(
                            f"Warning: Reference column '{reference_col}' for '{col}' "
                            "has no valid numeric values. Falling back to analytic data."
                        )
                else:
                    values = pd.to_numeric(df_plot[col], errors="coerce").dropna()
                    print(
                        f"Warning: Could not find reference-data column for '{col}'. "
                        "Falling back to analytic data."
                    )
            else:
                values = pd.to_numeric(df_plot[col], errors="coerce").dropna()

            if values.empty:
                ax.set_axis_off()
                continue

            scale_info = self._get_scale_info_for_variable(
                col=col,
                scale_boundaries=scale_boundaries,
            )

            background_values_arr = self._finite_numeric_array(values)
            background_min = float(np.nanmin(background_values_arr))
            background_max = float(np.nanmax(background_values_arr))

            if plot_percentile:
                x_min, x_max = self._get_percentile_axis_limits(
                    values=background_values_arr,
                    percentile=percentile,
                    padding_frac=x_padding_frac,
                )

            else:
                if scale_info is not None:
                    configured_x_min = scale_info.get("x_min", None)
                    configured_x_max = scale_info.get("x_max", None)

                    if configured_x_min is not None and configured_x_max is not None:
                        x_min = float(configured_x_min)
                        x_max = float(configured_x_max)
                    else:
                        x_min, x_max = self._add_axis_padding(
                            x_min=background_min,
                            x_max=background_max,
                            padding_frac=x_padding_frac,
                        )
                else:
                    x_min, x_max = self._add_axis_padding(
                        x_min=background_min,
                        x_max=background_max,
                        padding_frac=x_padding_frac,
                    )

            if values_source == "reference":
                ax.hlines(
                    y=0,
                    xmin=background_min,
                    xmax=background_max,
                    color="lightgray",
                    linewidth=8,
                    zorder=0,
                )
            else:
                ax.hlines(
                    y=0,
                    xmin=background_min,
                    xmax=background_max,
                    color="gray",
                    linewidth=8,
                    zorder=1,
                )

            violin_parts = ax.violinplot(
                dataset=[background_values_arr],
                positions=[0],
                vert=False,
                widths=0.75,
                showmeans=False,
                showmedians=False,
                showextrema=False,
            )

            for body in violin_parts["bodies"]:
                body.set_facecolor(violin_color)
                body.set_edgecolor("black")
                body.set_alpha(0.45)
                body.set_linewidth(0.8)

            ax.boxplot(
                [background_values_arr],
                positions=[0],
                vert=False,
                widths=0.22,
                showfliers=False,
                patch_artist=True,
                boxprops=dict(facecolor="white", edgecolor="black", linewidth=1.0),
                medianprops=dict(color="black", linewidth=1.2),
                whiskerprops=dict(color="black", linewidth=1.0),
                capprops=dict(color="black", linewidth=1.0),
            )

            highlighted_values_all_arr = self._finite_numeric_array(df_plot[col])
            highlighted_values_visible_arr = self._values_within_xlim(
                values=highlighted_values_all_arr,
                x_min=x_min,
                x_max=x_max,
            )

            background_values_visible_arr = self._values_within_xlim(
                values=background_values_arr,
                x_min=x_min,
                x_max=x_max,
            )

            coverage_pct = self._compute_range_coverage_pct(
                numerator_values=highlighted_values_visible_arr,
                denominator_values=background_values_visible_arr,
            )

            coverage_text = self._format_range_coverage_text(
                coverage_pct=coverage_pct,
                label=coverage_label,
                decimals=coverage_decimals,
            )

            if values_source == "reference" and reference_var_df is not None:
                numeric_reference_values = pd.to_numeric(
                    reference_var_df[reference_col],
                    errors="coerce",
                )

                valid_reference_values_arr = self._finite_numeric_array(
                    numeric_reference_values
                )
                valid_reference_values_visible_arr = self._values_within_xlim(
                    values=valid_reference_values_arr,
                    x_min=x_min,
                    x_max=x_max,
                )

                if (
                    reference_cfg.get("plot_all_points", True)
                    and valid_reference_values_visible_arr.size > 0
                ):
                    all_jitter = rng.uniform(
                        -all_points_jitter_width,
                        all_points_jitter_width,
                        size=len(valid_reference_values_visible_arr),
                    )

                    ax.scatter(
                        valid_reference_values_visible_arr,
                        all_jitter,
                        s=all_points_size,
                        alpha=all_points_alpha,
                        color=all_points_color,
                        linewidths=0,
                        zorder=3,
                    )

                if (
                    reference_cfg.get("highlight_own_points", True)
                    and highlighted_values_visible_arr.size > 0
                ):
                    highlight_jitter = rng.uniform(
                        -highlight_jitter_width,
                        highlight_jitter_width,
                        size=len(highlighted_values_visible_arr),
                    )

                    ax.scatter(
                        highlighted_values_visible_arr,
                        highlight_jitter,
                        s=highlight_size,
                        alpha=highlight_alpha,
                        color=highlight_color,
                        edgecolor="black",
                        linewidths=0.5,
                        zorder=6,
                    )

                    any_highlighted_points = True

                    ax.hlines(
                        y=highlight_range_y,
                        xmin=float(np.nanmin(highlighted_values_visible_arr)),
                        xmax=float(np.nanmax(highlighted_values_visible_arr)),
                        color=highlight_color,
                        linewidth=highlight_range_linewidth,
                        alpha=highlight_range_alpha,
                        zorder=5,
                    )

            elif show_points:
                analytic_values_visible_arr = self._values_within_xlim(
                    values=background_values_arr,
                    x_min=x_min,
                    x_max=x_max,
                )

                if analytic_values_visible_arr.size > 0:
                    jitter = rng.uniform(
                        -0.08,
                        0.08,
                        size=len(analytic_values_visible_arr),
                    )

                    ax.scatter(
                        analytic_values_visible_arr,
                        jitter,
                        s=12,
                        alpha=0.45,
                        color="black",
                        linewidths=0,
                        zorder=3,
                    )

            if scale_info is not None:
                show_configured_min_max_points = scale_info.get(
                    "show_min_max_points",
                    show_configured_min_max_points_default,
                )

                if show_configured_min_max_points:
                    any_configured_extrema = True

                    self._add_configured_min_max_points(
                        ax=ax,
                        scale_info=scale_info,
                        y_pos=highlight_range_y,
                    )

            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_yticks([])

            pretty_label = self._format_country_variable_label(
                col,
                display_name_mapping=display_name_mapping,
            )
            pretty_label = self.line_break_strings(
                strng=pretty_label,
                max_char_on_line=24,
                balance=True,
            )

            ax.set_title(
                pretty_label,
                fontsize=subplot_title_fontsize,
                fontweight=subplot_title_fontweight,
                pad=subplot_title_pad,
            )

            ax.text(
                0.02,
                0.95,
                coverage_text,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=coverage_fontsize,
            )

            for spine in ["top", "right", "left"]:
                ax.spines[spine].set_visible(False)

            ax.tick_params(axis="x", labelsize=8)

        for ax in axes_flat[n_vars:]:
            ax.set_axis_off()

        if show_legend:
            legend_handles = []

            if any_reference_distribution:
                legend_handles.append(
                    mlines.Line2D(
                        [],
                        [],
                        color="lightgray",
                        linewidth=8,
                        label="All available min–max range",
                    )
                )
            else:
                legend_handles.append(
                    mlines.Line2D(
                        [],
                        [],
                        color="gray",
                        linewidth=8,
                        label="Observed analytic min–max range",
                    )
                )

            legend_handles.extend(
                [
                    mpatches.Patch(
                        facecolor=violin_color,
                        edgecolor="black",
                        alpha=0.45,
                        label=(
                            "Reference distribution"
                            if any_reference_distribution
                            else distribution_label
                        ),
                    ),
                    mpatches.Patch(
                        facecolor="white",
                        edgecolor="black",
                        label="Boxplot based on plotted distribution",
                    ),
                ]
            )

            if any_reference_distribution and reference_cfg.get(
                "plot_all_points", True
            ):
                legend_handles.append(
                    mlines.Line2D(
                        [],
                        [],
                        color=all_points_color,
                        marker="o",
                        linestyle="None",
                        markersize=5,
                        alpha=all_points_alpha,
                        label=reference_cfg.get(
                            "all_points_label",
                            "All available data points",
                        ),
                    )
                )

            if any_highlighted_points:
                legend_handles.append(
                    mlines.Line2D(
                        [],
                        [],
                        color=highlight_color,
                        marker="o",
                        markeredgecolor="black",
                        linestyle="None",
                        markersize=6,
                        alpha=highlight_alpha,
                        label=reference_cfg.get(
                            "highlight_label",
                            "Analytic sample values",
                        ),
                    )
                )

            if any_configured_extrema:
                legend_handles.extend(
                    [
                        mlines.Line2D(
                            [],
                            [],
                            color="#B22222",
                            marker="v",
                            linestyle="None",
                            markersize=7,
                            label="Configured minimum country",
                        ),
                        mlines.Line2D(
                            [],
                            [],
                            color="#006400",
                            marker="^",
                            linestyle="None",
                            markersize=7,
                            label="Configured maximum country",
                        ),
                    ]
                )

            fig.legend(
                handles=legend_handles,
                loc="lower center",
                bbox_to_anchor=(0.5, -0.01),
                ncol=3,
                frameon=False,
                fontsize=10,
            )

        title = distribution_label
        if any_reference_distribution:
            title = (
                f"{distribution_label} — boxplots based on all available reference data"
            )

        if show_suptitle:
            fig.suptitle(
                title,
                fontsize=14,
                fontweight="bold",
                y=0.995,
            )

        if show_legend and show_suptitle:
            layout_rect = (0, 0.06, 1, 0.94)
        elif show_legend:
            layout_rect = (0, 0.06, 1, 0.98)
        elif show_suptitle:
            layout_rect = (0, 0, 1, 0.94)
        else:
            layout_rect = (0, 0, 1, 0.98)

        fig.tight_layout(
            rect=layout_rect,
            h_pad=tight_layout_h_pad,
            w_pad=tight_layout_w_pad,
        )
        fig.subplots_adjust(hspace=row_hspace)

        if store:
            self.store_plot(
                plot_name=plot_name,
                plot_format=plot_format,
                dpi=dpi,
            )
        else:
            plt.show()

    @staticmethod
    def _resolve_country_plot_columns(
        available_columns: list[str],
        requested_variables: list[str],
    ) -> tuple[list[str], list[str]]:
        """
        Matches requested variable names from config to actual dataframe columns.

        Supports both prefixed and unprefixed names, e.g.
        - "youth_unemployment"
        - "mac_youth_unemployment"

        Returns:
            matched_columns: list of dataframe columns that were found
            missing_variables: list of requested variables that could not be matched
        """
        matched_columns = []
        missing_variables = []

        for requested_var in requested_variables:
            matched_col = None

            for col in available_columns:
                candidate_keys = ResultPlotter._get_country_var_candidate_keys(col)

                if requested_var in candidate_keys:
                    matched_col = col
                    break

            if matched_col is None:
                missing_variables.append(requested_var)
            else:
                matched_columns.append(matched_col)

        matched_columns = list(dict.fromkeys(matched_columns))

        return matched_columns, missing_variables

    @staticmethod
    def _get_scale_info_for_variable(
        col: str,
        scale_boundaries: dict[str, dict[str, Any]],
    ) -> dict[str, Any] | None:
        """
        Retrieves configured scale information for a variable.

        The config entry may contain:
        - unit
        - x_min
        - x_max
        - min_country
        - max_country
        - show_min_max_points

        Unlike earlier versions, this method does not require min_country/max_country.
        This allows using scale info only for axis limits.
        """
        candidate_keys = ResultPlotter._get_country_var_candidate_keys(col)

        for key in candidate_keys:
            if key in scale_boundaries:
                return scale_boundaries[key]

        return None

    @staticmethod
    def _get_country_var_candidate_keys(col: str) -> list[str]:
        """
        Creates candidate lookup keys for config dictionaries from a possibly prefixed column name.
        """
        candidate_keys = [col]

        for prefix in ["mac_", "pl_", "srmc_", "sens_", "crit_"]:
            if col.startswith(prefix):
                candidate_keys.append(col[len(prefix) :])

        if "_" in col:
            candidate_keys.append(col.partition("_")[2])

        return list(dict.fromkeys(candidate_keys))

    @staticmethod
    def _add_axis_padding(
        x_min: float,
        x_max: float,
        padding_frac: float,
    ) -> tuple[float, float]:
        """
        Adds symmetric padding to an x-axis range.
        """
        if x_min == x_max:
            return x_min - 0.5, x_max + 0.5

        x_range = x_max - x_min
        padding = x_range * padding_frac

        return x_min - padding, x_max + padding

    @staticmethod
    def _add_configured_min_max_points(
        ax: Axes,
        scale_info: dict[str, Any],
        y_pos: float,
    ) -> None:
        """
        Adds configured global min/max country points in distinct colors.
        """
        point_specs = {
            "min_country": {
                "marker": "v",
                "color": "#B22222",
                "label_prefix": "min",
                "y_offset": 0.00,
                "text_offset": -8,
                "va": "top",
            },
            "max_country": {
                "marker": "^",
                "color": "#006400",
                "label_prefix": "max",
                "y_offset": 0.00,
                "text_offset": -8,
                "va": "top",
            },
        }

        for key, spec in point_specs.items():
            country_info = scale_info.get(key)

            if not country_info:
                continue

            value = country_info.get("value")
            country_name = country_info.get("name", "")

            if value is None or pd.isna(value):
                continue

            try:
                value = float(value)
            except (TypeError, ValueError):
                continue

            y_current = y_pos + spec["y_offset"]

            ax.scatter(
                value,
                y_current,
                s=48,
                marker=spec["marker"],
                facecolor=spec["color"],
                edgecolor="black",
                linewidth=0.8,
                zorder=6,
            )

            label = f"{spec['label_prefix']}: {country_name}"

            ax.annotate(
                label,
                xy=(value, y_current),
                xytext=(0, spec["text_offset"]),
                textcoords="offset points",
                ha="center",
                va=spec["va"],
                fontsize=7,
                rotation=0,
                zorder=7,
            )

    @staticmethod
    def _format_country_variable_label(
        col: str,
        display_name_mapping: dict[str, str] | None = None,
    ) -> str:
        """
        Creates a readable variable label from a column name.

        Uses special display labels for the top-6 country variables and allows
        optional config overrides via `country_vars.display_name_mapping`.
        """
        default_display_name_mapping = {
            "inflation_rate": "Inflation rate",
            "gdp_growth": "GDP growth",
            "adult_unemployment": "Unemployment rate",
            "unemployment_rate": "Unemployment rate",
            "youth_unemployment": "Youth unemployment rate",
            "youth_unemployment_rate": "Youth unemployment rate",
            "poverty_rate": "Poverty rate",
            "shortterm_orientation": "Short- vs. long-term\norientation",
        }

        if display_name_mapping is None:
            display_name_mapping = {}

        display_name_mapping = {
            **default_display_name_mapping,
            **display_name_mapping,
        }

        candidate_keys = ResultPlotter._get_country_var_candidate_keys(col)

        for candidate in candidate_keys:
            if candidate in display_name_mapping:
                return display_name_mapping[candidate]

        label = col

        for prefix in ["mac_", "pl_", "srmc_", "sens_", "crit_"]:
            if label.startswith(prefix):
                label = label[len(prefix) :]
                break

        return label.replace("_", " ").title()

    @staticmethod
    def _get_country_var_grouped_columns(
        columns: list[str],
        country_var_groups: dict[str, list[str]],
    ) -> dict[str, list[str]]:
        """
        Matches dataframe columns to higher-level country-variable groups.

        Handles both prefixed and unprefixed column names, e.g.,
        'mac_power_distance' and 'power_distance'.
        """
        grouped_columns = {group_name: [] for group_name in country_var_groups}

        for col in columns:
            candidate_keys = ResultPlotter._get_country_var_candidate_keys(col)

            for group_name, group_vars in country_var_groups.items():
                if any(candidate in group_vars for candidate in candidate_keys):
                    grouped_columns[group_name].append(col)
                    break

        return grouped_columns

    @staticmethod
    def _load_country_reference_data(reference_cfg: dict[str, Any]) -> pd.DataFrame:
        """
        Loads the reference country-level data from an Excel file.

        The file is expected to contain one row per country-year or country observation.
        """
        path = reference_cfg.get("path")
        sheet_name = reference_cfg.get("sheet_name", 0)

        if not path:
            raise ValueError(
                "`reference_data.path` must be provided when reference_data.use is true."
            )

        if not os.path.exists(path):
            raise FileNotFoundError(f"Reference data file not found: {path}")

        reference_df = pd.read_excel(path, sheet_name=sheet_name)
        reference_df.columns = [str(col).strip() for col in reference_df.columns]

        return reference_df

    @staticmethod
    def _resolve_reference_variable_column(
        plot_col: str,
        reference_columns: list[str],
        variable_mapping: dict[str, str],
    ) -> str | None:
        """
        Resolves the plotting variable column to the corresponding reference-data column.

        Supports:
        - direct matches
        - prefixed analytic columns, e.g. mac_youth_unemployment
        - config-based mappings, e.g. youth_unemployment -> youth_unemployment_rate
        """
        default_mapping = {
            "adult_unemployment": "unemployment_rate",
            "youth_unemployment": "youth_unemployment_rate",
        }

        variable_mapping = {
            **default_mapping,
            **variable_mapping,
        }

        reference_columns_set = set(reference_columns)
        candidate_keys = ResultPlotter._get_country_var_candidate_keys(plot_col)

        for candidate in candidate_keys:
            mapped_col = variable_mapping.get(candidate)

            if mapped_col in reference_columns_set:
                return mapped_col

        for candidate in candidate_keys:
            if candidate in reference_columns_set:
                return candidate

        return None

    @staticmethod
    def _prepare_reference_variable_df(
        reference_df: pd.DataFrame,
        reference_col: str,
        reference_cfg: dict[str, Any],
    ) -> pd.DataFrame:
        """
        Prepares reference data for one variable.

        Simplified behavior:
        - uses all available numeric values from the reference column
        - ignores country and year columns
        - no country matching
        - no year filtering
        """
        df_var = reference_df[[reference_col]].copy()
        df_var[reference_col] = pd.to_numeric(df_var[reference_col], errors="coerce")
        df_var = df_var.dropna(subset=[reference_col])

        return df_var

    @staticmethod
    def _finite_numeric_array(values: Any) -> np.ndarray:
        """
        Converts values to a finite numeric numpy array.
        """
        values_arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(
            dtype=float
        )
        values_arr = values_arr[np.isfinite(values_arr)]

        return values_arr

    @staticmethod
    def _values_within_xlim(
        values: np.ndarray,
        x_min: float,
        x_max: float,
    ) -> np.ndarray:
        """
        Returns values inside the visible x-axis range.

        This ensures that the highlighted range bar aligns with the actually visible
        highlighted points when configured x-axis limits clip some values.
        """
        if values.size == 0:
            return values

        return values[(values >= x_min) & (values <= x_max)]

    @staticmethod
    def _compute_range_coverage_pct(
        numerator_values: np.ndarray,
        denominator_values: np.ndarray,
    ) -> float:
        """
        Computes the percentage of the denominator range covered by the numerator range.

        Used for:
            analytic-data range / all-world reference-data range
        """
        numerator_values = numerator_values[np.isfinite(numerator_values)]
        denominator_values = denominator_values[np.isfinite(denominator_values)]

        if numerator_values.size == 0 or denominator_values.size == 0:
            return np.nan

        numerator_range = float(
            np.nanmax(numerator_values) - np.nanmin(numerator_values)
        )
        denominator_range = float(
            np.nanmax(denominator_values) - np.nanmin(denominator_values)
        )

        if denominator_range <= 0:
            return np.nan

        return 100 * numerator_range / denominator_range

    @staticmethod
    def _format_range_coverage_text(
        coverage_pct: float,
        label: str = "range covered",
        decimals: int = 1,
    ) -> str:
        """
        Formats the in-panel range-coverage annotation.
        """
        if coverage_pct is None or not np.isfinite(coverage_pct):
            return f"{label}: n/a"

        return f"{label}: {coverage_pct:.{decimals}f}%"

    @staticmethod
    def _get_percentile_axis_limits(
        values: np.ndarray,
        percentile: float,
        padding_frac: float,
    ) -> tuple[float, float]:
        """
        Computes symmetric percentile-based x-axis limits.

        Example:
            percentile = 0.90 uses the 5th and 95th percentiles.
            percentile = 0.80 uses the 10th and 90th percentiles.
        """
        values = values[np.isfinite(values)]

        if values.size == 0:
            return np.nan, np.nan

        percentile = float(percentile)

        # Allow both 0.90 and 90 as input
        if percentile > 1:
            percentile = percentile / 100

        if percentile <= 0 or percentile > 1:
            raise ValueError(
                "`percentile` must be in the interval (0, 1] or given as a percentage, "
                "for example 0.90 or 90."
            )

        lower_q = (1 - percentile) / 2
        upper_q = 1 - lower_q

        x_min = float(np.nanquantile(values, lower_q))
        x_max = float(np.nanquantile(values, upper_q))

        return ResultPlotter._add_axis_padding(
            x_min=x_min,
            x_max=x_max,
            padding_frac=padding_frac,
        )
