"""
Viz Builder: generates matplotlib/seaborn charts from analyst output.

Responsibilities:
  - Read suggested_charts from Analyst output (list of dicts with type/x/y/title)
  - Render each chart: histogram, bar, scatter, line, heatmap
  - Apply a consistent dark professional style across all figures
  - Save each figure to output/figures/<ISO-timestamp>_<slug>.png
  - Skip gracefully with a warning if a column is missing or a spec is invalid
  - Return a dict: figure_paths, charts_rendered, charts_skipped

Works with any tabular dataset — no column names are hardcoded.
No Claude API calls — pure matplotlib/seaborn.
"""

from __future__ import annotations

import os
import re
import warnings
from datetime import datetime, timezone

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

OUTPUT_DIR = "output/figures"

# Dark professional palette
BG_COLOR      = "#0d1117"
SURFACE_COLOR = "#161b22"
TEXT_COLOR    = "#e6edf3"
GRID_COLOR    = "#30363d"
SPINE_COLOR   = "#30363d"

PALETTE = ["#4C8BB5", "#E8834E", "#5CB85C", "#D9534F", "#9B59B6", "#1ABC9C", "#F0C27F"]

# Suppress seaborn / matplotlib deprecation noise
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

BAR_MAX_CATEGORIES = 25   # truncate bar charts beyond this many categories


def _apply_global_style() -> None:
    mpl.rcParams.update({
        "figure.facecolor":    BG_COLOR,
        "axes.facecolor":      SURFACE_COLOR,
        "axes.edgecolor":      SPINE_COLOR,
        "axes.labelcolor":     TEXT_COLOR,
        "axes.titlecolor":     TEXT_COLOR,
        "axes.grid":           True,
        "grid.color":          GRID_COLOR,
        "grid.linestyle":      "--",
        "grid.alpha":          0.5,
        "xtick.color":         TEXT_COLOR,
        "ytick.color":         TEXT_COLOR,
        "text.color":          TEXT_COLOR,
        "legend.facecolor":    SURFACE_COLOR,
        "legend.edgecolor":    SPINE_COLOR,
        "font.family":         "sans-serif",
        "font.size":           11,
        "axes.titlesize":      13,
        "axes.labelsize":      11,
        "figure.dpi":          100,
    })


_apply_global_style()


def _label(col: str) -> str:
    """Turn a snake_case column name into a readable axis label."""
    return col.replace("_", " ").title()


def _slugify(text: str) -> str:
    """Convert a chart title into a safe filename stem (max 60 chars)."""
    slug = text.lower().strip()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"[\s-]+", "_", slug)
    return slug[:60].strip("_")


def _style_ax(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(SPINE_COLOR)
    ax.spines["bottom"].set_color(SPINE_COLOR)
    ax.tick_params(colors=TEXT_COLOR, length=4)


class VizBuilder:
    def __init__(self, output_dir: str = OUTPUT_DIR):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def run(self, wrangler_output: dict, insights: dict) -> dict:
        """
        Render and save all charts suggested by the Analyst.

        Args:
            wrangler_output: Output from DataWrangler.run() — must contain "dataframe".
            insights:        Output from Analyst.run() — must contain "suggested_charts".

        Returns:
            dict with keys:
              - "figure_paths":    list[str] — paths to every saved PNG
              - "charts_rendered": int
              - "charts_skipped":  int
        """
        df: pd.DataFrame = wrangler_output["dataframe"]
        suggested: list[dict] = insights.get("suggested_charts", [])

        figure_paths: list[str] = []
        charts_skipped = 0

        for spec in suggested:
            try:
                path = self._render(df, spec)
                figure_paths.append(path)
            except (KeyError, ValueError, TypeError) as exc:
                title = spec.get("title", str(spec))
                print(f"[VizBuilder] Skipped '{title}': {exc}")
                charts_skipped += 1

        return {
            "figure_paths":    figure_paths,
            "charts_rendered": len(figure_paths),
            "charts_skipped":  charts_skipped,
        }

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def _render(self, df: pd.DataFrame, spec: dict) -> str:
        """
        Validate a chart spec, call the right renderer, save and return the path.
        Raises ValueError for missing columns or unknown chart types so run() can skip.
        """
        chart_type = str(spec.get("type", "")).lower().strip()
        x_col  = spec.get("x") or None
        y_col  = spec.get("y") or None
        title  = spec.get("title", f"{chart_type} chart")

        # Validate columns exist before rendering
        for col in (x_col, y_col):
            if col is not None and col not in df.columns:
                raise ValueError(f"column '{col}' not found in dataframe")

        renderers = {
            "histogram": self._plot_histogram,
            "bar":       self._plot_bar,
            "scatter":   self._plot_scatter,
            "line":      self._plot_line,
            "heatmap":   self._plot_heatmap,
        }
        renderer = renderers.get(chart_type)
        if renderer is None:
            raise ValueError(f"unknown chart type '{chart_type}'")

        fig = renderer(df, x_col, y_col, title)
        return self._save(fig, _slugify(title))

    # ------------------------------------------------------------------
    # Renderers
    # ------------------------------------------------------------------

    def _plot_histogram(
        self, df: pd.DataFrame, x_col: str, y_col: str | None, title: str
    ) -> plt.Figure:
        if x_col is None:
            raise ValueError("histogram requires an x column")

        fig, ax = plt.subplots(figsize=(10, 6))
        data = df[x_col].dropna()

        sns.histplot(
            data, ax=ax,
            color=PALETTE[0], edgecolor=BG_COLOR,
            bins="auto", alpha=0.9,
            kde=True, kde_kws={"color": PALETTE[1], "linewidth": 2},
        )

        ax.set_title(title, pad=14)
        ax.set_xlabel(_label(x_col))
        ax.set_ylabel("Count")
        _style_ax(ax)
        fig.tight_layout()
        return fig

    def _plot_bar(
        self, df: pd.DataFrame, x_col: str, y_col: str | None, title: str
    ) -> plt.Figure:
        if x_col is None or y_col is None:
            raise ValueError("bar chart requires both x and y columns")

        # Aggregate if multiple rows per category (e.g. multi-year data)
        agg = df.groupby(x_col)[y_col].mean().sort_values(ascending=False)

        if len(agg) > BAR_MAX_CATEGORIES:
            agg = agg.head(BAR_MAX_CATEGORIES)
            title = f"{title} (top {BAR_MAX_CATEGORIES})"

        fig, ax = plt.subplots(figsize=(11, max(6, len(agg) * 0.35)))

        colors = [PALETTE[i % len(PALETTE)] for i in range(len(agg))]
        ax.barh(
            y=agg.index[::-1],
            width=agg.values[::-1],
            color=colors[::-1],
            edgecolor="none",
            height=0.7,
        )

        ax.set_title(title, pad=14)
        ax.set_xlabel(_label(y_col))
        ax.set_ylabel(_label(x_col))
        _style_ax(ax)
        ax.grid(axis="y", visible=False)
        fig.tight_layout()
        return fig

    def _plot_scatter(
        self, df: pd.DataFrame, x_col: str, y_col: str | None, title: str
    ) -> plt.Figure:
        if x_col is None or y_col is None:
            raise ValueError("scatter chart requires both x and y columns")

        # For multi-year datasets aggregate per entity to avoid overplotting
        cat_cols = df.select_dtypes(exclude="number").columns.tolist()
        if cat_cols:
            plot_df = df.groupby(cat_cols[0])[[x_col, y_col]].mean().reset_index()
        else:
            plot_df = df[[x_col, y_col]].dropna()

        fig, ax = plt.subplots(figsize=(10, 7))

        ax.scatter(
            plot_df[x_col], plot_df[y_col],
            color=PALETTE[0], alpha=0.75,
            edgecolors=TEXT_COLOR, linewidths=0.4, s=65,
        )

        # Trend line
        mask = plot_df[[x_col, y_col]].notna().all(axis=1)
        xv = plot_df.loc[mask, x_col].values
        yv = plot_df.loc[mask, y_col].values
        if len(xv) > 2:
            coeffs = np.polyfit(xv, yv, 1)
            x_line = np.linspace(xv.min(), xv.max(), 200)
            ax.plot(
                x_line, np.polyval(coeffs, x_line),
                color=PALETTE[1], linewidth=2, linestyle="--", alpha=0.85,
                label=f"trend (slope {coeffs[0]:+.3g})",
            )
            ax.legend(fontsize=9)

        # Label outliers (top/bottom 3 by y value) if a category column exists
        if cat_cols and cat_cols[0] in plot_df.columns:
            label_col = cat_cols[0]
            extremes = pd.concat([
                plot_df.nlargest(3, y_col),
                plot_df.nsmallest(3, y_col),
            ]).drop_duplicates()
            for _, row in extremes.iterrows():
                ax.annotate(
                    str(row[label_col]),
                    xy=(row[x_col], row[y_col]),
                    xytext=(5, 3), textcoords="offset points",
                    fontsize=7.5, color=TEXT_COLOR, alpha=0.8,
                )

        ax.set_title(title, pad=14)
        ax.set_xlabel(_label(x_col))
        ax.set_ylabel(_label(y_col))
        _style_ax(ax)
        fig.tight_layout()
        return fig

    def _plot_line(
        self, df: pd.DataFrame, x_col: str, y_col: str | None, title: str
    ) -> plt.Figure:
        if x_col is None or y_col is None:
            raise ValueError("line chart requires both x and y columns")

        # Aggregate numeric y by x (e.g. mean across all counties per year)
        agg = df.groupby(x_col)[y_col].agg(["mean", "std"]).reset_index()
        agg = agg.sort_values(x_col)

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(
            agg[x_col], agg["mean"],
            color=PALETTE[0], linewidth=2.5,
            marker="o", markersize=7, markerfacecolor=PALETTE[1],
            markeredgecolor=BG_COLOR, markeredgewidth=1.5,
        )

        # Confidence band (±1 std)
        ax.fill_between(
            agg[x_col],
            agg["mean"] - agg["std"],
            agg["mean"] + agg["std"],
            color=PALETTE[0], alpha=0.15,
            label="±1 std across groups",
        )

        ax.set_title(title, pad=14)
        ax.set_xlabel(_label(x_col))
        ax.set_ylabel(_label(y_col))
        ax.legend(fontsize=9)
        _style_ax(ax)
        fig.tight_layout()
        return fig

    def _plot_heatmap(
        self, df: pd.DataFrame, x_col: str | None, y_col: str | None, title: str
    ) -> plt.Figure:
        numeric_df = df.select_dtypes(include="number")
        if numeric_df.shape[1] < 2:
            raise ValueError("heatmap needs at least 2 numeric columns")

        corr = numeric_df.corr()
        labels = [c.replace("_", "\n") for c in corr.columns]

        n = len(corr)
        fig, ax = plt.subplots(figsize=(max(10, n * 0.9), max(8, n * 0.75)))

        cmap = sns.diverging_palette(220, 20, as_cmap=True)
        sns.heatmap(
            corr, ax=ax,
            cmap=cmap, center=0, vmin=-1, vmax=1,
            annot=True, fmt=".2f", annot_kws={"size": 8.5},
            linewidths=0.5, linecolor=BG_COLOR,
            xticklabels=labels, yticklabels=labels,
            square=True,
            cbar_kws={"shrink": 0.8, "label": "Pearson r"},
        )

        ax.set_title(title, pad=16)
        ax.tick_params(axis="x", labelsize=8, rotation=0)
        ax.tick_params(axis="y", labelsize=8, rotation=0)
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def _save(self, fig: plt.Figure, slug: str) -> str:
        """Save figure with an ISO-UTC timestamp prefix. Returns the file path."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        filename = f"{timestamp}_{slug}.png"
        path = os.path.join(self.output_dir, filename)
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        return path
