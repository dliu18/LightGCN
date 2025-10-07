#!/usr/bin/env python3
from __future__ import annotations
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Iterable, DefaultDict
from collections import defaultdict
import argparse

import matplotlib.pyplot as plt

# TensorBoard loader
try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except Exception:
    print("[FATAL] tensorboard not available. pip install tensorboard", file=sys.stderr)
    raise

# -----------------------------
# Defaults (CLI-overridable)
# -----------------------------
DEFAULT_BASELINE = "pop_reg"
DEFAULT_BASELINE_DISPLAY = "Popularity Regularization"  # shown in legend
DEFAULT_PARAM_DISPLAY = "Regularization Strength"       # x-axis label
DEFAULT_MODELS = None                                   # e.g., ["lgn", "mf"]; if None, autodiscover
DEFAULT_DATASETS = None                                  # e.g., ["gowalla", "yelp2018", "amazon-book"]; if None, autodiscover per model
DEFAULT_XLOG = False                                    # log-scale x-axis when parameters are numeric and > 0

DATASET_ORDER = ["gowalla", "yelp2018", "amazon-book"]  # fixed left->right order

DATASET_PRETTY = {
    "gowalla": "Gowalla",
    "yelp2018": "Yelp2018",
    "amazon-book": "Amazon-Book",
}

# -----------------------------
# Paths
# -----------------------------
SCRIPT_DIR   = Path(__file__).resolve().parent
RUNS_ROOT    = (SCRIPT_DIR / "../../runs/www").resolve()

# Write to <project>/outputs/final_eval/www
PROJECT_ROOT = RUNS_ROOT.parents[2] if len(RUNS_ROOT.parents) >= 3 else SCRIPT_DIR
OUT_DIR      = PROJECT_ROOT / "outputs" / "final_eval" / "www"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Metric tags & helpers
# -----------------------------
RECALL_DIR_NAME  = "Recall@[20, 10000]"       # under Test/
RECALL_EVENTS_20 = "20"                       # subfolder
RECALL_TAG       = "Test/Recall__20__10000_"  # in those events

POB_TAG          = "Test/Popularity_Opportunity_Bias_20"  # top-level trial events

def _last_scalar_from_dir(events_dir: Path, tag: str) -> Optional[float]:
    """Use EventAccumulator on 'events_dir' to fetch last scalar for 'tag'."""
    if not events_dir.exists():
        return None
    try:
        ea = EventAccumulator(str(events_dir))
        ea.Reload()
        scalars = ea.Tags().get("scalars", []) or []
        if tag not in scalars:
            return None
        vals = ea.Scalars(tag)
        if not vals:
            return None
        return float(vals[-1].value)
    except Exception:
        return None

def _read_recall_value(trial_dir: Path) -> Optional[float]:
    """Recall@20 is under trial_dir/Test/Recall@[20, 10000]/20/ with tag RECALL_TAG."""
    evdir = trial_dir / "Test" / RECALL_DIR_NAME / RECALL_EVENTS_20
    return _last_scalar_from_dir(evdir, RECALL_TAG)

def _read_pob_value(trial_dir: Path) -> Optional[float]:
    """POB lives in the trial_dir top-level events."""
    return _last_scalar_from_dir(trial_dir, POB_TAG)

# -----------------------------
# Trial discovery (assume ONE parameter layer for baselines)
# -----------------------------
def _iter_vanilla_or_ours_trials(variant_root: Path) -> List[Path]:
    """
    Vanilla/Ours: variant_root/<trial_int>/...
    If no integer children, fall back to variant_root as a single 'trial'.
    """
    trials: List[Path] = []
    if not variant_root.exists():
        return trials
    try:
        for child in variant_root.iterdir():
            if child.is_dir() and child.name.isdigit():
                trials.append(child)
    except Exception:
        pass
    if not trials:
        trials = [variant_root]
    return sorted(trials, key=lambda p: (p.name.isdigit(), int(p.name) if p.name.isdigit() else p.name))

def _iter_baseline_param_trials(baseline_root: Path) -> Dict[str, List[Path]]:
    """
    ONE parameter level:
      baseline_root/<param>/<trial_int>/
    Return: param_str -> list of trial_dirs
    """
    param_to_trials: Dict[str, List[Path]] = defaultdict(list)
    if not baseline_root.exists():
        return param_to_trials

    try:
        for param_dir in baseline_root.iterdir():
            if not param_dir.is_dir():
                continue
            try:
                for trial_dir in param_dir.iterdir():
                    if trial_dir.is_dir() and trial_dir.name.isdigit():
                        param_to_trials[param_dir.name].append(trial_dir)
            except Exception:
                continue
    except Exception:
        pass

    for k in list(param_to_trials.keys()):
        pts = param_to_trials[k]
        param_to_trials[k] = sorted(pts, key=lambda p: (p.name.isdigit(), int(p.name) if p.name.isdigit() else p.name))
    return param_to_trials

# -----------------------------
# Aggregation helpers
# -----------------------------
def _mean_metric_over_trials(trials: List[Path], reader) -> Optional[float]:
    vals: List[float] = []
    for tdir in trials:
        v = reader(tdir)
        if v is not None:
            vals.append(float(v))
    if not vals:
        return None
    return sum(vals) / len(vals)

def _mean_metric_for_variant(variant_root: Path, reader) -> Optional[float]:
    trials = _iter_vanilla_or_ours_trials(variant_root)
    return _mean_metric_over_trials(trials, reader)

def _collect_baseline_param_curve(baseline_root: Path, reader) -> Tuple[List[str], List[float]]:
    """
    Return (param_labels, mean_values) for baseline.
    Each mean is across all trials for that parameter.
    Parameters sorted numerically when possible, else lexicographically.
    """
    param_trials = _iter_baseline_param_trials(baseline_root)
    if not param_trials:
        return [], []

    def try_float(s: str):
        try: return float(s)
        except Exception: return None

    numeric_pairs = []
    nonnum = []
    for p in param_trials.keys():
        f = try_float(p)
        (numeric_pairs if f is not None else nonnum).append((f, p) if f is not None else p)

    sorted_params = [p for _, p in sorted(numeric_pairs, key=lambda t: t[0])] + sorted(nonnum)

    xs: List[str] = []
    ys: List[float] = []
    for p in sorted_params:
        m = _mean_metric_over_trials(param_trials[p], reader)
        if m is None:
            continue
        xs.append(p)
        ys.append(m)
    return xs, ys

# -----------------------------
# Plotting (generic for any metric)
# -----------------------------
def make_figure_for_metric(
    baseline: str,
    baseline_display: str,
    param_display: str,
    models: List[str],
    per_model_datasets: Dict[str, List[str]],
    metric_name: str,                # "Recall@20" or "Popularity Bias"
    reader,                          # _read_recall_value or _read_pob_value
    y_label: str,
    color_baseline: str = "#1f77b4",
    xlog: bool = False,
) -> Path:
    # Fonts & layout
    plt.rcParams.update({
        "font.family": "Nimbus Roman",
        "font.serif": ["Nimbus Roman", "Times New Roman", "Times", "DejaVu Serif", "Serif"],
        "axes.grid": True, "grid.linestyle": "--", "grid.alpha": 0.35,
        "font.size": 24, "axes.titlesize": 28, "axes.labelsize": 26,
        "legend.fontsize": 26,
        "xtick.labelsize": 15, "ytick.labelsize": 15,
    })

    nrows = len(models)
    ncols = len(DATASET_ORDER)  # fixed left->right order

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5.2*ncols, 4.2*nrows), squeeze=False)
    plt.subplots_adjust(hspace=0.35, wspace=0.3)

    # Build legend once from first populated subplot
    handles_for_legend = None
    labels_for_legend = None

    for r, model in enumerate(models):
        for c, dataset in enumerate(DATASET_ORDER):
            ax = axes[r][c]
            if dataset not in per_model_datasets.get(model, []):
                ax.set_axis_off()
                continue

            root = RUNS_ROOT / model / dataset
            if not root.exists():
                ax.set_axis_off()
                continue

            vanilla_dir = root / "vanilla"
            ours_dir    = root / "ours"
            base_dir    = root / baseline

            # Baseline curve: x=params, y=metric mean across trials
            params, ys = _collect_baseline_param_curve(base_dir, reader)

            if not params or not ys:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                ax.set_axis_off()
                continue

            # Numeric vs categorical x
            numeric_x, all_numeric = [], True
            for p in params:
                try:
                    val = float(p)
                except Exception:
                    all_numeric = False
                    break
                else:
                    numeric_x.append(-val) ## The negative here is to account for the fact that in the paper, beta is > 0.

            if all_numeric:
                ax.plot(numeric_x, ys, marker="o", linewidth=2.4, color=color_baseline, label=f"{baseline_display}")
                ax.set_xlabel(param_display)
                if xlog and all(v > 0 for v in numeric_x):
                    ax.set_xscale("log")
                if len(numeric_x) >= 2:
                    x_min, x_max = min(numeric_x), max(numeric_x)
                    pad = (x_max - x_min) * 0.04 if x_max > x_min else (0.5 if x_max == x_min else 0.0)
                    ax.set_xlim(x_min - pad, x_max + pad)
                x_line = [ax.get_xlim()[0], ax.get_xlim()[1]]
            else:
                xi = list(range(len(params)))
                ax.plot(xi, ys, marker="o", linewidth=2.4, color=color_baseline, label=f"{baseline_display}")
                ax.set_xlabel(param_display)
                ax.set_xticks(xi)
                ax.set_xticklabels(params, rotation=0, ha="center")
                x_line = [xi[0] - 0.2, xi[-1] + 0.2]
                ax.set_xlim(x_line)

            # Vanilla & Ours dashed reference lines (same metric)
            v_mean = _mean_metric_for_variant(vanilla_dir, reader) if vanilla_dir.exists() else None
            o_mean = _mean_metric_for_variant(ours_dir,    reader) if ours_dir.exists()    else None

            if v_mean is not None:
                ax.plot(x_line, [v_mean, v_mean], linestyle="--", color="black", linewidth=1.9, label="Vanilla")
            if o_mean is not None:
                ax.plot(x_line, [o_mean, o_mean], linestyle=(0, (6, 3)), color="#d95f02", linewidth=2.0, label="PAIR")

            ax.set_ylabel(y_label)
            title_ds = DATASET_PRETTY.get(dataset, dataset)
            ax.set_title(f"{model.upper()} — {title_ds}")

            if handles_for_legend is None:
                h, l = ax.get_legend_handles_labels()
                if h:
                    handles_for_legend, labels_for_legend = h, l

    leg = None
    if handles_for_legend:
        try:
            leg = fig.legend(handles_for_legend, labels_for_legend, loc="upper center",
                             ncols=len(labels_for_legend), frameon=False, bbox_to_anchor=(0.5, 1.02))
        except TypeError:
            leg = fig.legend(handles_for_legend, labels_for_legend, loc="upper center",
                             ncol=len(labels_for_legend), frameon=False, bbox_to_anchor=(0.5, 1.02))

    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return fig, leg

# --- NEW: 2×3 LightGCN-only figure (top row Recall@20, bottom row Pop Bias) ---
def make_lgn_two_row_figure(
    baseline: str,
    baseline_display: str,
    param_display: str,
    datasets_for_lgn: List[str],
    xlog: bool = False,
) -> Tuple[Optional[plt.Figure], Optional[plt.Legend]]:
    """
    Builds a 2x3 figure for LGN only:
      Row 0: Recall@20 vs parameter
      Row 1: Popularity Bias vs parameter
      Cols: Gowalla, Yelp2018, Amazon-Book
    """
    model = "lgn"
    if not (RUNS_ROOT / model).exists():
        print("[WARN] LGN directory not found; skipping LGN 2-row figure.")
        return None, None

    # Fonts & layout
    plt.rcParams.update({
        "font.family": "Nimbus Roman",
        "font.serif": ["Nimbus Roman", "Times New Roman", "Times", "DejaVu Serif", "Serif"],
        "axes.grid": True, "grid.linestyle": "--", "grid.alpha": 0.35,
        "font.size": 20, "axes.titlesize": 24, "axes.labelsize": 22,
        "legend.fontsize": 24,
        "xtick.labelsize": 15, "ytick.labelsize": 15,
    })

    nrows, ncols = 2, len(DATASET_ORDER)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5.2*ncols, 8.6), squeeze=False)
    plt.subplots_adjust(hspace=0.32, wspace=0.3)

    # Legend handles
    handles_for_legend = None
    labels_for_legend = None

    # Helper to plot one panel
    def _plot_panel(ax, dataset: str, reader, metric_name: str, y_label: str, color_baseline: str):
        root = RUNS_ROOT / model / dataset
        if not root.exists():
            ax.set_axis_off()
            return False

        vanilla_dir = root / "vanilla"
        ours_dir    = root / "ours"
        base_dir    = root / baseline

        params, ys = _collect_baseline_param_curve(base_dir, reader)
        if not params or not ys:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            return False

        numeric_x, all_numeric = [], True
        for p in params:
            try:
                val = float(p)
            except Exception:
                all_numeric = False
                break
            else:
                numeric_x.append(-val) ## The negative here is to account for the fact that in the paper, beta is > 0.

        if all_numeric:
            ax.plot(numeric_x, ys, marker="o", linewidth=2.4, color=color_baseline, label=f"{baseline_display} (mean)")
            if xlog and all(v > 0 for v in numeric_x):
                ax.set_xscale("log")
            if len(numeric_x) >= 2:
                x_min, x_max = min(numeric_x), max(numeric_x)
                pad = (x_max - x_min) * 0.04 if x_max > x_min else (0.5 if x_max == x_min else 0.0)
                ax.set_xlim(x_min - pad, x_max + pad)
            x_line = [ax.get_xlim()[0], ax.get_xlim()[1]]
            ax.set_xlabel(param_display)
        else:
            xi = list(range(len(params)))
            ax.plot(xi, ys, marker="o", linewidth=3.0, color=color_baseline, label=f"{baseline_display}")
            ax.set_xticks(xi)
            ax.set_xticklabels(params, rotation=0, ha="center")
            x_line = [xi[0] - 0.2, xi[-1] + 0.2]
            ax.set_xlim(x_line)
            ax.set_xlabel(param_display)

        v_mean = _mean_metric_for_variant(vanilla_dir, reader) if vanilla_dir.exists() else None
        o_mean = _mean_metric_for_variant(ours_dir,    reader) if ours_dir.exists()    else None

        if v_mean is not None:
            ax.plot(x_line, [v_mean, v_mean], linestyle="--", color="black", linewidth=1.9, label="Vanilla")
        if o_mean is not None:
            ax.plot(x_line, [o_mean, o_mean], linestyle=(0, (6, 3)), color="#d95f02", linewidth=2.0, label="PAIR")

        ax.set_ylabel(y_label)
        pretty_ds = DATASET_PRETTY.get(dataset, dataset)
        ax.set_title(f"{pretty_ds} -- {metric_name}")

        return True

    # Plot each dataset column for both rows
    any_plotted = False
    for c, dataset in enumerate(DATASET_ORDER):
        if dataset not in datasets_for_lgn:
            axes[0][c].set_axis_off()
            axes[1][c].set_axis_off()
            continue

        top_ax = axes[0][c]
        bot_ax = axes[1][c]

        ok_top = _plot_panel(top_ax, dataset, _read_recall_value, "Recall@20", "Recall@20", "#1f77b4")
        ok_bot = _plot_panel(bot_ax, dataset, _read_pob_value,    "Popularity Bias", "Popularity Bias", "#1f77b4")
        any_plotted = any_plotted or ok_top or ok_bot

        # capture legend handles from first valid panel
        if any_plotted and (handles_for_legend is None):
            h, l = top_ax.get_legend_handles_labels()
            if not h:
                h, l = bot_ax.get_legend_handles_labels()
            if h:
                handles_for_legend, labels_for_legend = h, l

    if not any_plotted:
        print("[WARN] No LGN data plotted; aborting figure.")
        return None, None

    leg = None
    if handles_for_legend:
        try:
            leg = fig.legend(handles_for_legend, labels_for_legend, loc="upper center",
                             ncols=len(labels_for_legend), frameon=False, bbox_to_anchor=(0.5, 1.02))
        except TypeError:
            leg = fig.legend(handles_for_legend, labels_for_legend, loc="upper center",
                             ncol=len(labels_for_legend), frameon=False, bbox_to_anchor=(0.5, 1.02))

    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return fig, leg

# -----------------------------
# Discovery helpers
# -----------------------------
def autodiscover(models_opt: Optional[List[str]], datasets_opt: Optional[List[str]]) -> Tuple[List[str], Dict[str, List[str]]]:
    # Models
    models = models_opt or sorted([p.name for p in RUNS_ROOT.iterdir() if p.is_dir()])
    if not models:
        print(f"[FATAL] No model directories found under {RUNS_ROOT}")
        sys.exit(1)

    # Per-model datasets (respect user filter, then fix order to DATASET_ORDER)
    per_model_datasets: Dict[str, List[str]] = {}
    for m in models:
        model_dir = RUNS_ROOT / m
        if datasets_opt is None:
            ds_available = {p.name for p in model_dir.iterdir() if p.is_dir()} if model_dir.exists() else set()
            ordered = [d for d in DATASET_ORDER if d in ds_available]
        else:
            requested = set(datasets_opt)
            ordered = [d for d in DATASET_ORDER if d in requested and (model_dir / d).exists()]
        per_model_datasets[m] = ordered
    return models, per_model_datasets

# -----------------------------
# CLI
# -----------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Baseline parameter vs metric (Recall and Popularity Bias) with Vanilla/Ours dashed references.")
    ap.add_argument("--baseline", type=str, default=DEFAULT_BASELINE, help="Baseline variant folder name (e.g., ipw, pop_reg).")
    ap.add_argument("--baseline-display", type=str, default=DEFAULT_BASELINE_DISPLAY, help="Display name for the baseline in the legend.")
    ap.add_argument("--param-display", type=str, default=DEFAULT_PARAM_DISPLAY, help="Display name for the baseline parameter (x-axis label).")
    ap.add_argument("--models", type=str, nargs="*", default=None, help="Models to include (default: autodiscover).")
    ap.add_argument("--datasets", type=str, nargs="*", default=None, help="Datasets to include (default: autodiscover per model).")
    ap.add_argument("--xlog", action="store_true", help="Use log-scale for x-axis (only if parameters are numeric and > 0).")
    return ap.parse_args()

# -----------------------------
# Main
# -----------------------------
def main():
    args = parse_args()
    baseline          = args.baseline
    baseline_display  = args.baseline_display
    param_display     = args.param_display
    models_opt        = args.models
    datasets_opt      = args.datasets
    xlog              = bool(args.xlog)

    models, per_model_datasets = autodiscover(models_opt, datasets_opt)

    # Figure 1: Recall@20
    fig1, leg1 = make_figure_for_metric(
        baseline=baseline,
        baseline_display=baseline_display,
        param_display=param_display,
        models=models,
        per_model_datasets=per_model_datasets,
        metric_name="Recall@20",
        reader=_read_recall_value,
        y_label="Recall@20",
        color_baseline="#1f77b4",
        xlog=xlog,
    )
    out_path1 = OUT_DIR / f"baseline_param_vs_recall_{baseline}{'_logx' if xlog else ''}.pdf"
    if fig1 is not None:
        if leg1 is not None:
            fig1.savefig(out_path1, bbox_inches="tight", bbox_extra_artists=(leg1,), pad_inches=0.25)
        else:
            fig1.savefig(out_path1, bbox_inches="tight")
        print(f"[OK] Saved figure: {out_path1}")

    # Figure 2: Popularity Bias
    fig2, leg2 = make_figure_for_metric(
        baseline=baseline,
        baseline_display=baseline_display,
        param_display=param_display,
        models=models,
        per_model_datasets=per_model_datasets,
        metric_name="Popularity Bias",
        reader=_read_pob_value,
        y_label="Popularity Bias",
        color_baseline="#1f77b4",
        xlog=xlog,
    )
    out_path2 = OUT_DIR / f"baseline_param_vs_popbias_{baseline}{'_logx' if xlog else ''}.pdf"
    if fig2 is not None:
        if leg2 is not None:
            fig2.savefig(out_path2, bbox_inches="tight", bbox_extra_artists=(leg2,), pad_inches=0.25)
        else:
            fig2.savefig(out_path2, bbox_inches="tight")
        print(f"[OK] Saved figure: {out_path2}")

    # Figure 3: LGN-only, two rows (Recall@20 on top, Pop Bias bottom)
    lgn_datasets = per_model_datasets.get("lgn", [])
    fig3, leg3 = make_lgn_two_row_figure(
        baseline=baseline,
        baseline_display=baseline_display,
        param_display=param_display,
        datasets_for_lgn=lgn_datasets,
        xlog=xlog,
    )
    if fig3 is not None:
        out_path3 = OUT_DIR / f"baseline_param_LGN_2row_{baseline}{'_logx' if xlog else ''}.pdf"
        if leg3 is not None:
            fig3.savefig(out_path3, bbox_inches="tight", bbox_extra_artists=(leg3,), pad_inches=0.25)
        else:
            fig3.savefig(out_path3, bbox_inches="tight")
        print(f"[OK] Saved figure: {out_path3}")

if __name__ == "__main__":
    main()
