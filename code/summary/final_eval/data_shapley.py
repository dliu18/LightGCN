#!/usr/bin/env python3
"""
Generate a grouped bar chart of %Δ <METRIC_NAME>@20 vs 'fixed' for each dataset.

Groups are DATASETS on x-axis; bars inside each group are TREATMENTS
(each treatment has its own fixed color). Error bars are 95% CIs.

Set METRIC_NAME to one of: "Recall", "Precision", "NDCG".
"""

from __future__ import annotations
import os, sys, math
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt

# =============================
# CONFIG: choose your metric
# =============================
METRIC_NAME = "Recall"   # "Recall" | "Precision" | "NDCG"
SAMPLE_TYPE = "constant_users" # constant_users | constant_interactions

# -----------------------------
# Validation & derived constants
# -----------------------------
_ALLOWED = {"Recall", "Precision", "NDCG"}
if METRIC_NAME not in _ALLOWED:
    raise ValueError(f"METRIC_NAME must be one of {_ALLOWED}, got {METRIC_NAME}")

TAG_SUFFIX = f"Test/{METRIC_NAME}@[20, 2000]"
_PATH_KEYS = [f"{METRIC_NAME}@20", f"{METRIC_NAME}@[20, 2000]", f"{METRIC_NAME}@[20,2000]"]
_EXCLUDE_UNDERSCORE_MARKER = f"_{METRIC_NAME}@[20, 2000]"

METRIC_SHORT = f"{METRIC_NAME}@20"
YLABEL = f"% Δ {METRIC_SHORT} vs. Fixed"
TITLE = f"Effect of User Selection Strategies on {METRIC_SHORT}\n(mean % change ± 95% CI)"

# -----------------------------
# TensorBoard readers
# -----------------------------
_EVENT_ACCUMULATOR = None
_TF_SUMMARY_ITER = None
try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    _EVENT_ACCUMULATOR = EventAccumulator
except Exception:
    _EVENT_ACCUMULATOR = None

if _EVENT_ACCUMULATOR is None:
    try:
        from tensorflow.python.summary.summary_iterator import summary_iterator as tf_summary_iterator  # type: ignore
        _TF_SUMMARY_ITER = tf_summary_iterator
    except Exception:
        _TF_SUMMARY_ITER = None

# -----------------------------
# Paths
# -----------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
CODE_DIR = SCRIPT_DIR.parent.parent
PROJECT_ROOT = CODE_DIR.parent
RUNS_DIR = CODE_DIR / "runs" / "data_shapley"
OUT_DIR = PROJECT_ROOT / "outputs" / "final_eval"
OUT_PATH = OUT_DIR / f"data_shapley_treatment_ratio_{SAMPLE_TYPE}_{METRIC_NAME.lower()}.pdf"

# Treatments, labels, and fixed colors
TREATMENT_ORDER = [
    "random",
    "low_niche",
    "low_mainstream",
    "power_niche",
    "power_mainstream",
]
TREATMENT_LABELS = {
    "random": "Random",
    "low_niche": "Low-Niche",
    "low_mainstream": "Low-Mainstream",
    "power_niche": "Power-Niche",
    "power_mainstream": "Power-Mainstream",
}
TREATMENT_COLORS = {
    "random": "#1b9e77",
    "low_niche": "#d95f02",
    "low_mainstream": "#7570b3",
    "power_niche": "#e7298a",
    "power_mainstream": "#66a61e",
}

# -----------------------------
# Discovery & reading helpers
# -----------------------------
def list_datasets(runs_dir: Path) -> List[str]:
    return ["gowalla-{}".format(ratio) for ratio in [0.01, 0.05, 0.1, 0.3, 0.4, 0.5]]
    return sorted([p.name for p in runs_dir.iterdir() if p.is_dir()]) if runs_dir.exists() else []

def find_event_files(root: Path) -> List[Path]:
    event_files = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.startswith("events.out.tfevents."):
                event_files.append(Path(dirpath) / fn)
    return event_files

def _read_scalar_last_with_event_accumulator(event_file: Path, want_suffix: str) -> Optional[Tuple[int, float, float]]:
    try:
        ea = _EVENT_ACCUMULATOR(str(event_file.parent))
        ea.Reload()
        tags = ea.Tags().get("scalars", []) or []
        candidates = [t for t in tags if t == want_suffix] or [t for t in tags if t.endswith(want_suffix)]
        if not candidates: return None
        scalars = ea.Scalars(sorted(candidates)[0])
        if not scalars: return None
        last = scalars[-1]
        return int(last.step), float(last.value), float(last.wall_time)
    except Exception:
        return None

def _read_scalar_last_with_tf_summary_iterator(event_file: Path, want_suffix: str) -> Optional[Tuple[int, float, float]]:
    if _TF_SUMMARY_ITER is None: return None
    last_found = None
    try:
        for event in _TF_SUMMARY_ITER(str(event_file)):
            if not getattr(event, "summary", None): continue
            for v in event.summary.value:
                tag = getattr(v, "tag", "")
                if tag and (tag == want_suffix or tag.endswith(want_suffix)):
                    val = getattr(v, "simple_value", None)
                    if val is None:
                        try: val = float(v.tensor.float_val[0])  # type: ignore
                        except Exception: continue
                    last_found = (int(event.step), float(val), float(getattr(event, "wall_time", 0.0)))
    except Exception:
        return None
    return last_found

def _get_last_scalar_any_tag_event_acc(dir_for_ea: Path) -> Optional[Tuple[int, float, float]]:
    if _EVENT_ACCUMULATOR is None: return None
    try:
        ea = _EVENT_ACCUMULATOR(str(dir_for_ea))
        ea.Reload()
        best = None
        for t in ea.Tags().get("scalars", []):
            vals = ea.Scalars(t)
            if not vals: continue
            last = vals[-1]
            cand = (float(last.wall_time), int(last.step), float(last.value))
            if best is None or cand > best: best = cand
        if best is None: return None
        return int(best[1]), float(best[2]), float(best[0])
    except Exception:
        return None

def read_metric20_last(run_root: Path) -> Optional[float]:
    event_files = [ef for ef in find_event_files(run_root) if _EXCLUDE_UNDERSCORE_MARKER not in ef.as_posix()]
    candidates = []
    for ef in event_files:
        chosen = None
        if _EVENT_ACCUMULATOR is not None:
            found = _read_scalar_last_with_event_accumulator(ef, TAG_SUFFIX)
            if found: step, value, wall_time = found; chosen = (wall_time, step, value)
        if chosen is None and _TF_SUMMARY_ITER is not None:
            found = _read_scalar_last_with_tf_summary_iterator(ef, TAG_SUFFIX)
            if found: step, value, wall_time = found; chosen = (wall_time, step, value)
        if chosen is None and any(key in ef.as_posix() for key in _PATH_KEYS):
            got_any = _get_last_scalar_any_tag_event_acc(ef.parent)
            if got_any: step, value, wall_time = got_any; chosen = (wall_time, step, value)
        if chosen: candidates.append((*chosen, ef))
    if not candidates: return None
    candidates.sort()
    _, step, value, chosen_file = candidates[-1]
    print(f"[INFO] Selected file for {run_root}: {chosen_file}, {METRIC_SHORT}={value:.6f} (step={step})")
    return float(value)

# -----------------------------
# Stats helpers
# -----------------------------
def mean_ci_95(xs: List[float]) -> Tuple[float, float]:
    n = len(xs)
    if n == 0: return float("nan"), float("nan")
    if n == 1: return xs[0], 0.0
    mean = sum(xs) / n
    var = sum((x - mean) ** 2 for x in xs) / (n - 1)
    se = math.sqrt(var) / math.sqrt(n)
    try:
        from scipy.stats import t as student_t
        t_crit = float(student_t.ppf(0.975, df=n - 1))
    except Exception:
        t_crit = 1.96
    return mean, t_crit * se

def collect_dataset_results(dataset_dir: Path) -> Tuple[Optional[float], Dict[str, List[float]]]:
    baseline = read_metric20_last(dataset_dir / "fixed" / SAMPLE_TYPE)
    pct_changes = {cfg: [] for cfg in TREATMENT_ORDER}
    if baseline in (None, 0): return baseline, pct_changes
    for cfg in TREATMENT_ORDER:
        cfg_root = dataset_dir / cfg
        trials_root = cfg_root / SAMPLE_TYPE
        if not trials_root.exists(): trials_root = cfg_root
        trials = [p for p in trials_root.iterdir() if p.is_dir() and p.name.isdigit()] or ([cfg_root] if cfg_root.exists() else [])
        for tdir in sorted(trials, key=lambda p: int(p.name) if p.name.isdigit() else p.name):
            val = read_metric20_last(tdir)
            if val is None: continue
            pct_changes[cfg].append(100.0 * (val - baseline) / baseline)
    return baseline, pct_changes

# -----------------------------
# Main
# -----------------------------
def main() -> None:
    datasets = list_datasets(RUNS_DIR)
    baselines, pct_all, raw_counts = {}, {cfg: {} for cfg in TREATMENT_ORDER}, {cfg: {} for cfg in TREATMENT_ORDER}
    for ds in datasets:
        baseline, pct_by_treatment = collect_dataset_results(RUNS_DIR / ds)
        baselines[ds] = baseline
        for cfg in TREATMENT_ORDER:
            xs = pct_by_treatment[cfg]
            if xs: m, c = mean_ci_95(xs); pct_all[cfg][ds] = (m, c); raw_counts[cfg][ds] = len(xs)
            else: pct_all[cfg][ds] = (float("nan"), float("nan")); raw_counts[cfg][ds] = 0
    used_datasets = [ds for ds in datasets if baselines.get(ds) not in (None, 0.0)]
    if not used_datasets: sys.exit("Nothing to plot.")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({"figure.figsize": (max(8.0, 1.8 * len(used_datasets)), 5.2),
                         "axes.grid": True,"grid.linestyle": "--","grid.alpha": 0.35,
                         "font.size": 12,"axes.titlesize": 14,"axes.labelsize": 13,
                         "legend.fontsize": 11})
    fig, ax = plt.subplots()
    num_groups, num_series = len(used_datasets), len(TREATMENT_ORDER)
    x_centers = list(range(num_groups))
    total_bar_width, bar_width = 0.8, 0.8 / max(num_series, 1)
    offsets = [(-total_bar_width / 2) + (i + 0.5) * bar_width for i in range(num_series)]

    # Draw per treatment (legend = treatments)
    for si, cfg in enumerate(TREATMENT_ORDER):
        means, cis = [], []
        for ds in used_datasets: m, c = pct_all[cfg][ds]; means.append(m); cis.append(c)
        xs = [xc + offsets[si] for xc in x_centers]
        ax.bar(xs, means, width=bar_width, label=TREATMENT_LABELS[cfg],
               color=TREATMENT_COLORS[cfg], edgecolor="black", linewidth=0.5, zorder=3)
        ax.errorbar(xs, means, yerr=cis, fmt="none", ecolor="black", elinewidth=1, capsize=3, zorder=4)

    ax.set_xticks(x_centers); ax.set_xticklabels(used_datasets, rotation=0, ha="center")
    ax.set_ylabel(YLABEL); ax.set_title(TITLE); ax.axhline(0.0, color="black", linewidth=1.0)
    ax.legend(title="Treatment", bbox_to_anchor=(1.02, 1.0), loc="upper left")
    fig.tight_layout(); fig.savefig(OUT_PATH, bbox_inches="tight")
    print(f"[OK] Saved figure: {OUT_PATH}")

if __name__ == "__main__":
    main()
