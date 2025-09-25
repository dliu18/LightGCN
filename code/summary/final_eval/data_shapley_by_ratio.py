#!/usr/bin/env python3
"""
Generate a line chart of %Δ <METRIC_NAME> vs 'fixed' across treatment ratios
for a SINGLE dataset.

- X-axis: treatment ratio (parsed from runs directories named <dataset>-<ratio>)
- Y-axis: percent change over the corresponding fixed model at the same ratio
- One line per treatment, with 95% CI error bars at each ratio point

Set METRIC_NAME to one of: "Recall", "Precision", "NDCG", "Popularity Bias".
Set DATASET_NAME to one of: "gowalla", "yelp2018", "amazon-book".

Special handling:
- Popularity Bias is read ONLY from the exact tag "Test/Popularity_Opportunity_Bias_20".
- For metrics other than Popularity Bias, any event file located directly in a trial/root
  directory (the "top-level events file") is IGNORED; we only consider files in subfolders.
"""

from __future__ import annotations
import os, sys, math, re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from tqdm import tqdm

# =============================
# CONFIG
# =============================
METRIC_NAME   = "Popularity Bias"            # "Recall" | "Precision" | "NDCG" | "Popularity Bias"
SAMPLE_TYPE   = "constant_users"    # "constant_users" | "constant_interactions"
DATASET_NAME  = "gowalla"           # "gowalla" | "yelp2018" | "amazon-book"

# -----------------------------
# Validation & metric config
# -----------------------------
_ALLOWED_METRICS   = {"Recall", "Precision", "NDCG", "Popularity Bias"}
_ALLOWED_DATASETS  = {"gowalla", "yelp2018", "amazon-book"}
if METRIC_NAME not in _ALLOWED_METRICS:
    raise ValueError(f"METRIC_NAME must be one of {_ALLOWED_METRICS}, got {METRIC_NAME}")
if DATASET_NAME not in _ALLOWED_DATASETS:
    raise ValueError(f"DATASET_NAME must be one of {_ALLOWED_DATASETS}, got {DATASET_NAME}")

def _metric_cfg(metric_name: str) -> Dict[str, object]:
    """
    Metric-specific details:
      - tag_suffix: TensorBoard tag to read (exact or ending match)
      - path_keys:  substrings that enable fallback 'any-tag latest' read (disabled for PopBias)
      - exclude_marker: substring to exclude certain event files
      - display_short: short label for axis/title
      - strict_tag_only: True => do NOT use path fallbacks (Popularity Bias needs exact tag)
    """
    if metric_name in {"Recall", "Precision", "NDCG"}:
        tag_suffix = f"Test/{metric_name}@[20, 2000]"
        path_keys = [f"{metric_name}@20", f"{metric_name}@[20, 2000]", f"{metric_name}@[20,2000]"]
        exclude_marker = f"_{metric_name}@[20, 2000]"
        display_short = f"{metric_name}@20"
        strict_tag_only = False
    elif metric_name == "Popularity Bias":
        tag_suffix = "Test/Popularity_Opportunity_Bias_20"
        path_keys = []      # disable path-based fallback for safety
        exclude_marker = "" # no special underscore copies known
        display_short = "Popularity Bias@20"
        strict_tag_only = True
    else:
        raise ValueError(metric_name)

    return {
        "tag_suffix": tag_suffix,
        "path_keys": path_keys,
        "exclude_marker": exclude_marker,
        "display_short": display_short,
        "strict_tag_only": strict_tag_only,
    }

_cfg = _metric_cfg(METRIC_NAME)
TAG_SUFFIX = _cfg["tag_suffix"]                      # type: ignore
_PATH_KEYS = _cfg["path_keys"]                       # type: ignore
_EXCLUDE_UNDERSCORE_MARKER = _cfg["exclude_marker"]  # type: ignore
STRICT_TAG_ONLY = _cfg["strict_tag_only"]            # type: ignore
METRIC_SHORT = _cfg["display_short"]                 # type: ignore

YLABEL = f"% Δ {METRIC_SHORT} vs. Fixed"
TITLE  = f"{DATASET_NAME}: Effect of User Selection Strategies on {METRIC_SHORT}\n(mean % change ± 95% CI)"

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
SCRIPT_DIR   = Path(__file__).resolve().parent
CODE_DIR     = SCRIPT_DIR.parent.parent
PROJECT_ROOT = CODE_DIR.parent
RUNS_DIR     = CODE_DIR / "runs" / "data_shapley" / "tight-quadrants"

OUT_DIR  = PROJECT_ROOT / "outputs" / "final_eval"
OUT_PATH = OUT_DIR / f"data_shapley_ratio_sweep_{DATASET_NAME}_{SAMPLE_TYPE}_{METRIC_NAME.lower().replace(' ', '_')}.pdf"

# -----------------------------
# Treatments, labels, and colors
# -----------------------------
TREATMENT_ORDER = [
    "random",
    "low_mainstream",
    "low_niche",
    "power_niche",
    "power_mainstream",
]
TREATMENT_ALIASES = {
    "low-mainstream": "low_mainstream",
    "low-niche": "low_niche",
    "power-niche": "power_niche",
    "power-mainstream": "power_mainstream",
}
TREATMENT_LABELS = {
    "random": "Random",
    "low_mainstream": "Low-Mainstream",
    "low_niche": "Low-Niche",
    "power_niche": "Power-Niche",
    "power_mainstream": "Power-Mainstream",
}
TREATMENT_COLORS = {
    "random": "black",
    "low_mainstream": "#d95f02",
    "low_niche": "#7570b3",
    "power_niche": "#e7298a",
    "power_mainstream": "#66a61e",
}

# -----------------------------
# Discovery helpers
# -----------------------------
def _ratio_dir_regex(dataset_name: str) -> re.Pattern:
    return re.compile(rf"^{re.escape(dataset_name)}-(?P<ratio>[\d\.]+)$")

def list_ratio_dirs(runs_dir: Path, dataset_name: str) -> List[Tuple[float, Path]]:
    """Find '<dataset>-<ratio>' directories; return sorted list of (ratio, path)."""
    _RE = _ratio_dir_regex(dataset_name)
    results: List[Tuple[float, Path]] = []
    if not runs_dir.exists():
        return results
    for p in runs_dir.iterdir():
        if not p.is_dir(): continue
        m = _RE.match(p.name)
        if m:
            try:
                ratio = float(m.group("ratio"))
                results.append((ratio, p))
            except ValueError:
                continue
    results.sort(key=lambda x: x[0])
    return results

def _list_event_files(run_root: Path, strict_tag_only: bool) -> List[Path]:
    """
    Return event files under run_root.
    - If strict_tag_only (Popularity Bias): include all (top-level allowed).
    - Else (other metrics): EXCLUDE event files whose parent is exactly run_root
      (i.e., 'top-level events file'); only include ones in subdirectories.
    """
    files: List[Path] = []
    if not run_root.exists():
        return files
    run_root_resolved = run_root.resolve()
    for dirpath, _, filenames in os.walk(run_root_resolved):
        dpath = Path(dirpath)
        for fn in filenames:
            if not fn.startswith("events.out.tfevents."):
                continue
            # Skip top-level only for non-PopBias
            if not strict_tag_only and dpath == run_root_resolved:
                continue
            files.append(dpath / fn)
    return files

# -----------------------------
# Reading helpers
# -----------------------------
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
                        try:
                            val = float(v.tensor.float_val[0])  # type: ignore
                        except Exception:
                            continue
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

def read_metric_last(run_root: Path) -> Optional[float]:
    """
    Read the metric from event files under `run_root`.
    - Popularity Bias: require exact/ending tag match; include top-level event files.
    - Others: ignore top-level event files; allow conservative path-based fallback.
    """
    event_files = _list_event_files(run_root, STRICT_TAG_ONLY)
    if _EXCLUDE_UNDERSCORE_MARKER:
        event_files = [ef for ef in event_files if _EXCLUDE_UNDERSCORE_MARKER not in ef.as_posix()]

    candidates = []
    for ef in event_files:
        chosen = None
        # Preferred: tag-based reads
        if _EVENT_ACCUMULATOR is not None:
            found = _read_scalar_last_with_event_accumulator(ef, TAG_SUFFIX)
            if found:
                step, value, wall_time = found
                chosen = (wall_time, step, value)
        if chosen is None and _TF_SUMMARY_ITER is not None:
            found = _read_scalar_last_with_tf_summary_iterator(ef, TAG_SUFFIX)
            if found:
                step, value, wall_time = found
                chosen = (wall_time, step, value)

        # Fallback (ONLY for non-PopBias): latest scalar of any tag in that dir,
        # but only if path hints match.
        if not STRICT_TAG_ONLY and chosen is None and _PATH_KEYS and any(key in ef.as_posix() for key in _PATH_KEYS):
            got_any = _get_last_scalar_any_tag_event_acc(ef.parent)
            if got_any:
                step, value, wall_time = got_any
                chosen = (wall_time, step, value)

        if chosen:
            candidates.append((*chosen, ef))

    if not candidates:
        if STRICT_TAG_ONLY:
            print(f"[WARN] No tag '{TAG_SUFFIX}' found under: {run_root}")
        return None

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

# -----------------------------
# Collection per ratio
# -----------------------------
def normalize_treatment_name(name: str) -> str:
    return TREATMENT_ALIASES.get(name, name)

def collect_ratio_results(dataset_ratio_dir: Path) -> Tuple[Optional[float], Dict[str, List[float]]]:
    """
    For one '<dataset>-<ratio>' directory, return:
      - baseline fixed metric (float or None)
      - pct_changes dict: treatment -> list of trial % changes vs fixed
    """
    baseline = read_metric_last(dataset_ratio_dir / "fixed" / SAMPLE_TYPE)
    pct_changes: Dict[str, List[float]] = {cfg: [] for cfg in TREATMENT_ORDER}
    if baseline in (None, 0):
        return baseline, pct_changes

    for raw_cfg in tqdm(TREATMENT_ORDER):
        cfg = normalize_treatment_name(raw_cfg)
        cfg_root = dataset_ratio_dir / cfg
        trials_root = cfg_root / SAMPLE_TYPE
        if not trials_root.exists():
            trials_root = cfg_root
        if not cfg_root.exists() and not trials_root.exists():
            continue

        try:
            entries = list(trials_root.iterdir()) if trials_root.exists() else []
        except Exception:
            entries = []

        trials = [p for p in entries if p.is_dir() and p.name.isdigit()]
        if not trials:
            trials = [cfg_root] if cfg_root.exists() else []

        for tdir in sorted(trials, key=lambda p: int(p.name) if p.name.isdigit() else p.name):
            val = read_metric_last(tdir)
            if val is None:
                continue
            pct = 100.0 * (val - float(baseline)) / float(baseline)
            pct_changes[raw_cfg].append(pct)

    return baseline, pct_changes

# -----------------------------
# Main
# -----------------------------
def main() -> None:
    ratio_dirs = list_ratio_dirs(RUNS_DIR, DATASET_NAME)
    if not ratio_dirs:
        sys.exit(f"No directories like '{DATASET_NAME}-<ratio>' found under {RUNS_DIR}")

    means_by_treatment: Dict[str, List[float]] = {cfg: [] for cfg in TREATMENT_ORDER}
    cis_by_treatment: Dict[str, List[float]] = {cfg: [] for cfg in TREATMENT_ORDER}
    counts_by_treatment: Dict[str, List[int]] = {cfg: [] for cfg in TREATMENT_ORDER}
    baselines_by_ratio: Dict[float, Optional[float]] = {}

    plotted_ratios: List[float] = []
    for ratio, ratio_path in ratio_dirs:
        baseline, pct_changes = collect_ratio_results(ratio_path)
        baselines_by_ratio[ratio] = baseline
        if baseline in (None, 0.0):
            for cfg in TREATMENT_ORDER:
                means_by_treatment[cfg].append(float("nan"))
                cis_by_treatment[cfg].append(float("nan"))
                counts_by_treatment[cfg].append(0)
            continue

        plotted_ratios.append(ratio)
        for cfg in TREATMENT_ORDER:
            xs = pct_changes[cfg]
            if xs:
                m, c = mean_ci_95(xs)
                means_by_treatment[cfg].append(m)
                cis_by_treatment[cfg].append(c)
                counts_by_treatment[cfg].append(len(xs))
            else:
                means_by_treatment[cfg].append(float("nan"))
                cis_by_treatment[cfg].append(float("nan"))
                counts_by_treatment[cfg].append(0)

    if not plotted_ratios:
        sys.exit("Nothing to plot (no valid baselines).")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Plot
    plt.rcParams.update({
        "figure.figsize": (max(7.5, 1.2 * len(plotted_ratios)), 5.2),
        "axes.grid": True, "grid.linestyle": "--", "grid.alpha": 0.35,
        "font.size": 12, "axes.titlesize": 14, "axes.labelsize": 13,
        "legend.fontsize": 11
    })

    fig, ax = plt.subplots()
    x = list(range(len(plotted_ratios)))

    for cfg in TREATMENT_ORDER:
        y = means_by_treatment[cfg]
        yerr = cis_by_treatment[cfg]
        valid_pts = [(i, yi, ei) for i, (yi, ei) in enumerate(zip(y, yerr)) if (yi == yi and ei == ei)]
        if not valid_pts:
            continue
        xi = [plotted_ratios[i] for (i, _, _) in valid_pts]
        yi = [yi for (_, yi, _) in valid_pts]
        ei = [ei for (_, _, ei) in valid_pts]

        ax.plot(xi, yi, marker="o", linewidth=2, label=TREATMENT_LABELS[cfg],
                color=TREATMENT_COLORS[cfg], zorder=3)
        # ax.errorbar(xi, yi, yerr=ei, fmt="none", ecolor="black", elinewidth=1,
        #             capsize=3, zorder=4)

    # ax.set_xticks(x)
    # ax.set_xticklabels([str(r) for r in plotted_ratios], rotation=0, ha="center")
    ax.set_xlabel("Treatment ratio")
    ax.set_ylabel(YLABEL)
    ax.set_title(TITLE)
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.legend(title="Treatment", bbox_to_anchor=(1.02, 1.0), loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT_PATH, bbox_inches="tight")
    print(f"[OK] Saved figure: {OUT_PATH}")

    # Optional: counts table
    print("\n[INFO] Trial counts per ratio (treatment: count):")
    header = "ratio".ljust(8) + "  " + "  ".join(t.ljust(16) for t in TREATMENT_ORDER)
    print(header)
    for idx, r in enumerate(plotted_ratios):
        row = f"{str(r).ljust(8)}  " + "  ".join(str(counts_by_treatment[t][idx]).ljust(16) for t in TREATMENT_ORDER)
        print(row)

if __name__ == "__main__":
    main()
