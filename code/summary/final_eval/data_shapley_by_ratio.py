from __future__ import annotations
import os, sys, math, re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from tqdm import tqdm

# =============================
# CONFIG
# =============================
METRIC_NAME   = "Recall"            # "Recall" | "Precision" | "NDCG" | "Popularity Bias" | "Niche Recall"
SAMPLE_TYPE   = "constant_users"    # "constant_users" | "constant_interactions"
DATASETS      = ["gowalla", "yelp2018", "amazon-book"]  # order is subplot order (left->right)

# -----------------------------
# Validation & metric config
# -----------------------------
_ALLOWED_METRICS   = {"Recall", "Precision", "NDCG", "Popularity Bias", "Niche Recall"}
_ALLOWED_DATASETS  = {"gowalla", "yelp2018", "amazon-book"}
if METRIC_NAME not in _ALLOWED_METRICS:
    raise ValueError(f"METRIC_NAME must be one of {_ALLOWED_METRICS}, got {METRIC_NAME}")

def _metric_cfg(metric_name: str) -> Dict[str, object]:
    """
    Metric-specific details:
      - tag_regex: compiled regex to match the desired TensorBoard tag(s)
      - path_hints: substrings that enable a conservative 'any-tag latest' fallback (non-PopBias only)
      - exclude_marker: substring to exclude certain event files (e.g., underscored duplicates)
      - display_short: short label for axis/title
      - strict_tag_only: True => do NOT use path fallbacks (Popularity Bias needs exact tag)
      - bracket_metric: True for Recall/Precision/NDCG (to parse top_k)
    """
    if metric_name in {"Recall", "Precision", "NDCG"}:
        # Match: Test/Metric@[20, <top_k>] with optional spaces
        tag_regex = re.compile(rf"^Test/{metric_name}@\[\s*20\s*,\s*(\d+)\s*\]$")
        path_hints = [f"{metric_name}@20", f"{metric_name}@[20", f"{metric_name}@[20,"]
        exclude_marker = f"_{metric_name}@["  # skip underscored duplicates if present
        display_short = f"{metric_name}@20"
        strict_tag_only = False
        bracket_metric = True
    elif metric_name == "Popularity Bias":
        tag_regex = re.compile(r"^Test/Popularity_Opportunity_Bias_20$")
        path_hints = []
        exclude_marker = ""
        display_short = "Popularity Bias@20"
        strict_tag_only = True
        bracket_metric = False
    elif metric_name == "Niche Recall":
        tag_regex = re.compile(r"^Test/Niche-Recall_20$")
        path_hints = []
        exclude_marker = ""
        display_short = "Niche Recall@20"
        strict_tag_only = True  # treat like Pop Bias (exact tag; ignore path fallback)
        bracket_metric = False
    else:
        raise ValueError(metric_name)

    return {
        "tag_regex": tag_regex,
        "path_hints": path_hints,
        "exclude_marker": exclude_marker,
        "display_short": display_short,
        "strict_tag_only": strict_tag_only,
        "bracket_metric": bracket_metric,
    }

_cfg = _metric_cfg(METRIC_NAME)
_TAG_REGEX = _cfg["tag_regex"]                           # type: ignore
_PATH_HINTS = _cfg["path_hints"]                         # type: ignore
_EXCLUDE_UNDERSCORE_MARKER = _cfg["exclude_marker"]      # type: ignore
STRICT_TAG_ONLY = _cfg["strict_tag_only"]                # type: ignore
BRACKET_METRIC = _cfg["bracket_metric"]                  # type: ignore
METRIC_SHORT = _cfg["display_short"]                     # type: ignore

YLABEL = f"% Δ {METRIC_SHORT} vs. Fixed"

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
RUNS_DIR     = CODE_DIR / "runs" / "data_shapley"

OUT_DIR  = PROJECT_ROOT / "outputs" / "final_eval" / "www"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / f"data_shapley_ratio_sweep_ALLDATASETS_{SAMPLE_TYPE}_{METRIC_NAME.lower().replace(' ', '_')}.pdf"

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
    "low_mainstream": "Light-Mainstream",
    "low_niche": "Light-Niche",
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

DATASET_PRETTY = {
    "gowalla": "Gowalla",
    "yelp2018": "Yelp2018",
    "amazon-book": "Amazon-Book",
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
    - If strict_tag_only (Popularity Bias / Niche Recall): include all (top-level allowed).
    - Else (Recall/Precision/NDCG): EXCLUDE event files whose parent is exactly run_root
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
            if not strict_tag_only and dpath == run_root_resolved:
                continue
            files.append(dpath / fn)
    return files

# -----------------------------
# Reading helpers
# -----------------------------
def _gather_candidates_with_ea(dir_for_ea: Path, tag_regex: re.Pattern) -> List[Tuple[float, int, float, str]]:
    out: List[Tuple[float, int, float, str]] = []
    if _EVENT_ACCUMULATOR is None:
        return out
    try:
        ea = _EVENT_ACCUMULATOR(str(dir_for_ea))
        ea.Reload()
        for t in ea.Tags().get("scalars", []) or []:
            if not tag_regex.match(t):
                continue
            vals = ea.Scalars(t)
            if not vals:
                continue
            last = vals[-1]
            out.append((float(last.wall_time), int(last.step), float(last.value), t))
    except Exception:
        return out
    return out

def _gather_candidates_with_tf(event_file: Path, tag_regex: re.Pattern) -> List[Tuple[float, int, float, str]]:
    out: List[Tuple[float, int, float, str]] = []
    if _TF_SUMMARY_ITER is None:
        return out
    try:
        for event in _TF_SUMMARY_ITER(str(event_file)):
            if not getattr(event, "summary", None):
                continue
            wt = float(getattr(event, "wall_time", 0.0) or 0.0)
            st = int(getattr(event, "step", 0) or 0)
            for v in event.summary.value:
                tag = getattr(v, "tag", "")
                if tag and tag_regex.match(tag):
                    val = getattr(v, "simple_value", None)
                    if val is None:
                        try:
                            val = float(v.tensor.float_val[0])  # type: ignore
                        except Exception:
                            continue
                    out.append((wt, st, float(val), tag))
    except Exception:
        return out
    return out

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
    - Popularity Bias: exact tag match; include top-level event files.
    - Niche Recall: exact tag "Test/Niche-Recall_20"; include top-level events (strict).
    - Recall/Precision/NDCG: match any tag "Test/<Metric>@[20, <top_k>]". If a trial has multiple
      distinct <top_k> values, print a warning. Select the candidate with the latest wall_time.
    """
    event_files = _list_event_files(run_root, STRICT_TAG_ONLY)
    if _EXCLUDE_UNDERSCORE_MARKER:
        event_files = [ef for ef in event_files if _EXCLUDE_UNDERSCORE_MARKER not in ef.as_posix()]

    candidates: List[Tuple[float, int, float, str, Path]] = []
    topks_found: set = set()

    for ef in event_files:
        # Prefer EA on the directory, then TF on the file
        ea_candidates = _gather_candidates_with_ea(ef.parent, _TAG_REGEX)
        if ea_candidates:
            for (wt, st, val, tag) in ea_candidates:
                if BRACKET_METRIC:
                    m = _TAG_REGEX.match(tag)
                    if m:
                        topks_found.add(int(m.group(1)))
                candidates.append((wt, st, val, tag, ef))
            continue

        tf_candidates = _gather_candidates_with_tf(ef, _TAG_REGEX)
        for (wt, st, val, tag) in tf_candidates:
            if BRACKET_METRIC:
                m = _TAG_REGEX.match(tag)
                if m:
                    topks_found.add(int(m.group(1)))
            candidates.append((wt, st, val, tag, ef))

        # Fallback (ONLY for non-PopBias): latest scalar of any tag in that dir,
        # but only if path hints match.
        if not STRICT_TAG_ONLY and not ea_candidates and not tf_candidates and _PATH_HINTS and any(h in ef.as_posix() for h in _PATH_HINTS):
            got_any = _get_last_scalar_any_tag_event_acc(ef.parent)
            if got_any:
                st, val, wt = got_any
                candidates.append((wt, st, float(val), "<any>", ef))

    if not candidates:
        if STRICT_TAG_ONLY:
            print(f"[WARN] No tag matching '{_TAG_REGEX.pattern}' found under: {run_root}")
        return None

    if BRACKET_METRIC and len(topks_found) > 1:
        sorted_ks = sorted(topks_found)
        print(f"[WARN] Multiple top_k values found under {run_root}: {sorted_ks}")

    candidates.sort(key=lambda t: (t[0], t[1]))  # latest by wall-time, then step
    wt, st, value, tag, chosen_file = candidates[-1]
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
# Per-dataset collection
# -----------------------------
def normalize_treatment_name(name: str) -> str:
    return TREATMENT_ALIASES.get(name, name)

def collect_ratio_results_for_dataset(dataset_name: str) -> Tuple[List[float], Dict[str, List[float]], Dict[str, List[float]], Dict[str, List[int]]]:
    """Return plotted_ratios, means_by_treatment, cis_by_treatment, counts_by_treatment."""
    ratio_dirs = list_ratio_dirs(RUNS_DIR, dataset_name)
    means_by_treatment: Dict[str, List[float]] = {cfg: [] for cfg in TREATMENT_ORDER}
    cis_by_treatment: Dict[str, List[float]] = {cfg: [] for cfg in TREATMENT_ORDER}
    counts_by_treatment: Dict[str, List[int]] = {cfg: [] for cfg in TREATMENT_ORDER}

    plotted_ratios: List[float] = []
    for ratio, ratio_path in ratio_dirs:
        if ratio > 0.5:
            continue
        baseline = read_metric_last(ratio_path / "fixed" / SAMPLE_TYPE)
        if baseline in (None, 0.0):
            for cfg in TREATMENT_ORDER:
                means_by_treatment[cfg].append(float("nan"))
                cis_by_treatment[cfg].append(float("nan"))
                counts_by_treatment[cfg].append(0)
            continue

        plotted_ratios.append(ratio)
        for raw_cfg in TREATMENT_ORDER:
            cfg = normalize_treatment_name(raw_cfg)
            cfg_root = ratio_path / cfg
            trials_root = cfg_root / SAMPLE_TYPE
            if not trials_root.exists():
                trials_root = cfg_root
            if not cfg_root.exists() and not trials_root.exists():
                xs = []
            else:
                try:
                    entries = list(trials_root.iterdir()) if trials_root.exists() else []
                except Exception:
                    entries = []
                trials = [p for p in entries if p.is_dir() and p.name.isdigit()]
                if not trials:
                    trials = [cfg_root] if cfg_root.exists() else []
                xs = []
                for tdir in sorted(trials, key=lambda p: int(p.name) if p.name.isdigit() else p.name):
                    val = read_metric_last(tdir)
                    if val is None:
                        continue
                    xs.append(100.0 * (val - float(baseline)) / float(baseline))

            if xs:
                m, c = mean_ci_95(xs)
                means_by_treatment[raw_cfg].append(m)
                cis_by_treatment[raw_cfg].append(c)
                counts_by_treatment[raw_cfg].append(len(xs))
            else:
                means_by_treatment[raw_cfg].append(float("nan"))
                cis_by_treatment[raw_cfg].append(float("nan"))
                counts_by_treatment[raw_cfg].append(0)

    return plotted_ratios, means_by_treatment, cis_by_treatment, counts_by_treatment

# -----------------------------
# Main
# -----------------------------
def main() -> None:
    # Fonts & style — Nimbus Roman + readable sizes
    plt.rcParams.update({
        "font.family": "Nimbus Roman",
        "font.serif": ["Nimbus Roman", "Times New Roman", "Times", "DejaVu Serif", "Serif"],
        "axes.grid": True, "grid.linestyle": "--", "grid.alpha": 0.35,
        "font.size": 24, "axes.titlesize": 26, "axes.labelsize": 24,
        "legend.fontsize": 18,
        "xtick.labelsize": 15, "ytick.labelsize": 15,
        "figure.figsize": (14, 5.2),   # 3 side-by-side panels
    })

    fig, axes = plt.subplots(1, 3, sharey=True)
    axes = list(axes)  # ensure indexable list

    # Build each subplot
    handles_for_legend = None
    labels_for_legend = None

    for ax, ds in zip(axes, DATASETS):
        if ds not in _ALLOWED_DATASETS:
            ax.set_visible(False)
            continue

        plotted_ratios, means_by_treatment, cis_by_treatment, _ = collect_ratio_results_for_dataset(ds)
        if not plotted_ratios:
            ax.set_visible(False)
            continue

        # Plot lines
        for cfg in TREATMENT_ORDER:
            y = means_by_treatment[cfg]
            yerr = cis_by_treatment[cfg]
            valid_pts = [(i, yi, ei) for i, (yi, ei) in enumerate(zip(y, yerr)) if (yi == yi and ei == ei)]
            if not valid_pts:
                continue
            xi = [plotted_ratios[i] for (i, _, _) in valid_pts]
            yi = [yi for (_, yi, _) in valid_pts]
            # ei = [ei for (_, _, ei) in valid_pts]  # CI available if you want error bars

            ax.plot(xi, yi, marker="o", linewidth=3, label=TREATMENT_LABELS[cfg],
                    color=TREATMENT_COLORS[cfg], zorder=3)

        ax.set_xlabel("Treatment ratio")
        if ds == 'gowalla':
            ax.set_ylabel(YLABEL)
        # Title like "Yelp2018 - Recall@20"
        pretty_ds = DATASET_PRETTY.get(ds, ds)
        ax.set_title(f"{pretty_ds}")
        ax.axhline(0.0, color="black", linewidth=1.0)

        # Grab legend handles from the first populated subplot
        if handles_for_legend is None:
            h, l = ax.get_legend_handles_labels()
            if h:
                handles_for_legend, labels_for_legend = h, l

    # build legend once (keep a handle)
    if handles_for_legend:
        try:
            leg = fig.legend(handles_for_legend, labels_for_legend,
                             loc="upper center",
                             ncols=len(labels_for_legend),   # fallback to ncol below if needed
                             frameon=False,
                             bbox_to_anchor=(0.5, 1.0),
                             borderaxespad=0.3)
        except TypeError:
            leg = fig.legend(handles_for_legend, labels_for_legend,
                             loc="upper center",
                             ncol=len(labels_for_legend),
                             frameon=False,
                             bbox_to_anchor=(0.5, 1.0),
                             borderaxespad=0.3)

    # Reserve some headroom for the legend (10–15% usually works)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.88))   # top=0.88 leaves 12% for the legend

    # Ensure the legend is included in the saved figure and not clipped
    fig.savefig(OUT_PATH, bbox_inches="tight", bbox_extra_artists=(leg,), pad_inches=0.2)
    print(f"[OK] Saved figure: {OUT_PATH}")

if __name__ == "__main__":
    main()
