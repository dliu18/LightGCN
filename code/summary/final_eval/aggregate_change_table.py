import os
import math
import argparse
import numpy as np
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# ----------- Config -----------
models = ["mf", "lgn"]  # order matters: MF first, then LGN
# Each base_dir should contain <model>/<dataset>/<variant>/...
# Variants include "vanilla", "vanilla-bpr", "ours", and other baselines (each with ONE execution, possibly nested).
base_dirs = ["../../runs/www-final"]

# Core (non-baseline) variants to always include explicitly, in display order
core_variants = ["vanilla", "vanilla-bpr", "ours"]

target_tags = [
    "Test/Recall__20__10000_",
    "Test/Precision__20__10000_",
    "Test/NDCG__20__10000_",
    "Test/Popularity_Opportunity_Bias_20",
    "Test/Coverage_20",
    "Test/Niche-Recall_20",
]

pretty_names = {
    "Test/Recall__20__10000_": "Recall@20",
    "Test/Precision__20__10000_": "Precision@20",
    "Test/NDCG__20__10000_": "NDCG@20",
    "Test/Popularity_Opportunity_Bias_20": "POB@20",
    "Test/Coverage_20": "Coverage@20",
    "Test/Niche-Recall_20": "Niche Recall@20",
}

# Pretty dataset names for LaTeX display
DATASET_PRETTY = {
    "gowalla": "Gowalla",
    "yelp2018": "Yelp2018",
    "amazon-book": "Amazon-Book",
}

# Total items per dataset (for Coverage percentage conversion)
DATASET_NUM_ITEMS = {
    "gowalla": 40981,
    "yelp2018": 38048,
    "amazon-book": 91599,
}

# Pretty names for variants/baselines in the LaTeX "Variant" column
VARIANT_PRETTY = {
    "vanilla": "Vanilla",
    "vanilla-bpr": "Vanilla-BPR",
    "ours": "Ours",
    "ipw": "IPW",
    "pop_reg": "Pop. Reg.",
    "pop_comp": "Pop. Comp.",
}

# For "best" highlighting: True => larger-is-better; False => smaller-is-better
BEST_IS_MAX = {
    "Test/Recall__20__10000_": True,
    "Test/Precision__20__10000_": True,
    "Test/NDCG__20__10000_": True,
    "Test/Popularity_Opportunity_Bias_20": False,  # POB => smaller is better
    "Test/Coverage_20": True,                       # after converting to %
    "Test/Niche-Recall_20": True,
}
# --------------------------------


def parse_event_file(event_file, target_tags):
    """Parse one events file for the final value of tags in target_tags."""
    ea = EventAccumulator(event_file, size_guidance={"scalars": 0})
    try:
        ea.Reload()
    except Exception:
        return {}
    final_values = {}
    scalar_tags = ea.Tags().get("scalars", [])
    for tag in scalar_tags:
        if tag in target_tags:
            events = ea.Scalars(tag)
            if events:
                final_values[tag] = events[-1].value
    return final_values


def iter_event_files_under_test(test_root):
    """
    Yield event files under:
      <run_root>/Test/<metric_folder>/20/events.out.tfevents.*
    """
    if not os.path.isdir(test_root):
        return
    try:
        for metric_folder in os.listdir(test_root):
            metric_path = os.path.join(test_root, metric_folder, "20")
            if not os.path.isdir(metric_path):
                continue
            for f in os.listdir(metric_path):
                if f.startswith("events.out.tfevents"):
                    yield os.path.join(metric_path, f)
    except Exception:
        return


def iter_root_event_files(run_root):
    """
    Yield event files directly under a run root (the directory that contains 'Test' and/or top-level events).
    """
    if not os.path.isdir(run_root):
        return
    try:
        for f in os.listdir(run_root):
            if f.startswith("events.out.tfevents"):
                yield os.path.join(run_root, f)
    except Exception:
        return


def _deepest_digit_component(rel_path: str) -> int:
    """
    From a relative path string (posix-like), return the integer value of the
    deepest path component that is all digits. If none is found, return 0.
    """
    parts = [p for p in rel_path.replace("\\", "/").split("/") if p]
    trial_id = 0
    for p in parts:
        if p.isdigit():
            trial_id = int(p)
    return trial_id


def find_run_leaf_dirs(variant_path, max_trial=None):
    """
    Robustly find the directory (or directories) under 'variant_path' that act as the run root(s),
    i.e., the directory that actually contains a 'Test' subdir and/or top-level event files.

    This handles layouts such as:
      variant/0/Test/...
      variant/0/alpha/1e-06/beta/0.25/Test/...
    and avoids descending into 'Test'/'Quadrants' subtrees.

    If max_trial is not None, only include run roots whose deepest digit-named component
    (relative to variant_path) is <= max_trial. If there is no digit component, treat as 0.
    """
    leaf_dirs = set()
    if not os.path.isdir(variant_path):
        return []

    variant_root_abs = os.path.abspath(variant_path)

    for dirpath, dirnames, filenames in os.walk(variant_root_abs, topdown=True):
        # Don’t traverse into these (we only need the run root; metrics helpers read inside Test)
        for prune in ["Test", "Quadrants"]:
            if prune in dirnames:
                dirnames.remove(prune)

        has_test = os.path.isdir(os.path.join(dirpath, "Test"))
        has_events = any(fn.startswith("events.out.tfevents") for fn in filenames)

        if has_test or has_events:
            rel = os.path.relpath(dirpath, variant_root_abs)
            trial_id = _deepest_digit_component(rel)
            if (max_trial is None) or (trial_id <= max_trial):
                leaf_dirs.add(os.path.abspath(dirpath))

    # Return sorted list (depth, then alpha) — we average across all anyway
    return sorted(leaf_dirs, key=lambda p: (p.count(os.sep), p))


def aggregate_metrics_over_run_root(run_root):
    """
    Read metrics for a single run root directory by parsing:
      - Test hierarchy scalars (Recall/Precision/NDCG/Coverage/Niche-Recall)
      - Root events (POB lives at the run root events)
    Returns a dict[tag] = value for this run root.
    """
    metrics = {}

    # 1) Test hierarchy scalars
    test_path = os.path.join(run_root, "Test")
    for event_file in iter_event_files_under_test(test_path):
        parsed_vals = parse_event_file(event_file, target_tags)
        metrics.update(parsed_vals)

    # 2) Root event file(s) (e.g., POB)
    for event_file in iter_root_event_files(run_root):
        parsed_vals = parse_event_file(event_file, target_tags)
        metrics.update(parsed_vals)

    return metrics


def _mean_std_n_ci(values):
    """
    Return mean, std (sample), n, and 95% CI half-width (t-based; CI=0 if n<=1).
    """
    n = len(values)
    if n == 0:
        return None, None, 0, None
    if n == 1:
        return float(values[0]), 0.0, 1, 0.0
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1))
    se = std / math.sqrt(n)
    try:
        from scipy.stats import t as student_t
        tcrit = float(student_t.ppf(0.975, df=n - 1))
    except Exception:
        tcrit = 1.96
    ci = tcrit * se
    return mean, std, n, float(ci)


def aggregate_variant_recursive(variant_path, max_trial=None):
    """
    Aggregate metrics for a variant by:
      - Locating all run roots (directories that contain 'Test' and/or top-level events)
        anywhere under the variant path (filtered by max_trial if provided).
      - For each run root, read metrics.
      - Average across all run roots (trials).
    Returns per-tag mean/std/n/ci across found run roots.
    """
    run_roots = find_run_leaf_dirs(variant_path, max_trial=max_trial)
    if not run_roots:
        # Legacy fallback: if variant itself has events/test
        has_test = os.path.isdir(os.path.join(variant_path, "Test"))
        has_events = False
        try:
            has_events = any(
                fn.startswith("events.out.tfevents")
                for fn in os.listdir(variant_path)
                if os.path.isfile(os.path.join(variant_path, fn))
            )
        except Exception:
            pass
        if has_test or has_events:
            # Respect max_trial even here: treat as trial_id 0
            if (max_trial is None) or (0 <= max_trial):
                run_roots = [variant_path]

    vals_by_tag = {t: [] for t in target_tags}
    for rr in run_roots:
        metrics = aggregate_metrics_over_run_root(rr)
        for t in target_tags:
            if t in metrics:
                vals_by_tag[t].append(metrics[t])

    row = {}
    for t in target_tags:
        mean, std, n, ci = _mean_std_n_ci(vals_by_tag[t])
        row[f"{t}_mean"] = mean
        row[f"{t}_std"] = std
        row[f"{t}_n"] = n
        row[f"{t}_ci"] = ci
    return row


def collect_metrics_all_trials_for_model(base_dirs, model, max_trial=None):
    """
    Collects, for each dataset:
      - vanilla:       stats over its execution(s)
      - vanilla-bpr:   stats over its execution(s)
      - ours:          stats over its execution(s)
      - each baseline: stats over its single-parameter execution (robust to nested folders)
    Returns a DataFrame with columns:
      model, dataset, model_variant, <tag>_{mean,std,n,ci}
    """
    records = []

    for base_dir in base_dirs:
        model_root = os.path.join(base_dir, model)
        if not os.path.isdir(model_root):
            continue

        for dataset_name in os.listdir(model_root):
            dataset_path = os.path.join(model_root, dataset_name)
            if not os.path.isdir(dataset_path):
                continue

            # Collect core variants (explicit order)
            for variant in core_variants:
                variant_path = os.path.join(dataset_path, variant)
                if not os.path.isdir(variant_path):
                    continue
                row = {"model": model, "dataset": dataset_name, "model_variant": variant}
                row.update(aggregate_variant_recursive(variant_path, max_trial=max_trial))
                records.append(row)

            # Baselines: everything else (skip core variants)
            for variant in os.listdir(dataset_path):
                if variant in core_variants:
                    continue
                variant_path = os.path.join(dataset_path, variant)
                if not os.path.isdir(variant_path):
                    continue

                stats_row = aggregate_variant_recursive(variant_path, max_trial=max_trial)
                if not any(stats_row.get(f"{t}_mean") is not None for t in target_tags):
                    continue

                row = {"model": model, "dataset": dataset_name, "model_variant": variant}
                row.update(stats_row)
                records.append(row)

    return pd.DataFrame(records)


def fmt_value_with_ci(tag, dataset_name, mean, ci):
    """
    Format a metric value with a smaller-font 95% CI.
    - Coverage: convert to percentage of total items for the dataset.
    - Others: 4 decimals.
    - If mean is None => "--"
    - If n==1 (ci==0) display ± 0.0000 (still smaller font).
    """
    if mean is None:
        return "--"

    # Coverage as percentage
    if tag == "Test/Coverage_20":
        total = DATASET_NUM_ITEMS.get(dataset_name)
        if total and total > 0:
            pct_mean = (mean / total) * 100.0
            pct_ci = (ci / total) * 100.0 if (ci is not None) else None
            base = f"{pct_mean:.1f}\\%"
            if pct_ci is None:
                return base
            return f"{base}{{\\scriptsize\\,(\\,$\\pm$\\,{pct_ci:.1f}\\%\\,)}}"
        else:
            base = f"{int(round(mean))}"
            if ci is None:
                return base
            return f"{base}{{\\scriptsize\\,(\\,$\\pm$\\,{ci:.1f}\\,)}}"

    # Other metrics (4 decimals)
    base = f"{mean:.4f}"
    if ci is None:
        return base
    return f"{base}{{\\scriptsize\\,(\\,$\\pm$\\,{ci:.4f}\\,)}}"


def latex_highlight_if_best(cell_str, is_best):
    """Wrap with bold+underline if is_best and cell_str is not '--'."""
    if is_best and cell_str != "--":
        return f"\\textbf{{\\underline{{{cell_str}}}}}"
    return cell_str


def pretty_variant_name(name: str) -> str:
    """Map folder variant names to display names (Vanilla, Vanilla-BPR, Ours, IPW, Pop. Reg., Pop. Comp., etc.)."""
    if name in VARIANT_PRETTY:
        return VARIANT_PRETTY[name]
    return name.replace("_", " ")


def build_latex_table(df):
    """
    Build a LaTeX table where MF rows come first, then LGN.
    Per (model, dataset) block:
      - Vanilla
      - Vanilla-BPR
      - Ours
      - All baselines (alphabetical)
    Bold+underline the best per metric per block (using BEST_IS_MAX).
    Insert a partial horizontal rule between blocks that spans from the Variant column to the end.
    """
    if df is None or df.empty:
        return ""

    # Sort: MF first, then LGN; within model -> dataset -> variant (vanilla, vanilla-bpr, ours, then baselines alpha)
    model_order = {m: i for i, m in enumerate(models)}
    df["__model_order"] = df["model"].map(model_order).fillna(999)

    def _variant_key(v):
        if v == "vanilla":
            return (0, "")
        if v == "vanilla-bpr":
            return (1, "")
        if v == "ours":
            return (2, "")
        return (3, v)

    df["__variant_sort"] = df["model_variant"].map(_variant_key)
    df.sort_values(["__model_order", "dataset", "__variant_sort"], inplace=True)

    # LaTeX header (with arrows)
    def header_with_arrow(tag):
        arrow = "$\\uparrow$" if BEST_IS_MAX[tag] else "$\\downarrow$"
        return f"\\textbf{{{pretty_names[tag]} ({arrow})}}"

    tab_spec = "lll" + "c" * len(target_tags)
    header_cells = [
        "\\textbf{Dataset}",
        "\\textbf{Model}",
        "\\textbf{Variant}",
    ] + [header_with_arrow(t) for t in target_tags]

    latex_lines = []
    latex_lines.append("\\begin{table*}[t]")
    latex_lines.append("\\centering")
    latex_lines.append("\\resizebox{\\linewidth}{!}{%")
    latex_lines.append(f"\\begin{{tabular}}{{{tab_spec}}}")
    latex_lines.append("\\toprule")
    latex_lines.append(" & ".join(header_cells) + " \\\\")
    latex_lines.append("\\midrule")

    last_col = 3 + len(target_tags)  # 1:Dataset, 2:Model, 3:Variant, 4..last: metrics

    # Emit per (model, dataset) block
    for model in models:
        mdf = df[df["model"] == model]
        if mdf.empty:
            continue
        for dataset in mdf["dataset"].unique():
            block = mdf[mdf["dataset"] == dataset].copy()
            if block.empty:
                continue

            # Best index per metric for highlighting
            idx_list = block.index.tolist()
            best_idx_for_tag = {}
            for tag in target_tags:
                want_max = BEST_IS_MAX[tag]
                best_idx, best_val = None, None
                for idx in idx_list:
                    v = block.at[idx, f"{tag}_mean"] if f"{tag}_mean" in block.columns else None
                    if v is None:
                        continue
                    v = float(v)
                    if best_val is None or (want_max and v > best_val) or ((not want_max) and v < best_val):
                        best_val, best_idx = v, idx
                if best_idx is not None:
                    best_idx_for_tag[tag] = best_idx

            # Order rows: Vanilla, Vanilla-BPR, Ours, then baselines alphabetical
            vanilla_row     = block[block["model_variant"] == "vanilla"]
            vanilla_bpr_row = block[block["model_variant"] == "vanilla-bpr"]
            ours_row        = block[block["model_variant"] == "ours"]
            base_rows       = block[
                (block["model_variant"] != "vanilla")
                & (block["model_variant"] != "vanilla-bpr")
                & (block["model_variant"] != "ours")
            ].sort_values(by="model_variant", kind="stable")

            ordered_rows = []
            if not vanilla_row.empty:
                ordered_rows.append(vanilla_row.iloc[0])
            if not vanilla_bpr_row.empty:
                ordered_rows.append(vanilla_bpr_row.iloc[0])
            if not ours_row.empty:
                ordered_rows.append(ours_row.iloc[0])
            for _, r in base_rows.iterrows():
                ordered_rows.append(r)

            if not ordered_rows:
                continue

            pretty_dataset = DATASET_PRETTY.get(dataset, dataset)
            model_text = model.upper()
            first = True
            for r in ordered_rows:
                row_cells = []
                row_cells.append(pretty_dataset if first else "")
                row_cells.append(model_text if first else "")
                variant_disp = pretty_variant_name(r["model_variant"])
                row_cells.append(variant_disp)

                for tag in target_tags:
                    mean = r.get(f"{tag}_mean")
                    ci   = r.get(f"{tag}_ci")
                    cell = fmt_value_with_ci(tag, dataset, mean, ci)
                    is_best = (best_idx_for_tag.get(tag) == r.name)
                    row_cells.append(latex_highlight_if_best(cell, is_best))

                latex_lines.append(" & ".join(row_cells) + " \\\\")
                first = False

            # Partial horizontal rule between blocks (from Variant col to end)
            latex_lines.append(f"\\cmidrule(lr){{3-{last_col}}}")

    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}%")
    latex_lines.append("}")
    latex_lines.append("\\caption{Performance (means $\\pm$ 95\\% CI across trials). For each (model, dataset), rows list \\emph{Vanilla}, \\emph{Vanilla-BPR}, \\emph{Ours}, then baselines (each baseline has a single parameter execution). Best per metric in each block is \\textbf{bold and underlined}. Coverage is reported as a percentage of total items for the dataset.}")
    latex_lines.append("\\label{tab:results}")
    latex_lines.append("\\end{table*}")

    return "\n".join(latex_lines)


def main():
    parser = argparse.ArgumentParser(description="Summarize metrics into a LaTeX table with optional max trial filtering.")
    parser.add_argument("--max-trial", type=int, default=None,
                        help="Maximum integer trial ID to include (filters run roots by their deepest digit-named path component).")
    args = parser.parse_args()
    max_trial = args.max_trial

    # Collect per-model, then concatenate (skipping empty)
    dfs = []
    for m in models:
        try:
            df_m = collect_metrics_all_trials_for_model(base_dirs, m, max_trial=max_trial)
            if df_m is not None and not df_m.empty:
                dfs.append(df_m)
        except Exception as e:
            print(f"[WARN] Skipping model '{m}' due to error: {e}")

    if dfs:
        df = pd.concat(dfs, ignore_index=True)
    else:
        df = pd.DataFrame(
            columns=["model", "dataset", "model_variant"]
            + [f"{t}_mean" for t in target_tags]
            + [f"{t}_std" for t in target_tags]
            + [f"{t}_n" for t in target_tags]
            + [f"{t}_ci" for t in target_tags]
        )

    # Ensure output dirs exist
    os.makedirs("../../../outputs/final_eval/www", exist_ok=True)

    # Save combined CSV (MF first, then LGN), with variant sort: vanilla, vanilla-bpr, ours, baselines
    model_order = {m: i for i, m in enumerate(models)}
    if not df.empty:
        df["__model_order"] = df["model"].map(model_order).fillna(999)

        def _variant_key(v):
            if v == "vanilla": return (0, "")
            if v == "vanilla-bpr": return (1, "")
            if v == "ours": return (2, "")
            return (3, v)

        df["__variant_sort"] = df["model_variant"].map(_variant_key)
        df.sort_values(["__model_order", "dataset", "__variant_sort"], inplace=True)
        df.drop(columns=["__model_order", "__variant_sort"], inplace=True, errors="ignore")

    df.to_csv("../../../outputs/final_eval/www/aggregate_all.csv", index=False)

    # Also keep per-model CSVs (optional)
    for m in models:
        mdf = df[df["model"] == m]
        if not mdf.empty:
            mdf.to_csv(f"../../../outputs/final_eval/www/aggregate_{m}.csv", index=False)

    # Build LaTeX
    latex_table = build_latex_table(df)
    with open("../../../outputs/final_eval/www/aggregate_all.txt", "w") as f:
        f.write(latex_table)


if __name__ == "__main__":
    main()
