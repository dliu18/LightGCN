import os
import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

SMALL_SIZE = 24
MEDIUM_SIZE = 28
BIGGER_SIZE = 32
plt.rc('font', size=SMALL_SIZE, family="Nimbus Roman")
plt.rc('axes', titlesize=MEDIUM_SIZE)
plt.rc('axes', labelsize=MEDIUM_SIZE)
plt.rc('xtick', labelsize=SMALL_SIZE)
plt.rc('ytick', labelsize=SMALL_SIZE)
plt.rc('legend', fontsize=MEDIUM_SIZE)
plt.rc('figure', titlesize=BIGGER_SIZE)

def parse_event_file(event_file, target_tag):
    ea = EventAccumulator(event_file, size_guidance={'scalars': 0})
    ea.Reload()
    if target_tag in ea.Tags()['scalars']:
        events = ea.Scalars(target_tag)
        if events:
            return events[-1].value
    return None

def collect_baseline_vs_ours_metrics():
    rows = []
    recall_tag = "Test/Recall__20__2000_"
    pob_tag = "Test/Popularity_Opportunity_Bias_20"
    base_dir = "../../runs/final-1"

    for model in ["mf", "lgn"]:
        model_path = os.path.join(base_dir, model)

        if not os.path.isdir(model_path):
            continue

        for dataset in os.listdir(model_path):
            dataset_path = os.path.join(model_path, dataset)
            if not os.path.isdir(dataset_path):
                continue

            # Ours variant
            ours_path = os.path.join(dataset_path, "ours")
            if os.path.isdir(ours_path):
                recall_value, pob_value = None, None
                recall_dir = os.path.join(ours_path, "Test", "Recall@[20, 2000]", "20")
                if os.path.isdir(recall_dir):
                    for f in os.listdir(recall_dir):
                        if f.startswith("events.out.tfevents"):
                            recall_value = parse_event_file(os.path.join(recall_dir, f), recall_tag)
                            break
                for f in os.listdir(ours_path):
                    if f.startswith("events.out.tfevents"):
                        pob_value = parse_event_file(os.path.join(ours_path, f), pob_tag)
                        break
                if recall_value is not None and pob_value is not None:
                    rows.append({
                        "dataset": dataset,
                        "model": model,
                        "variant": "ours",
                        "Recall@20": recall_value,
                        "POB@20": pob_value
                    })

            # Baselines
            vanilla_path = os.path.join(dataset_path, "vanilla", "alpha")
            if not os.path.isdir(vanilla_path):
                continue

            for alpha in os.listdir(vanilla_path):
                alpha_path = os.path.join(vanilla_path, alpha, "beta")
                if not os.path.isdir(alpha_path):
                    continue

                for beta in os.listdir(alpha_path):
                    beta_path = os.path.join(alpha_path, beta)
                    if not os.path.isdir(beta_path):
                        continue

                    recall_value, pob_value = None, None
                    recall_dir = os.path.join(beta_path, "Test", "Recall@[20, 2000]", "20")
                    if os.path.isdir(recall_dir):
                        for f in os.listdir(recall_dir):
                            if f.startswith("events.out.tfevents"):
                                recall_value = parse_event_file(os.path.join(recall_dir, f), recall_tag)
                                break
                    for f in os.listdir(beta_path):
                        if f.startswith("events.out.tfevents"):
                            pob_value = parse_event_file(os.path.join(beta_path, f), pob_tag)
                            break

                    if recall_value is not None and pob_value is not None:
                        rows.append({
                            "dataset": dataset,
                            "model": model,
                            "variant": f"baseline (α={alpha}, β={beta})",
                            "Recall@20": recall_value,
                            "POB@20": pob_value
                        })

    return pd.DataFrame(rows)

def plot_scatter(df):
    datasets = ["gowalla", "yelp2018", "amazon-book"]
    dataset_titles = {
        "gowalla": "Gowalla",
        "yelp2018": "Yelp2018",
        "amazon-book": "Amazon-Book"
    }

    fig, axes = plt.subplots(2, len(datasets), figsize=(5 * len(datasets), 10))
    for col_idx, dataset in enumerate(datasets):
        for row_idx, model in enumerate(["mf", "lgn"]):
            ax = axes[row_idx, col_idx]
            subset = df[(df["dataset"] == dataset) & (df["model"] == model)]

            # Plot baselines
            baseline_data = subset[subset["variant"].str.startswith("baseline")]
            if not baseline_data.empty:
                ax.scatter(
                    baseline_data["Recall@20"],
                    baseline_data["POB@20"],
                    marker='x',
                    color="#66a61e",
                    s=100,
                    label="Post-Processing Baseline" if (col_idx == 0 and row_idx == 0) else None
                )

            # Plot ours
            ours_data = subset[subset["variant"] == "ours"]
            if not ours_data.empty:
                ax.scatter(
                    ours_data["Recall@20"],
                    ours_data["POB@20"],
                    marker='s',
                    color="#d95f02",
                    s=200,
                    label="Ours" if (col_idx == 0 and row_idx == 0) else None
                )

            if row_idx == 0:
                ax.set_title(dataset_titles[dataset])
            if col_idx == 0:
                ax.set_ylabel("Bias")
            ax.set_xlabel("Recall@20")

            if not subset.empty:
                ideal_x = 1.1 * subset["Recall@20"].max()
                ideal_y = 0.9 * subset["POB@20"].min()
                ax.scatter(ideal_x, ideal_y, marker='*', color='black', s=250, label="Ideal" if (col_idx == 0 and row_idx == 0) else None)

    fig.text(0.09, 0.7, r"$\mathbf{MF}$", va='center', ha='right')
    fig.text(0.09, 0.28, r"$\mathbf{LGN}$", va='center', ha='right')

    fig.legend(loc="upper center", ncol=3, fontsize=24)
    plt.tight_layout(rect=[0.1, 0, 1, 0.92])
    plt.savefig("../../../outputs/final_eval/post_processing_baseline_vs_ours.pdf")
    plt.close()

def generate_latex_table(df):
    # Get best baseline by Recall@20
    best_baselines = (
        df[df["variant"].str.startswith("baseline")]
        .sort_values("Recall@20", ascending=False)
        .groupby(["dataset", "model"])
        .first()
        .reset_index()
    )

    # Get ours
    ours = df[df["variant"] == "ours"]

    # Merge
    merged = pd.merge(
        ours, best_baselines, on=["dataset", "model"], suffixes=("_ours", "_baseline")
    )

    # Compute percent change
    merged["RecallChange"] = 100 * (merged["Recall@20_ours"] - merged["Recall@20_baseline"]) / merged["Recall@20_baseline"]
    merged["POBChange"] = 100 * (merged["POB@20_ours"] - merged["POB@20_baseline"]) / merged["POB@20_baseline"]

    # Round values
    merged = merged.round(4)

    # Format rows
    rows = []
    for _, row in merged.iterrows():
        dataset = row["dataset"].replace("-", " ").title()
        model = row["model"].upper()
        recall_base = row["Recall@20_baseline"]
        pob_base = row["POB@20_baseline"]
        recall_ours = row["Recall@20_ours"]
        pob_ours = row["POB@20_ours"]
        recall_delta = row["RecallChange"]
        pob_delta = row["POBChange"]

        rows.append(
            f"{dataset} & {model} & "
            f"{recall_base:.4f} & {pob_base:.4f} & "
            f"{recall_ours:.4f} & {pob_ours:.4f} & "
            f"\\textbf{{({recall_delta:+.1f}\\%)}} & \\textbf{{({pob_delta:+.1f}\\%)}} \\\\"
        )

    # LaTeX table string
    table = r"""
    \begin{table}[ht]
    \centering
    \caption{Comparison of our variant to the best-performing post-processing baseline (highest Recall@20) across datasets and models.}
    \begin{tabular}{llrrrrrr}
    \toprule
    Dataset & Model & Recall$_\mathrm{base}$ & POB$_\mathrm{base}$ & Recall$_\mathrm{ours}$ & POB$_\mathrm{ours}$ & $\Delta$ Recall & $\Delta$ Bias \\
    \midrule
    """ + "\n".join(rows) + r"""
    \bottomrule
    \end{tabular}
    \label{tab:best_baseline_comparison}
    \end{table}
    """

    with open("../../../outputs/final_eval/post_processing_baseline_vs_ours.tex", "w") as f:
        f.write(table)




if __name__ == "__main__":
    csv_path = "../../../outputs/final_eval/post_processing_baseline_vs_ours.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        print(f"Loaded cached results from {csv_path}")
    else:
        df = collect_baseline_vs_ours_metrics()
        df.to_csv(csv_path, index=False)
        print(f"Saved new results to {csv_path}")
    plot_scatter(df)
    generate_latex_table(df)
