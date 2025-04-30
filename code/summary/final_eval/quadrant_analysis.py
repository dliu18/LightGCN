import os
import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

SMALL_SIZE = 24
MEDIUM_SIZE = 28
BIGGER_SIZE = 30
plt.rc('font', size=SMALL_SIZE, family="Nimbus Roman")
plt.rc('axes', titlesize=BIGGER_SIZE)
plt.rc('axes', labelsize=MEDIUM_SIZE)
plt.rc('xtick', labelsize=SMALL_SIZE)
plt.rc('ytick', labelsize=SMALL_SIZE)
plt.rc('legend', fontsize=SMALL_SIZE)
plt.rc('figure', titlesize=BIGGER_SIZE)

colors = [
    "#1b9e77",  # LightGCN
    "#d95f02",  # Matrix Factorization
]

our_variant = "ours"  # or "our-non-zero-beta"

def parse_event_file(event_file, target_tag):
    ea = EventAccumulator(event_file, size_guidance={'scalars': 0})
    ea.Reload()
    if target_tag in ea.Tags()['scalars']:
        events = ea.Scalars(target_tag)
        if events:
            return events[-1].value
    return None

def collect_quadrant_metrics(base_dir="../../runs/final"):
    rows = []
    quadrant_mapping = {
        "Low_Low_Recall@[20, 2000]": ("Low-Niche", "Low_Low_Recall__20__2000_", "low_low_Popularity_Opportunity_Bias_20"),
        "Low_High_Recall@[20, 2000]": ("Low-Mainstream", "Low_High_Recall__20__2000_", "low_high_Popularity_Opportunity_Bias_20"),
        "High_Low_Recall@[20, 2000]": ("Power-Niche", "High_Low_Recall__20__2000_", "high_low_Popularity_Opportunity_Bias_20"),
        "High_High_Recall@[20, 2000]": ("Power-Mainstream", "High_High_Recall__20__2000_", "high_high_Popularity_Opportunity_Bias_20"),
    }

    for model in ["lgn", "mf"]:
        model_path = os.path.join(base_dir, model)
        if not os.path.isdir(model_path):
            continue

        for dataset_name in os.listdir(model_path):
            dataset_path = os.path.join(model_path, dataset_name)
            if not os.path.isdir(dataset_path):
                continue

            for variant in os.listdir(dataset_path):
                variant_path = os.path.join(dataset_path, variant)
                if not os.path.isdir(variant_path):
                    continue

                quadrant_path = os.path.join(variant_path, "Quadrants")
                if not os.path.isdir(quadrant_path):
                    continue

                quadrant_vals = {}

                for folder_name, (clean_name, recall_tag, pop_bias_tag) in quadrant_mapping.items():
                    # Recall
                    recall_dir = os.path.join(quadrant_path, folder_name, "20")
                    if os.path.isdir(recall_dir):
                        for f in os.listdir(recall_dir):
                            if f.startswith("events.out.tfevents"):
                                event_file = os.path.join(recall_dir, f)
                                tag = f"Quadrants/{recall_tag}"
                                val = parse_event_file(event_file, tag)
                                quadrant_vals[f"{clean_name} Recall"] = val

                    # POB from top level
                    for f in os.listdir(variant_path):
                        if f.startswith("events.out.tfevents"):
                            top_event_file = os.path.join(variant_path, f)
                            tag = f"Quadrants/{pop_bias_tag}"
                            val = parse_event_file(top_event_file, tag)
                            quadrant_vals[f"{clean_name} POB"] = val
                            break

                row = {
                    "model": model,
                    "dataset": dataset_name,
                    "model_variant": variant,
                    **quadrant_vals
                }
                rows.append(row)

    return pd.DataFrame(rows)

def plot_quadrant_metrics(df):
    quadrant_order = [
        ("Power-Mainstream Recall", "Power-Mainstream POB"),
        ("Low-Mainstream Recall", "Low-Mainstream POB"),
        ("Power-Niche Recall", "Power-Niche POB"),
        ("Low-Niche Recall", "Low-Niche POB"),
    ]
    datasets = ["gowalla", "yelp2018", "amazon-book"]
    dataset_titles = {
        "gowalla": "Gowalla",
        "yelp2018": "Yelp2018",
        "amazon-book": "Amazon-Book"
    }

    def make_bar_plot(metric_type: str, filename: str):
        fig, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 4))
        if len(datasets) == 1:
            axes = [axes]

        for idx, (ax, dataset) in enumerate(zip(axes, datasets)):
            y = range(len(quadrant_order))
            width = 0.6

            ours = df[(df['dataset'] == dataset) & (df['model'] == "lgn") & (df['model_variant'] == our_variant)]
            vanilla = df[(df['dataset'] == dataset) & (df['model'] == "lgn") & (df['model_variant'] == "vanilla")]

            if ours.empty or vanilla.empty:
                continue

            base_col = 0 if metric_type == "recall" else 1
            vanilla_vals = [vanilla[q[base_col]].iloc[0] for q in quadrant_order]
            ours_vals = [ours[q[base_col]].iloc[0] for q in quadrant_order]

            percent_changes = [
                100 * (o - v) / v if v != 0 else 0
                for o, v in zip(ours_vals, vanilla_vals)
            ]

            # Assign color based on whether the user group is "Mainstream" or "Niche"
            bar_colors = [
                colors[0] if "Mainstream" in q[0] else colors[1]
                for q in quadrant_order
            ]

            ax.barh(
                y,
                percent_changes,
                height=width,
                color=bar_colors,
                edgecolor='black',
                linewidth=2,
            )

            ax.axvline(0, color="black", linestyle="--", linewidth=1)
            ax.set_title(dataset_titles[dataset])
            ax.set_yticks(y)
            if idx == 0:
                ax.set_yticklabels([
                    q[0].replace(" Recall", "").replace("Low", "Light").replace("-", " ")
                    for q in quadrant_order
                ])
                ax.set_ylabel("Quadrant")
            else:
                ax.set_yticklabels([])
            ax.set_xlabel("Recall@20 % Change" if metric_type == "recall" else "Popularity Bias % Change")

        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
    make_bar_plot("recall", "../../../outputs/final_eval/quadrant_recalls_percent_change.pdf")
    make_bar_plot("pob", "../../../outputs/final_eval/quadrant_biases_percent_change.pdf")


if __name__ == "__main__":
    csv_path = "../../../outputs/final_eval/quadrant_metrics.csv"

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        print(f"Loaded cached results from {csv_path}")
    else:
        df = collect_quadrant_metrics()
        df.to_csv(csv_path, index=False)
        print(f"Saved new results to {csv_path}")

    plot_quadrant_metrics(df)

