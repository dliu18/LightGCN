import os
import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

our_varaint = "ours"
# our_varaint = "our-non-zero-beta"

SMALL_SIZE = 24
MEDIUM_SIZE = 28
BIGGER_SIZE = 32
plt.rc('font', size=SMALL_SIZE, family = "Nimbus Roman")
plt.rc('axes', titlesize=MEDIUM_SIZE)
plt.rc('axes', labelsize=MEDIUM_SIZE)
plt.rc('xtick', labelsize=SMALL_SIZE)
plt.rc('ytick', labelsize=SMALL_SIZE)
plt.rc('legend', fontsize=MEDIUM_SIZE)
plt.rc('figure', titlesize=BIGGER_SIZE)

colors = [
    "#1b9e77",  # vanilla
    "#d95f02",  # ours
    "#7570b3",  # only-items
    "#e7298a",  # only-users
    "#66a61e",  # extra if needed
]

def parse_event_file(event_file, target_tag):
    ea = EventAccumulator(event_file, size_guidance={'scalars': 0})
    ea.Reload()
    if target_tag in ea.Tags()['scalars']:
        events = ea.Scalars(target_tag)
        if events:
            return events[-1].value
    return None

def collect_metrics_for_scatter(base_dir="../../runs/final"):
    rows = []
    recall_tag = "Test/Recall__20__2000_"
    pob_tag = "Test/Popularity_Opportunity_Bias_20"

    for model in ["mf", "lgn"]:
        model_path = os.path.join(base_dir, model)
        if not os.path.isdir(model_path):
            continue

        for dataset_name in os.listdir(model_path):
            dataset_path = os.path.join(model_path, dataset_name)
            if not os.path.isdir(dataset_path):
                continue
            for variant in os.listdir(dataset_path):
                if variant not in ["vanilla", our_varaint, "only-items", "only-users"]:
                    continue
                variant_path = os.path.join(dataset_path, variant)
                if not os.path.isdir(variant_path):
                    continue

                recall_value = None
                pob_value = None

                test_path = os.path.join(variant_path, "Test", "Recall@[20, 2000]", "20")
                if os.path.isdir(test_path):
                    for f in os.listdir(test_path):
                        if f.startswith("events.out.tfevents"):
                            event_file = os.path.join(test_path, f)
                            recall_value = parse_event_file(event_file, recall_tag)
                            break

                for f in os.listdir(variant_path):
                    if f.startswith("events.out.tfevents"):
                        event_file = os.path.join(variant_path, f)
                        pob_value = parse_event_file(event_file, pob_tag)
                        break

                row = {
                    "dataset": dataset_name,
                    "model": model,
                    "model_variant": variant,
                    "Recall@20": recall_value,
                    "POB@20": pob_value,
                }
                rows.append(row)

    return pd.DataFrame(rows)

def plot_scatter_metrics(df):
    datasets = ["gowalla", "yelp2018", "amazon-book"]
    dataset_titles = {
        "gowalla": "Gowalla",
        "yelp2018": "Yelp2018",
        "amazon-book": "Amazon-Book"
    }

    markers = {
        "vanilla": "o",
        our_varaint: "s",
        "only-items": "^",
        "only-users": "P",
    }

    label_names = {
        "vanilla": "Vanilla",
        our_varaint: "Ours",
        "only-items": "Only Items",
        "only-users": "Only Users"
    }

    fig, axes = plt.subplots(2, len(datasets), figsize=(5 * len(datasets), 7))
    handles, labels = [], []

    for col_idx, dataset in enumerate(datasets):
        for row_idx, model in enumerate(["mf", "lgn"]):
            ax = axes[row_idx, col_idx]
            subset = df[(df['dataset'] == dataset) & (df['model'] == model)]
            variant_idx = 0

            for variant, marker in markers.items():
                variant_data = subset[subset['model_variant'] == variant]
                if not variant_data.empty:
                    sc = ax.scatter(
                        variant_data['Recall@20'],
                        variant_data['POB@20'],
                        marker=marker,
                        color=colors[variant_idx],
                        s=200
                    )
                    if label_names[variant] not in labels:
                        handles.append(sc)
                        labels.append(label_names[variant])
                variant_idx += 1

            # Axis titles and labels
            if row_idx == 0:
                ax.set_title(dataset_titles[dataset])
            if col_idx == 0:
                ax.set_ylabel("Bias")
            ax.set_xlabel("Recall@20")

            # Add black star per subplot
            if not subset.empty:
                ideal_x = 1.1 * subset['Recall@20'].max()
                ideal_y = 0.9 * subset['POB@20'].min()
                star = ax.scatter(ideal_x, ideal_y, marker='*', color='black', s=250)

    handles.append(star)
    labels.append("Ideal")


    # Add row labels (MF and LightGCN) beside each row
    fig.text(0.09, 0.7, r"$\mathbf{MF}$", va='center', ha='right',rotation=0)
    fig.text(0.09, 0.28, r"$\mathbf{LGN}$", va='center', ha='right',rotation=0)


    fig.legend(handles, labels, loc="upper center", ncol=5, fontsize=24)
    plt.tight_layout(rect=[0.1, 0, 1, 0.92])
    plt.savefig("../../../outputs/final_eval/pareto_combined.pdf")
    plt.close()


if __name__ == "__main__":
    csv_path = "../../../outputs/final_eval/pareto_combined.csv"

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        print(f"Loaded cached results from {csv_path}")
    else:
        df = collect_metrics_for_scatter()
        df.to_csv(csv_path, index=False)
        print(f"Saved new results to {csv_path}")

    plot_scatter_metrics(df)
