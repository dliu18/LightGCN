import os
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

our_variant = "ours"
model = "lgn"
# our_variant = "our-non-zero-beta"

def parse_event_file(event_file, target_tags):
    ea = EventAccumulator(event_file, size_guidance={'scalars': 0})
    ea.Reload()
    final_values = {}
    max_step = 0
    times = []
    for tag in ea.Tags()['scalars']:
        events = ea.Scalars(tag)
        if events:
            if tag in target_tags:
                final_values[tag] = events[-1].value
            max_step = max(max_step, events[-1].step)
            times.append(events[-1].wall_time)
    return final_values, max_step, times

def collect_metrics(base_dir="final"):
    rows = []
    target_tags = [
        "Test/Recall__20__2000_",
        "Test/Precision__20__2000_",
        "Test/NDCG__20__2000_",
        "Test/Popularity_Opportunity_Bias_20",
    ]

    for dataset_name in os.listdir(base_dir):
        dataset_path = os.path.join(base_dir, dataset_name)
        if not os.path.isdir(dataset_path):
            continue
        for variant in os.listdir(dataset_path):
            variant_path = os.path.join(dataset_path, variant)
            if not os.path.isdir(variant_path):
                continue

            metrics = {}
            max_epochs = 0
            all_times = []

            # Parse metrics under Test/
            test_path = os.path.join(variant_path, "Test")
            if os.path.isdir(test_path):
                for metric_folder in os.listdir(test_path):
                    metric_path = os.path.join(test_path, metric_folder, "20")
                    if not os.path.isdir(metric_path):
                        continue
                    for f in os.listdir(metric_path):
                        if f.startswith("events.out.tfevents"):
                            event_file = os.path.join(metric_path, f)
                            parsed_vals, steps, times = parse_event_file(event_file, target_tags)
                            metrics.update(parsed_vals)
                            max_epochs = max(max_epochs, steps)
                            all_times.extend(times)

            # Parse top-level event file for Popularity Opportunity Bias
            top_event_file = None
            for f in os.listdir(variant_path):
                if f.startswith("events.out.tfevents"):
                    top_event_file = os.path.join(variant_path, f)
                    break
            if top_event_file:
                try:
                    parsed_vals, steps, times = parse_event_file(top_event_file, target_tags)
                    if "Test/Popularity_Opportunity_Bias_20" in parsed_vals:
                        metrics["Test/Popularity_Opportunity_Bias_20"] = parsed_vals["Test/Popularity_Opportunity_Bias_20"]
                except Exception as e:
                    print(f"Warning: could not parse top-level event file {top_event_file}: {e}")

            row = {
                "dataset": dataset_name,
                "model_variant": variant,
                "Recall__20__2000_": metrics.get("Test/Recall__20__2000_", None),
                "Precision__20__2000_": metrics.get("Test/Precision__20__2000_", None),
                "NDCG__20__2000_": metrics.get("Test/NDCG__20__2000_", None),
                "Popularity_Opportunity_Bias_20": metrics.get("Test/Popularity_Opportunity_Bias_20", None),
                "epochs_trained": max_epochs,
                "training_time": (max(all_times) - min(all_times)) if all_times else None,
            }
            rows.append(row)

    return pd.DataFrame(rows)

if __name__ == "__main__":

    df = collect_metrics(f"../../runs/final/{model}")

    # Save full dataframe to CSV
    df.to_csv(f"../../../outputs/final_eval/aggregate_{model}.csv", index=False)


    # For LaTeX table: only keep 'vanilla' and 'ours' variants
    latex_df = df[df["model_variant"].isin(["vanilla", our_variant])].drop(
        columns=["epochs_trained", "training_time"]
    )

    # Pretty rename columns
    pretty_column_names = {
        "Recall__20__2000_": "Recall@20",
        "Precision__20__2000_": "Precision@20",
        "NDCG__20__2000_": "NDCG@20",
        "Popularity_Opportunity_Bias_20": "POB@20",
    }
    latex_df = latex_df.rename(columns=pretty_column_names)

    # Create LaTeX table with Delta rows
    final_rows = []
    for dataset in latex_df["dataset"].unique():
        subset = latex_df[latex_df["dataset"] == dataset]
        if "vanilla" not in subset["model_variant"].values or our_variant not in subset["model_variant"].values:
            continue

        vanilla_row = subset[subset["model_variant"] == "vanilla"].iloc[0]
        ours_row = subset[subset["model_variant"] == our_variant].iloc[0]

        vanilla_vals = vanilla_row.drop(["dataset", "model_variant"]).astype(float)
        ours_vals = ours_row.drop(["dataset", "model_variant"]).astype(float)

        delta_vals = ((ours_vals - vanilla_vals) / vanilla_vals * 100).round(1)

        final_rows.append(
            [dataset, "Vanilla"] + [f"{val:.4f}" for val in vanilla_vals.values]
        )
        final_rows.append(
            ["", "Ours"] + [f"{val:.4f}" for val in ours_vals.values]
        )
        final_rows.append(
            ["", "$\\Delta$"] + [f"\\textbf{{({val:+.1f}\\%)}}" for val in delta_vals.values]
        )
    
    # Create DataFrame for LaTeX
    final_latex_df = pd.DataFrame(
        final_rows,
        columns=["Dataset", "Variant", "Recall@20", "Pre@20", "NDCG@20", "Bias"]
    )

    # Build LaTeX table
    latex_table = (
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\resizebox{\\columnwidth}{!}{%\n"
        "\\begin{tabular}{llcccc}\n"
        "\\toprule\n"
        "\\textbf{Dataset} & \\textbf{Variant} & \\textbf{Recall@20} & \\textbf{Pre@20} & \\textbf{NDCG@20} & \\textbf{Bias} \\\\\n"
        "\\midrule\n"
    )

    last_dataset = None
    for idx, row in final_latex_df.iterrows():
        if idx != 0 and row["Dataset"] != "" and last_dataset != row["Dataset"]:
            latex_table += "\\hline\n"
        line = " & ".join(row.values) + " \\\\\n"
        latex_table += line
        last_dataset = row["Dataset"]

    latex_table += "\\bottomrule\n\\end{tabular}%\n}\\caption{Performance comparison between vanilla and ours models across datasets.}\\label{tab:results}\n\\end{table}\n"

    # Write to file
    with open(f"../../../outputs/final_eval/aggregate_{model}.txt", "w") as f:
        f.write(latex_table)
