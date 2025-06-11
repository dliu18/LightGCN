import os
import numpy as np
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

model = "mf"
# base_dirs = ["../../runs/final", "../../runs/final-1", "../../runs/final-2", "../../runs/final-3"]
base_dirs = ["../../runs/final"]

target_variants = ["vanilla", "ours", "only-items", "only-users"]

target_tags = [
    "Test/Recall__20__2000_",
    "Test/Precision__20__2000_",
    "Test/NDCG__20__2000_",
    "Test/Popularity_Opportunity_Bias_20",
]

pretty_names = {
    "Test/Recall__20__2000_": "Recall@20",
    "Test/Precision__20__2000_": "Precision@20",
    "Test/NDCG__20__2000_": "NDCG@20",
    "Test/Popularity_Opportunity_Bias_20": "POB@20",
}

def parse_event_file(event_file, target_tags):
    ea = EventAccumulator(event_file, size_guidance={'scalars': 0})
    ea.Reload()
    final_values = {}
    for tag in ea.Tags()['scalars']:
        events = ea.Scalars(tag)
        if events and tag in target_tags:
            final_values[tag] = events[-1].value
    return final_values

def collect_metrics_all_trials(base_dirs):
    all_data = {}
    for base_dir in base_dirs:
        base_dir = os.path.join(base_dir, model)
        for dataset_name in os.listdir(base_dir):
            dataset_path = os.path.join(base_dir, dataset_name)
            if not os.path.isdir(dataset_path):
                continue
            for variant in os.listdir(dataset_path):
                variant_path = os.path.join(dataset_path, variant)
                if not os.path.isdir(variant_path):
                    continue

                metrics = {}
                test_path = os.path.join(variant_path, "Test")
                # print(variant_path)
                # print(variant)
                if os.path.isdir(test_path):
                    for metric_folder in os.listdir(test_path):
                        metric_path = os.path.join(test_path, metric_folder, "20")
                        if not os.path.isdir(metric_path):
                            continue
                        for f in os.listdir(metric_path):
                            if f.startswith("events.out.tfevents"):
                                event_file = os.path.join(metric_path, f)
                                print(event_file)
                                parsed_vals = parse_event_file(event_file, target_tags)
                                metrics.update(parsed_vals)

                for f in os.listdir(variant_path):
                    if f.startswith("events.out.tfevents"):
                        event_file = os.path.join(variant_path, f)
                        print(event_file)
                        parsed_vals = parse_event_file(event_file, target_tags)
                        metrics.update(parsed_vals)
                        break

                key = (dataset_name, variant)
                if key not in all_data:
                    all_data[key] = {tag: [] for tag in target_tags}
                for tag in target_tags:
                    if tag in metrics:
                        all_data[key][tag].append(metrics[tag])

    records = []
    for (dataset, variant), tag_dict in all_data.items():
        row = {"dataset": dataset, "model_variant": variant}
        for tag in target_tags:
            vals = tag_dict[tag]
            row[tag + "_mean"] = np.mean(vals) if vals else None
            row[tag + "_std"] = np.std(vals) if vals else None
            print(dataset, variant, tag, vals)
        records.append(row)

    return pd.DataFrame(records)

if __name__ == "__main__":
    df = collect_metrics_all_trials(base_dirs)
    df.to_csv(f"../../../outputs/final_eval/aggregate_{model}.csv", index=False)

    latex_df = df[df["model_variant"].isin(target_variants)]

    final_rows = []
    for dataset in latex_df["dataset"].unique():
        subset = latex_df[latex_df["dataset"] == dataset]
        if "vanilla" not in subset["model_variant"].values:
            continue

        v_row = subset[subset["model_variant"] == "vanilla"].iloc[0]
        v_vals = {tag: (v_row[tag + "_mean"], v_row[tag + "_std"]) for tag in target_tags}

        def format_metrics(row):
            return [
                # f"{row[tag + '_mean']:.4f}$\\pm${row[tag + '_std']:.4f}"
                f"{row[tag + '_mean']:.4f}"
                for tag in target_tags
            ]

        def format_deltas(row):
            deltas = []
            for tag in target_tags:
                base = v_vals[tag][0]
                new = row[tag + "_mean"]
                delta = ((new - base) / base * 100) if base else 0
                deltas.append(f"\\textbf{{({delta:+.1f}\\%)}}")
            return deltas

        final_rows.append([dataset, "Vanilla"] + format_metrics(v_row))

        for variant in ["ours", "only-items", "only-users"]:
            if variant in subset["model_variant"].values:
                row = subset[subset["model_variant"] == variant].iloc[0]
                final_rows.append(["", model.upper() + " " + variant.replace("-", " ").title()] + format_metrics(row))
                final_rows.append(["", "$\\Delta$"] + format_deltas(row))

    final_latex_df = pd.DataFrame(
        final_rows,
        columns=["Dataset", "Variant"] + [pretty_names[t] for t in target_tags]
    )

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

    latex_table += "\\bottomrule\n\\end{tabular}%\n}\\caption{Performance (mean$\\pm$std) and relative deltas from vanilla across four trials.}\\label{tab:results}\n\\end{table}\n"

    with open(f"../../../outputs/final_eval/aggregate_{model}.txt", "w") as f:
        f.write(latex_table)
