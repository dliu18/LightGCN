import os
import sys
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

def extract_final_scalar(event_file, tag):
    final_val = None
    try:
        for e in tf.compat.v1.train.summary_iterator(event_file):
            for v in e.summary.value:
                if v.tag == tag:
                    final_val = v.simple_value
    except Exception:
        pass
    return final_val

def compute_duration(event_file):
    """Returns total time span of the event file in seconds."""
    try:
        times = [e.wall_time for e in tf.compat.v1.train.summary_iterator(event_file)]
        if times:
            return round(times[-1] - times[0], 1)
    except Exception:
        pass
    return None

def get_tag_name(metric_name):
    tag_names = {
        "Recall@[20, 100]": "Recall__20__100_",
        "Precision@[20, 100]": "Precision__20__100_",
        "NDCG@[20, 100]": "NDCG__20__100_"
    }
    if metric_name in tag_names:
        return tag_names[metric_name]
    else:
        return metric_name

def collect_metrics(base_dir):
    perf_metrics = ['Recall@[20, 100]', 'Precision@[20, 100]', 'NDCG@[20, 100]']
    fairness_metrics = ['Popularity_Opportunity_Bias_20', 'Niche-Recall_20', 'Gini_20', 'Coverage_20']
    results = []

    alpha_path = os.path.join(base_dir, "alpha")
    for alpha in os.listdir(alpha_path):
        beta_path = os.path.join(alpha_path, alpha, "beta")
        if not os.path.isdir(beta_path): continue
        for beta in os.listdir(beta_path):
            test_path = os.path.join(beta_path, beta)
            if not os.path.exists(test_path): continue

            metric_values = {'alpha': alpha, 'beta': beta}
            duration_found = False
            for fname in os.listdir(test_path):
                if fname.startswith("events.out"):
                    fpath = os.path.join(test_path, fname)

                    if not duration_found:
                        dur = compute_duration(fpath)
                        if dur is not None:
                            metric_values['duration_sec'] = dur
                            duration_found = True

                    for metric in fairness_metrics:
                        val = extract_final_scalar(fpath, f"Test/{get_tag_name(metric)}")
                        if val is not None:
                            metric_values[metric] = val

            for metric in perf_metrics:
                metric_dir = os.path.join(test_path, "Test", metric, '20')
                if os.path.exists(metric_dir):
                    for fname in os.listdir(metric_dir):
                        if fname.startswith("events.out"):
                            fpath = os.path.join(metric_dir, fname)
                            val = extract_final_scalar(fpath, f"Test/{get_tag_name(metric)}")
                            if val is not None:
                                metric_values[metric] = val
            results.append(metric_values)

    return pd.DataFrame(results)

def plot_heatmaps(df, output_pdf, dataset_name):
    metrics = [
        ('Recall@[20, 100]', 'max'),
        ('Precision@[20, 100]', 'max'),
        ('NDCG@[20, 100]', 'max'),
        ('Popularity_Opportunity_Bias_20', 'min'),
        ('Niche-Recall_20', 'max'),
        ('Gini_20', 'min'),
        ('duration_sec', 'max')
    ]
    df['alpha'] = df['alpha'].astype(float)
    df['beta'] = df['beta'].astype(float)

    with PdfPages(output_pdf) as pdf:
        for metric, mode in metrics:
            if metric not in df.columns:
                continue
            pivot = df.pivot(index='alpha', columns='beta', values=metric).astype(float)

            # Sort beta descending, alpha ascending
            pivot = pivot.sort_index(ascending=False, axis=0)   # alpha (rows): ascending
            pivot = pivot.sort_index(ascending=False, axis=1)  # beta (columns): descending

            plt.figure(figsize=(5, 4))
            ax = sns.heatmap(
                pivot,
                annot=True,
                fmt=".4f",
                cmap="Oranges",
                cbar_kws={'label': metric}
            )

            opt_val = pivot.min().min() if mode == 'min' else pivot.max().max()
            for y in range(pivot.shape[0]):
                for x in range(pivot.shape[1]):
                    val = pivot.iloc[y, x]
                    if val == opt_val:
                        ax.add_patch(plt.Rectangle((x, y), 1, 1, fill=False, edgecolor='red', lw=3))

            title = f"{metric} Heatmap ({'min' if mode == 'min' else 'max'} highlighted)"
            if metric == "duration_sec":
                title = f"Trial Duration Heatmap (seconds)"

            plt.title(title)
            plt.xlabel("Beta")
            plt.ylabel("Alpha")
            plt.tight_layout()
            pdf.savefig()
            plt.close()
    print(f"Saved heatmaps to {output_pdf}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python summarize_metrics.py <dataset_name> <model_name>")
        sys.exit(1)

    dataset_name = sys.argv[1]
    model_name = sys.argv[2]
    base_path = f"../runs/hyperparam/{model_name}/{dataset_name}"
    output_pdf = f"../../outputs/hyperparam/www/metric_heatmaps_{model_name}_{dataset_name}.pdf"

    df = collect_metrics(base_path)
    print(df.head())
    plot_heatmaps(df, output_pdf, dataset_name)
