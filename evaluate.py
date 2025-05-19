import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


def load_data(real_path, synthetic_path):
    """Load real and synthetic data."""
    real_data = pd.read_csv(real_path)
    synthetic_data = pd.read_csv(synthetic_path)
    return real_data, synthetic_data


def evaluate_statistical_similarity(real_data, synthetic_data, info):
    """Evaluate statistical similarity between real and synthetic data."""
    results = {}

    # Numerical features
    num_cols = info["num_col_idx"]
    for col in num_cols:
        real_col = real_data.iloc[:, col]
        syn_col = synthetic_data.iloc[:, col]

        # Basic statistics
        results[f"mean_diff_{col}"] = abs(real_col.mean() - syn_col.mean())
        results[f"std_diff_{col}"] = abs(real_col.std() - syn_col.std())

        # Wasserstein distance
        results[f"wasserstein_{col}"] = wasserstein_distance(real_col, syn_col)

    # Categorical features
    cat_cols = info["cat_col_idx"]
    for col in cat_cols:
        real_col = real_data.iloc[:, col]
        syn_col = synthetic_data.iloc[:, col]

        # Distribution similarity
        real_dist = real_col.value_counts(normalize=True)
        syn_dist = syn_col.value_counts(normalize=True)

        # Ensure same categories
        all_cats = set(real_dist.index) | set(syn_dist.index)
        real_dist = real_dist.reindex(all_cats, fill_value=0)
        syn_dist = syn_dist.reindex(all_cats, fill_value=0)

        # TV distance
        results[f"tv_distance_{col}"] = 0.5 * np.sum(np.abs(real_dist - syn_dist))

    return results


def evaluate_data_utility(real_data, synthetic_data, info):
    """Evaluate data utility for downstream tasks."""
    results = {}

    # Prepare data
    num_cols = info["num_col_idx"]
    cat_cols = info["cat_col_idx"]
    target_col = info["target_col_idx"][0]  # Assuming single target

    # Split features and target
    X_real = real_data.iloc[:, num_cols + cat_cols]
    y_real = real_data.iloc[:, target_col]
    X_syn = synthetic_data.iloc[:, num_cols + cat_cols]
    y_syn = synthetic_data.iloc[:, target_col]

    # Handle categorical features
    X_real = pd.get_dummies(X_real)
    X_syn = pd.get_dummies(X_syn)

    # Ensure same columns
    common_cols = list(set(X_real.columns) & set(X_syn.columns))
    X_real = X_real[common_cols]
    X_syn = X_syn[common_cols]

    # Handle target variable based on task type
    if info["task_type"] == "classification":
        # For classification, ensure target is properly encoded
        from sklearn.preprocessing import LabelEncoder

        le = LabelEncoder()
        y_real = le.fit_transform(y_real)
        y_syn = le.transform(y_syn)

        # Train on synthetic, test on real
        model = LogisticRegression(max_iter=1000)
        model.fit(X_syn, y_syn)
        y_pred = model.predict(X_real)
        results["accuracy"] = accuracy_score(y_real, y_pred)

        # Train on real, test on real (baseline)
        model_baseline = LogisticRegression(max_iter=1000)
        model_baseline.fit(X_real, y_real)
        y_pred_baseline = model_baseline.predict(X_real)
        results["baseline_accuracy"] = accuracy_score(y_real, y_pred_baseline)

        # Additional classification metrics
        from sklearn.metrics import precision_score, recall_score, f1_score

        results["precision"] = precision_score(y_real, y_pred, average="weighted")
        results["recall"] = recall_score(y_real, y_pred, average="weighted")
        results["f1"] = f1_score(y_real, y_pred, average="weighted")

    else:  # regression
        # For regression, ensure target is numeric
        y_real = pd.to_numeric(y_real, errors="coerce")
        y_syn = pd.to_numeric(y_syn, errors="coerce")

        # Remove any rows with NaN values
        mask = ~(np.isnan(y_real) | np.isnan(y_syn))
        X_real = X_real[mask]
        X_syn = X_syn[mask]
        y_real = y_real[mask]
        y_syn = y_syn[mask]

        # Train on synthetic, test on real
        model = LinearRegression()
        model.fit(X_syn, y_syn)
        y_pred = model.predict(X_real)
        results["mse"] = mean_squared_error(y_real, y_pred)
        results["rmse"] = np.sqrt(results["mse"])

        # Train on real, test on real (baseline)
        model_baseline = LinearRegression()
        model_baseline.fit(X_real, y_real)
        y_pred_baseline = model_baseline.predict(X_real)
        results["baseline_mse"] = mean_squared_error(y_real, y_pred_baseline)
        results["baseline_rmse"] = np.sqrt(results["baseline_mse"])

    return results


def plot_distributions(real_data, synthetic_data, info, save_path="distributions.png"):
    """Plot distributions of real vs synthetic data."""
    num_cols = info["num_col_idx"]
    cat_cols = info["cat_col_idx"]

    n_features = len(num_cols) + len(cat_cols)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten()

    for idx, col in enumerate(num_cols + cat_cols):
        ax = axes[idx]

        if col in num_cols:
            # Numerical feature
            sns.kdeplot(data=real_data.iloc[:, col], ax=ax, label="Real", color="blue")
            sns.kdeplot(
                data=synthetic_data.iloc[:, col], ax=ax, label="Synthetic", color="red"
            )
        else:
            # Categorical feature
            real_dist = real_data.iloc[:, col].value_counts(normalize=True)
            syn_dist = synthetic_data.iloc[:, col].value_counts(normalize=True)

            x = np.arange(len(real_dist))
            width = 0.35

            ax.bar(
                x - width / 2, real_dist, width, label="Real", color="blue", alpha=0.6
            )
            ax.bar(
                x + width / 2,
                syn_dist,
                width,
                label="Synthetic",
                color="red",
                alpha=0.6,
            )
            ax.set_xticks(x)
            ax.set_xticklabels(real_dist.index, rotation=45)

        ax.set_title(f"Feature {col}")
        ax.legend()

    # Remove empty subplots
    for idx in range(len(num_cols + cat_cols), len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataname", type=str, default="adult")
    parser.add_argument("--real_path", type=str, default="data/adult/train.csv")
    parser.add_argument("--synthetic_path", type=str, default="sample.csv")
    args = parser.parse_args()

    # Load data
    real_data, synthetic_data = load_data(args.real_path, args.synthetic_path)

    # Load info
    import json

    with open(f"data/{args.dataname}/info.json", "r") as f:
        info = json.load(f)

    # Evaluate
    print("Evaluating statistical similarity...")
    stat_results = evaluate_statistical_similarity(real_data, synthetic_data, info)

    print("\nEvaluating data utility...")
    utility_results = evaluate_data_utility(real_data, synthetic_data, info)

    # Print results
    print("\nStatistical Similarity Results:")
    for metric, value in stat_results.items():
        print(f"{metric}: {value:.4f}")

    print("\nData Utility Results:")
    for metric, value in utility_results.items():
        print(f"{metric}: {value:.4f}")

    # Plot distributions
    print("\nGenerating distribution plots...")
    plot_distributions(real_data, synthetic_data, info)
    print("Plots saved to distributions.png")


if __name__ == "__main__":
    main()
