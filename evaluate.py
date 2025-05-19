import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error
from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json
import os


def load_data(real_path, synthetic_path):
    """Load real and synthetic data."""
    real_data = pd.read_csv(real_path)
    synthetic_data = pd.read_csv(synthetic_path)
    return real_data, synthetic_data


def evaluate_statistical_similarity(real_data, synthetic_data):
    """Evaluate statistical similarity between real and synthetic data."""
    results = {"numerical_metrics": {}, "categorical_metrics": {}}

    print("\nStatistical Similarity Evaluation Details:")

    # Numerical features
    num_cols = real_data.select_dtypes(include=[np.number]).columns
    print(f"\nNumerical columns found: {list(num_cols)}")

    for col in num_cols:
        if col in synthetic_data.columns:
            print(f"\nProcessing numerical column: {col}")
            real_col = real_data[col]
            syn_col = synthetic_data[col]

            # Basic statistics
            wd = wasserstein_distance(real_col, syn_col)
            mse = mean_squared_error(real_col, syn_col)

            print(f"Wasserstein distance: {wd:.4f}")
            print(f"MSE: {mse:.4f}")
            print(f"Mean difference: {abs(real_col.mean() - syn_col.mean()):.4f}")
            print(f"Std difference: {abs(real_col.std() - syn_col.std()):.4f}")

            results["numerical_metrics"][col] = {
                "wasserstein": wd,
                "mse": mse,
                "mean_diff": abs(real_col.mean() - syn_col.mean()),
                "std_diff": abs(real_col.std() - syn_col.std()),
            }

    # Categorical features
    cat_cols = real_data.select_dtypes(include=["object", "category"]).columns
    print(f"\nCategorical columns found: {list(cat_cols)}")

    for col in cat_cols:
        if col in synthetic_data.columns:
            print(f"\nProcessing categorical column: {col}")
            # Convert to string type first
            real_col = real_data[col].astype(str)
            syn_col = synthetic_data[col].astype(str)

            # Get distributions
            orig_dist = real_col.value_counts(normalize=True)
            syn_dist = syn_col.value_counts(normalize=True)

            print(f"Number of unique categories in real data: {len(orig_dist)}")
            print(f"Number of unique categories in synthetic data: {len(syn_dist)}")

            # Ensure same categories
            all_cats = set(orig_dist.index) | set(syn_dist.index)
            orig_dist = orig_dist.reindex(all_cats, fill_value=0)
            syn_dist = syn_dist.reindex(all_cats, fill_value=0)

            # Calculate metrics
            wd = wasserstein_distance(orig_dist, syn_dist)
            tv_distance = 0.5 * np.sum(np.abs(orig_dist - syn_dist))

            print(f"Wasserstein distance: {wd:.4f}")
            print(f"TV distance: {tv_distance:.4f}")

            results["categorical_metrics"][col] = {
                "wasserstein": wd,
                "tv_distance": tv_distance,
            }

    return results


def evaluate_data_utility(real_data, synthetic_data, info):
    """Evaluate data utility for downstream tasks."""
    results = {}

    # Get column names from the data
    num_cols = [real_data.columns[i] for i in info["num_col_idx"]]
    cat_cols = [real_data.columns[i] for i in info["cat_col_idx"]]
    target_col = real_data.columns[info["target_col_idx"][0]]

    print(f"\nData Utility Evaluation Details:")
    print(f"Numerical columns: {num_cols}")
    print(f"Categorical columns: {cat_cols}")
    print(f"Target column: {target_col}")

    # Split features and target
    X_real = real_data[num_cols + cat_cols].copy()
    y_real = real_data[target_col].copy()
    X_syn = synthetic_data[num_cols + cat_cols].copy()
    y_syn = synthetic_data[target_col].copy()

    print(f"\nInitial data shapes:")
    print(f"X_real: {X_real.shape}, y_real: {y_real.shape}")
    print(f"X_syn: {X_syn.shape}, y_syn: {y_syn.shape}")

    # Handle categorical features
    print("\nProcessing categorical features...")
    for col in cat_cols:
        print(f"\nProcessing column: {col}")
        # Convert to string type first to avoid dtype issues
        X_real[col] = X_real[col].astype(str)
        X_syn[col] = X_syn[col].astype(str)

        # Get all unique categories from both datasets
        all_categories = pd.concat([X_real[col], X_syn[col]]).unique()
        print(f"Number of unique categories: {len(all_categories)}")

        # Convert to categorical with all possible categories
        X_real[col] = pd.Categorical(X_real[col], categories=all_categories)
        X_syn[col] = pd.Categorical(X_syn[col], categories=all_categories)

    # Now perform one-hot encoding
    print("\nPerforming one-hot encoding...")
    X_real = pd.get_dummies(X_real, columns=cat_cols)
    X_syn = pd.get_dummies(X_syn, columns=cat_cols)
    print(f"After one-hot encoding:")
    print(f"X_real columns: {len(X_real.columns)}")
    print(f"X_syn columns: {len(X_syn.columns)}")

    # Ensure same columns in both datasets
    all_columns = list(set(X_real.columns) | set(X_syn.columns))
    print(f"\nTotal unique columns after encoding: {len(all_columns)}")
    X_real = X_real.reindex(columns=all_columns, fill_value=0)
    X_syn = X_syn.reindex(columns=all_columns, fill_value=0)

    # Verify we have data
    if len(X_real) == 0 or len(X_syn) == 0:
        print(
            "Warning: Empty data after preprocessing. Skipping data utility evaluation."
        )
        return results

    print(f"\nFinal data shapes before model training:")
    print(f"X_real: {X_real.shape}, y_real: {y_real.shape}")
    print(f"X_syn: {X_syn.shape}, y_syn: {y_syn.shape}")

    # Handle target variable based on task type
    if info["task_type"] == "classification":
        print("\nPerforming classification evaluation...")
        from sklearn.preprocessing import LabelEncoder
        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
        )

        # Convert target to string first
        y_real = y_real.astype(str)
        y_syn = y_syn.astype(str)
        print(f"Unique target values in real data: {y_real.unique()}")
        print(f"Unique target values in synthetic data: {y_syn.unique()}")

        le = LabelEncoder()
        y_real = le.fit_transform(y_real)
        y_syn = le.transform(y_syn)

        # Train on synthetic, test on real
        from sklearn.linear_model import LogisticRegression

        print("\nTraining model on synthetic data...")
        model = LogisticRegression(max_iter=1000)
        model.fit(X_syn, y_syn)
        y_pred = model.predict(X_real)

        results["accuracy"] = accuracy_score(y_real, y_pred)
        results["precision"] = precision_score(y_real, y_pred, average="weighted")
        results["recall"] = recall_score(y_real, y_pred, average="weighted")
        results["f1"] = f1_score(y_real, y_pred, average="weighted")

        # Train on real, test on real (baseline)
        print("\nTraining baseline model on real data...")
        model_baseline = LogisticRegression(max_iter=1000)
        model_baseline.fit(X_real, y_real)
        y_pred_baseline = model_baseline.predict(X_real)
        results["baseline_accuracy"] = accuracy_score(y_real, y_pred_baseline)

    else:  # regression
        print("\nPerforming regression evaluation...")
        from sklearn.linear_model import LinearRegression

        # Ensure target is numeric
        y_real = pd.to_numeric(y_real, errors="coerce")
        y_syn = pd.to_numeric(y_syn, errors="coerce")
        print(f"Target value ranges:")
        print(f"Real data: [{y_real.min()}, {y_real.max()}]")
        print(f"Synthetic data: [{y_syn.min()}, {y_syn.max()}]")

        # Remove any rows with NaN values
        mask = ~(np.isnan(y_real) | np.isnan(y_syn))
        X_real = X_real[mask]
        X_syn = X_syn[mask]
        y_real = y_real[mask]
        y_syn = y_syn[mask]
        print(f"\nRows after removing NaN values: {len(X_real)}")

        # Verify we still have data after cleaning
        if len(X_real) == 0 or len(X_syn) == 0:
            print(
                "Warning: No valid samples after cleaning. Skipping regression evaluation."
            )
            return results

        # Train on synthetic, test on real
        print("\nTraining model on synthetic data...")
        model = LinearRegression()
        model.fit(X_syn, y_syn)
        y_pred = model.predict(X_real)
        results["mse"] = mean_squared_error(y_real, y_pred)
        results["rmse"] = np.sqrt(results["mse"])

        # Train on real, test on real (baseline)
        print("\nTraining baseline model on real data...")
        model_baseline = LinearRegression()
        model_baseline.fit(X_real, y_real)
        y_pred_baseline = model_baseline.predict(X_real)
        results["baseline_mse"] = mean_squared_error(y_real, y_pred_baseline)
        results["baseline_rmse"] = np.sqrt(results["baseline_mse"])

    return results


def plot_distributions(real_data, synthetic_data, results, save_dir="evaluation_plots"):
    """Plot distributions of real vs synthetic data."""
    os.makedirs(save_dir, exist_ok=True)

    # Plot numerical distributions
    num_cols = list(results["numerical_metrics"].keys())
    n_cols = min(4, len(num_cols))
    n_rows = (len(num_cols) + n_cols - 1) // n_cols

    plt.figure(figsize=(15, 4 * n_rows))
    for i, col in enumerate(num_cols):
        plt.subplot(n_rows, n_cols, i + 1)
        plt.hist(real_data[col], alpha=0.5, label="Real", bins=30)
        plt.hist(synthetic_data[col], alpha=0.5, label="Synthetic", bins=30)
        plt.title(f"{col}\nWD: {results['numerical_metrics'][col]['wasserstein']:.3f}")
        plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "numerical_distributions.png"))
    plt.close()

    # Plot categorical distributions
    cat_cols = list(results["categorical_metrics"].keys())
    n_cols = min(4, len(cat_cols))
    n_rows = (len(cat_cols) + n_cols - 1) // n_cols

    plt.figure(figsize=(15, 4 * n_rows))
    for i, col in enumerate(cat_cols):
        plt.subplot(n_rows, n_cols, i + 1)
        orig_dist = real_data[col].value_counts(normalize=True)
        syn_dist = synthetic_data[col].value_counts(normalize=True)
        plt.bar(range(len(orig_dist)), orig_dist.values, alpha=0.5, label="Real")
        plt.bar(range(len(syn_dist)), syn_dist.values, alpha=0.5, label="Synthetic")
        plt.title(
            f"{col}\nWD: {results['categorical_metrics'][col]['wasserstein']:.3f}"
        )
        plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "categorical_distributions.png"))
    plt.close()


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataname", type=str, default="adult")
    parser.add_argument("--real_path", type=str, default="data/adult/train.csv")
    parser.add_argument("--synthetic_path", type=str, default="sample.csv")
    args = parser.parse_args()

    # Load data
    print("Loading data...")
    real_data = pd.read_csv(args.real_path)
    synthetic_data = pd.read_csv(args.synthetic_path)

    # Print data shapes
    print(f"Real data shape: {real_data.shape}")
    print(f"Synthetic data shape: {synthetic_data.shape}")

    # Load info
    with open(f"data/{args.dataname}/info.json", "r") as f:
        info = json.load(f)

    # Evaluate statistical similarity
    print("\nEvaluating statistical similarity...")
    stat_results = evaluate_statistical_similarity(real_data, synthetic_data)

    # Evaluate data utility
    print("\nEvaluating data utility...")
    utility_results = evaluate_data_utility(real_data, synthetic_data, info)

    # Print results
    print("\nNumerical Features Quality:")
    for col, metrics in stat_results["numerical_metrics"].items():
        print(f"\n{col}:")
        print(f"  Wasserstein Distance: {metrics['wasserstein']:.4f}")
        print(f"  MSE: {metrics['mse']:.4f}")
        print(f"  Mean Difference: {metrics['mean_diff']:.4f}")
        print(f"  Std Difference: {metrics['std_diff']:.4f}")

    print("\nCategorical Features Quality:")
    for col, metrics in stat_results["categorical_metrics"].items():
        print(f"\n{col}:")
        print(f"  Wasserstein Distance: {metrics['wasserstein']:.4f}")
        print(f"  TV Distance: {metrics['tv_distance']:.4f}")

    if utility_results:
        print("\nData Utility Results:")
        for metric, value in utility_results.items():
            print(f"{metric}: {value:.4f}")

    # Plot distributions
    print("\nGenerating distribution plots...")
    plot_distributions(real_data, synthetic_data, stat_results)
    print("Plots saved to evaluation_plots/")

    # Save results
    results = {"statistical_similarity": stat_results, "data_utility": utility_results}
    with open("evaluation_results.json", "w") as f:
        json.dump(results, f, indent=4)
    print("\nResults saved to evaluation_results.json")


if __name__ == "__main__":
    main()
