!pip install --upgrade jax jaxlib pzflow matplotlib seaborn scipy

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.stats import gaussian_kde
from pzflow import Flow
from pzflow.bijectors import Chain, ShiftBounds, RollingSplineCoupling
from pzflow.distributions import Uniform
import jax.numpy as jnp

# Load the Adult dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
column_names = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", 
                "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss", 
                "hours-per-week", "native-country", "income"]
df = pd.read_csv(url, header=None, names=column_names, na_values=" ?", skipinitialspace=True)

# Select numerical columns
numerical_cols = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
df_numerical = df[numerical_cols].dropna()
print("Data shape:", df_numerical.shape)  # Should be (32561, 6)

# Split data into train and test sets
train_df, test_df = train_test_split(df_numerical, test_size=0.2, random_state=42)
print("Train shape:", train_df.shape, "Test shape:", test_df.shape)

# Standardize the data
scaler = StandardScaler()
train_scaled = pd.DataFrame(scaler.fit_transform(train_df), columns=numerical_cols)
test_scaled = pd.DataFrame(scaler.transform(test_df), columns=numerical_cols)

# Get minima and maxima for each column (from training data)
mins = jnp.array(train_scaled.min(axis=0))
maxs = jnp.array(train_scaled.max(axis=0))

# Get the number of dimensions
ndim = len(numerical_cols)

# Build the bijector
bijector = Chain(
    ShiftBounds(mins, maxs, B=4),  # Map data to [-4, 4]
    RollingSplineCoupling(nlayers=10, B=5),  # Corrected: use nlayers=10
)

# Define the latent distribution
latent = Uniform(input_dim=ndim, B=5)

# Define the normalizing flow
flow = Flow(data_columns=numerical_cols, bijector=bijector, latent=latent)

# Train the flow
losses = flow.train(train_scaled, epochs=200, batch_size=256, verbose=True)

# Generate samples
num_samples = 1000
samples = flow.sample(num_samples, seed=0)
samples_original = pd.DataFrame(scaler.inverse_transform(samples), columns=numerical_cols)

# Compute log-likelihood on test set
test_log_likelihood = flow.log_prob(test_scaled)
print(f"Mean Test Log-Likelihood: {test_log_likelihood.mean():.4f}")

# Compute KL divergence for each feature
kl_divergences = {}
for col in numerical_cols:
    # Fit KDEs for original and generated data
    original_data = train_df[col].values
    generated_data = samples_original[col].values
    
    # Define common grid for KDE
    min_val, max_val = min(original_data.min(), generated_data.min()), max(original_data.max(), generated_data.max())
    grid = np.linspace(min_val, max_val, 1000)
    
    # Compute KDEs
    kde_original = gaussian_kde(original_data)
    kde_generated = gaussian_kde(generated_data)
    
    # Evaluate KDEs on grid
    pdf_original = kde_original(grid)
    pdf_generated = kde_generated(grid)
    
    # Compute KL divergence (avoiding zeros)
    pdf_original = np.clip(pdf_original, 1e-10, None)
    pdf_generated = np.clip(pdf_generated, 1e-10, None)
    kl_div = np.sum(pdf_original * np.log(pdf_original / pdf_generated)) * (grid[1] - grid[0])
    kl_divergences[col] = kl_div
    
print("KL Divergences:", kl_divergences)

# Save results to PDF
pdf_filename = "adult_dataset_flow_results_nlayers10_epochs200_batch256.pdf"
with PdfPages(pdf_filename) as pdf:
    # Plot training losses
    plt.figure(figsize=(8, 5))
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Training Loss (nlayers=10, epochs=200, batch_size=256)")
    plt.grid(True)
    pdf.savefig()
    plt.close()

    # Plot KDE for each column
    for col in numerical_cols:
        plt.figure(figsize=(10, 6))
        sns.kdeplot(data=train_df[col], label='Original', color='blue', linestyle='-')
        sns.kdeplot(data=samples_original[col], label='Generated', color='orange', linestyle='--')
        plt.legend()
        plt.title(f'KDE of {col} (nlayers=10, epochs=200, batch_size=256)\nKL Divergence: {kl_divergences[col]:.4f}')
        plt.xlabel(col)
        plt.ylabel('Density')
        plt.grid(True)
        pdf.savefig()
        plt.close()

    # Plot correlation matrices
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.heatmap(train_df.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
    plt.title('Original Data Correlation')
    plt.subplot(1, 2, 2)
    sns.heatmap(samples_original.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
    plt.title('Generated Data Correlation (nlayers=10, epochs=200, batch_size=256)')
    plt.tight_layout()
    pdf.savefig()
    plt.close()

print(f"Results saved to {pdf_filename}")
