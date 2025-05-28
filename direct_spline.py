!pip install --upgrade jax jaxlib pzflow matplotlib seaborn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.preprocessing import StandardScaler
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

# Standardize the data
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df_numerical), columns=numerical_cols)

# Get minima and maxima for each column
mins = jnp.array(df_scaled.min(axis=0))
maxs = jnp.array(df_scaled.max(axis=0))

# Get the number of dimensions
ndim = len(numerical_cols)

# Build the bijector
bijector = Chain(
    ShiftBounds(mins, maxs, B=4),  # Map data to [-4, 4] to avoid edge effects
    RollingSplineCoupling(ndim, nlayers=5, B=5),  # Operate on [-5, 5]
)

# Define the latent distribution
latent = Uniform(input_dim=ndim, B=5)

# Define the normalizing flow
flow = Flow(data_columns=numerical_cols, bijector=bijector, latent=latent)

# Train the flow
losses = flow.train(df_scaled, epochs=100, batch_size=256, verbose=True)

# Generate samples
num_samples = 1000
samples = flow.sample(num_samples, seed=0)
samples_original = pd.DataFrame(scaler.inverse_transform(samples), columns=numerical_cols)

# Save results to PDF
pdf_filename = "adult_dataset_flow_results_nlayers5_epochs100_batch256.pdf"
with PdfPages(pdf_filename) as pdf:
    # Plot training losses
    plt.figure(figsize=(8, 5))
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Training Loss Over Epochs (n_layers=5, epochs=100, batch_size=256)")
    plt.grid(True)
    pdf.savefig()
    plt.close()

    # Plot KDE for each column
    for col in numerical_cols:
        plt.figure(figsize=(10, 6))
        sns.kdeplot(data=df_numerical[col], label='Original', color='blue', linestyle='-')
        sns.kdeplot(data=samples_original[col], label='Generated', color='orange', linestyle='--')
        plt.legend()
        plt.title(f'KDE of {col} (n_layers=5, epochs=100, batch_size=256)')
        plt.xlabel(col)
        plt.ylabel('Density')
        plt.grid(True)
        pdf.savefig()
        plt.close()

    # Plot correlation matrices
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.heatmap(df_numerical.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
    plt.title('Original Data Correlation')
    plt.subplot(1, 2, 2)
    sns.heatmap(samples_original.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
    plt.title('Generated Data Correlation (n_layers=5, epochs=100, batch_size=256)')
    plt.tight_layout()
    pdf.savefig()
    plt.close()

print(f"Results saved to {pdf_filename}")
