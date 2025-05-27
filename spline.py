import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from pzflow import Flow
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

# Standardize the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(df_numerical)
data_scaled = jnp.array(data_scaled)  # Convert to JAX array for pzflow

# Define the normalizing flow
flow = Flow(data_columns=numerical_cols, bijector={"name": "NeuralSpline", "n_layers": 5})

# Train the flow
flow.train(data_scaled, epochs=100, batch_size=256, verbose=True)

# Generate samples
num_samples = 1000
samples = flow.sample(num_samples, seed=0)
samples_original = scaler.inverse_transform(samples)

# Plot histograms for comparison
for i, col in enumerate(numerical_cols):
    plt.figure(figsize=(10, 6))
    plt.hist(df_numerical[col], bins=30, alpha=0.5, label='Original', density=True, color='blue')
    plt.hist(samples_original[:, i], bins=30, alpha=0.5, label='Generated', density=True, color='orange')
    plt.legend()
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Density')
    plt.show()
