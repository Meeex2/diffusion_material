# Normalizing Flows on Adult Dataset - Google Colab Script
# This script trains normalizing flows on numeric columns of the Adult dataset
# and generates synthetic samples for distribution comparison

# # Install required packages
# !pip install torch torchvision torchaudio
# !pip install pandas numpy matplotlib seaborn scikit-learn
# !pip install normflows  # For normalizing flows implementation

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import normflows as nf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

print("Libraries imported successfully!")

# =====================================
# 1. DATA LOADING AND PREPROCESSING
# =====================================

def load_and_preprocess_adult_data():
    """Load Adult dataset and preprocess numeric columns"""
    
    # Load Adult dataset from UCI repository
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    
    # Column names for Adult dataset
    columns = [
        'age', 'workclass', 'fnlwgt', 'education', 'education-num',
        'marital-status', 'occupation', 'relationship', 'race', 'sex',
        'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'
    ]
    
    # Load data
    df = pd.read_csv(url, names=columns, na_values=' ?', skipinitialspace=True)
    
    # Remove rows with missing values
    df = df.dropna()
    
    # Select only numeric columns
    numeric_columns = ['age', 'fnlwgt', 'education-num', 'capital-gain', 
                      'capital-loss', 'hours-per-week']
    
    numeric_data = df[numeric_columns].copy()
    
    print(f"Dataset shape: {numeric_data.shape}")
    print(f"Numeric columns: {numeric_columns}")
    print("\nDataset statistics:")
    print(numeric_data.describe())
    
    return numeric_data, numeric_columns

# =====================================
# 2. NORMALIZING FLOW MODEL
# =====================================

class RealNVPLayer(nn.Module):
    """Real NVP coupling layer"""
    
    def __init__(self, input_dim, hidden_dim=128, mask=None):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Create mask
        if mask is None:
            self.register_buffer('mask', torch.arange(input_dim) % 2)
        else:
            self.register_buffer('mask', mask)
        
        # Scale and translation networks
        self.scale_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Tanh()  # Bound the scale to prevent explosion
        )
        
        self.translate_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def forward(self, x, reverse=False):
        masked_x = x * self.mask
        
        if not reverse:
            # Forward pass
            scale = self.scale_net(masked_x) * (1 - self.mask)
            translate = self.translate_net(masked_x) * (1 - self.mask)
            
            # Clamp scale to prevent numerical instability
            scale = torch.clamp(scale, -3, 3)  # exp(3) ≈ 20, exp(-3) ≈ 0.05
            
            y = x * torch.exp(scale) + translate
            log_det = scale.sum(dim=1)
            
            return y, log_det
        else:
            # Reverse pass
            scale = self.scale_net(masked_x) * (1 - self.mask)
            translate = self.translate_net(masked_x) * (1 - self.mask)
            
            # Clamp scale
            scale = torch.clamp(scale, -3, 3)
            
            y = (x - translate) * torch.exp(-scale)
            log_det = -scale.sum(dim=1)
            
            return y, log_det

class SimpleNormalizingFlow(nn.Module):
    """Improved Normalizing Flow with numerical stability"""
    
    def __init__(self, input_dim, hidden_dim=128, num_layers=6):
        super().__init__()
        self.input_dim = input_dim
        self.num_layers = num_layers
        
        # Create alternating masks
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            mask = torch.arange(input_dim) % 2
            if i % 2 == 1:
                mask = 1 - mask  # Flip mask every other layer
            
            self.layers.append(RealNVPLayer(input_dim, hidden_dim, mask))
    
    def forward(self, x, reverse=False):
        if not reverse:
            log_det_total = 0
            for layer in self.layers:
                x, log_det = layer(x, reverse=False)
                log_det_total += log_det
            return x, log_det_total
        else:
            log_det_total = 0
            for layer in reversed(self.layers):
                x, log_det = layer(x, reverse=True)
                log_det_total += log_det
            return x, log_det_total
    
    def sample(self, num_samples):
        # Sample from standard normal
        z = torch.randn(num_samples, self.input_dim)
        
        # Transform through flow
        with torch.no_grad():
            x, _ = self.forward(z, reverse=True)
            
            # Check for numerical issues
            if torch.isnan(x).any() or torch.isinf(x).any():
                print("Warning: NaN or Inf detected in samples, clipping values")
                x = torch.clamp(x, -10, 10)  # Clip extreme values
        
        return x
    
    def log_prob(self, x):
        # Check input for numerical issues
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("Warning: NaN or Inf in input data")
            x = torch.clamp(x, -10, 10)
        
        z, log_det = self.forward(x, reverse=False)
        
        # Standard normal log probability
        log_prob_z = -0.5 * (z**2 + np.log(2 * np.pi)).sum(dim=1)
        
        # Total log probability
        log_prob_x = log_prob_z + log_det
        
        return log_prob_x

# =====================================
# 3. TRAINING FUNCTIONS
# =====================================

def train_normalizing_flow(data, input_dim, epochs=2000, batch_size=512, lr=5e-4):
    """Train normalizing flow model with improved stability"""
    
    # Convert to tensor
    data_tensor = torch.FloatTensor(data)
    
    # Create model
    model = SimpleNormalizingFlow(input_dim, hidden_dim=128, num_layers=6)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.8)
    
    # Training loop
    losses = []
    model.train()
    
    print("Starting training...")
    for epoch in range(epochs):
        # Shuffle data
        perm = torch.randperm(len(data_tensor))
        total_loss = 0
        num_batches = 0
        
        for i in range(0, len(data_tensor), batch_size):
            # Get batch
            batch_idx = perm[i:i+batch_size]
            batch = data_tensor[batch_idx]
            
            # Forward pass
            log_prob = model.log_prob(batch)
            
            # Check for numerical issues in loss
            if torch.isnan(log_prob).any() or torch.isinf(log_prob).any():
                print(f"Warning: NaN/Inf in log_prob at epoch {epoch}")
                continue
            
            loss = -log_prob.mean()  # Negative log-likelihood
            
            # Gradient clipping for stability
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        if num_batches > 0:
            avg_loss = total_loss / num_batches
            losses.append(avg_loss)
        
        scheduler.step()
        
        if (epoch + 1) % 200 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")
    
    print("Training completed!")
    
    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Negative Log-Likelihood')
    plt.grid(True)
    plt.show()
    
    return model, losses

# =====================================
# 4. EVALUATION AND VISUALIZATION
# =====================================

def compare_distributions(real_data, synthetic_data, columns, scaler=None):
    """Compare real and synthetic data distributions"""
    
    # If scaler is provided, inverse transform the data
    if scaler is not None:
        real_data_plot = scaler.inverse_transform(real_data)
        synthetic_data_plot = scaler.inverse_transform(synthetic_data)
    else:
        real_data_plot = real_data
        synthetic_data_plot = synthetic_data
    
    n_cols = len(columns)
    n_rows = (n_cols + 2) // 3
    
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 5*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, col in enumerate(columns):
        row = i // 3
        col_idx = i % 3
        
        ax = axes[row, col_idx]
        
        # Plot histograms
        ax.hist(real_data_plot[:, i], bins=50, alpha=0.7, label='Real', 
                density=True, color='blue')
        ax.hist(synthetic_data_plot[:, i], bins=50, alpha=0.7, label='Synthetic', 
                density=True, color='red')
        
        ax.set_title(f'{col}')
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Remove empty subplots
    for i in range(len(columns), n_rows * 3):
        row = i // 3
        col_idx = i % 3
        fig.delaxes(axes[row, col_idx])
    
    plt.tight_layout()
    plt.show()

def plot_correlation_comparison(real_data, synthetic_data, columns):
    """Compare correlation matrices"""
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Real data correlation
    real_corr = np.corrcoef(real_data.T)
    sns.heatmap(real_corr, annot=True, cmap='coolwarm', center=0,
                xticklabels=columns, yticklabels=columns, ax=axes[0])
    axes[0].set_title('Real Data Correlation')
    
    # Synthetic data correlation
    synthetic_corr = np.corrcoef(synthetic_data.T)
    sns.heatmap(synthetic_corr, annot=True, cmap='coolwarm', center=0,
                xticklabels=columns, yticklabels=columns, ax=axes[1])
    axes[1].set_title('Synthetic Data Correlation')
    
    plt.tight_layout()
    plt.show()
    
    # Compute correlation difference
    corr_diff = np.abs(real_corr - synthetic_corr)
    print(f"Mean absolute correlation difference: {corr_diff.mean():.4f}")
    print(f"Max absolute correlation difference: {corr_diff.max():.4f}")

def statistical_comparison(real_data, synthetic_data, columns):
    """Perform statistical tests comparing real and synthetic data"""
    
    print("Statistical Comparison (Kolmogorov-Smirnov Test):")
    print("=" * 60)
    
    for i, col in enumerate(columns):
        # Kolmogorov-Smirnov test
        ks_stat, p_value = stats.ks_2samp(real_data[:, i], synthetic_data[:, i])
        
        print(f"{col:15} - KS Statistic: {ks_stat:.4f}, p-value: {p_value:.4f}")
        if p_value > 0.05:
            print(f"               → No significant difference (p > 0.05)")
        else:
            print(f"               → Significant difference (p ≤ 0.05)")
        print()

# =====================================
# 5. MAIN EXECUTION
# =====================================

def main():
    """Main execution function"""
    
    print("🚀 Starting Normalizing Flows Training on Adult Dataset")
    print("=" * 60)
    
    # Load and preprocess data
    print("\n📊 Loading and preprocessing data...")
    data, columns = load_and_preprocess_adult_data()
    
    # Split data
    train_data, test_data = train_test_split(data.values, test_size=0.2, random_state=42)
    
    # Standardize data
    scaler = StandardScaler()
    train_data_scaled = scaler.fit_transform(train_data)
    test_data_scaled = scaler.transform(test_data)
    
    print(f"Training data shape: {train_data_scaled.shape}")
    print(f"Test data shape: {test_data_scaled.shape}")
    
    # Train normalizing flow
    print("\n🤖 Training Normalizing Flow...")
    model, losses = train_normalizing_flow(
        train_data_scaled, 
        input_dim=len(columns),
        epochs=2000,
        batch_size=2 * 4096,
        lr=5e-4
    )
    
    # Generate synthetic samples
    print("\n🎲 Generating synthetic samples...")
    model.eval()
    with torch.no_grad():
        synthetic_samples = model.sample(len(test_data_scaled)).numpy()
        
        # Check for numerical issues and clean data
        if np.isnan(synthetic_samples).any() or np.isinf(synthetic_samples).any():
            print("Warning: Found NaN/Inf in synthetic samples, cleaning...")
            # Replace NaN/Inf with reasonable values
            synthetic_samples = np.nan_to_num(synthetic_samples, 
                                            nan=0.0, 
                                            posinf=3.0, 
                                            neginf=-3.0)
        
        # Additional clipping for extreme values
        synthetic_samples = np.clip(synthetic_samples, -5, 5)
    
    print(f"Generated {len(synthetic_samples)} synthetic samples")
    
    # Compare distributions
    print("\n📈 Comparing distributions...")
    compare_distributions(test_data_scaled, synthetic_samples, columns, scaler)
    
    # Compare correlations
    print("\n📊 Comparing correlations...")
    plot_correlation_comparison(test_data_scaled, synthetic_samples, columns)
    
    # Statistical comparison
    print("\n📉 Statistical comparison...")
    statistical_comparison(test_data_scaled, synthetic_samples, columns)
    
    # Summary statistics comparison
    print("\n📋 Summary Statistics Comparison:")
    print("=" * 60)
    
    # Inverse transform for interpretable comparison
    test_data_orig = scaler.inverse_transform(test_data_scaled)
    synthetic_data_orig = scaler.inverse_transform(synthetic_samples)
    
    for i, col in enumerate(columns):
        real_mean = test_data_orig[:, i].mean()
        synthetic_mean = synthetic_data_orig[:, i].mean()
        real_std = test_data_orig[:, i].std()
        synthetic_std = synthetic_data_orig[:, i].std()
        
        print(f"{col:15}")
        print(f"  Real:      Mean={real_mean:8.2f}, Std={real_std:8.2f}")
        print(f"  Synthetic: Mean={synthetic_mean:8.2f}, Std={synthetic_std:8.2f}")
        print(f"  Difference: Mean={abs(real_mean-synthetic_mean):7.2f}, Std={abs(real_std-synthetic_std):7.2f}")
        print()
    
    print("✅ Analysis completed!")
    
    return model, scaler, data, columns

# Run the main function
if __name__ == "__main__":
    model, scaler, original_data, feature_columns = main()

# =====================================
# 6. ADDITIONAL UTILITY FUNCTIONS
# =====================================

def generate_new_samples(model, scaler, num_samples=1000):
    """Generate new synthetic samples with numerical stability checks"""
    model.eval()
    with torch.no_grad():
        samples_scaled = model.sample(num_samples).numpy()
        
        # Clean samples
        if np.isnan(samples_scaled).any() or np.isinf(samples_scaled).any():
            print("Warning: Found NaN/Inf in generated samples, cleaning...")
            samples_scaled = np.nan_to_num(samples_scaled, nan=0.0, posinf=3.0, neginf=-3.0)
        
        samples_scaled = np.clip(samples_scaled, -5, 5)
        samples_original = scaler.inverse_transform(samples_scaled)
    
    return samples_original

def save_synthetic_data(samples, columns, filename='synthetic_adult_data.csv'):
    """Save synthetic data to CSV"""
    df = pd.DataFrame(samples, columns=columns)
    df.to_csv(filename, index=False)
    print(f"Synthetic data saved to {filename}")

# Example usage:
# new_samples = generate_new_samples(model, scaler, 1000)
# save_synthetic_data(new_samples, feature_columns)
