# Step 1: Install dependencies
!pip install torch pandas matplotlib seaborn scikit-learn

# Step 2: Import libraries
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.stats import wasserstein_distance

# Step 3: Load and preprocess data
def load_adult_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data "
    columns = [
        "age", "workclass", "fnlwgt", "education", "education_num", "marital_status",
        "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss",
        "hours_per_week", "native_country", "income"
    ]
    data = pd.read_csv(url, header=None, names=columns, na_values=" ?", skipinitialspace=True)
    
    # Select numeric columns
    numeric_cols = ["age", "fnlwgt", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
    data = data[numeric_cols].dropna()
    
    # Handle skewness in capital_gain
    data["capital_gain"] = np.log1p(data["capital_gain"])
    
    # Normalize data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    return torch.tensor(data_scaled, dtype=torch.float32)

# Step 4: Define Flow Components
class AffineCoupling(nn.Module):
    def __init__(self, dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim//2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim)
        )
        self.net[-1].weight.data.zero_()
        self.net[-1].bias.data.zero_()

    def forward(self, x):
        x1, x2 = torch.chunk(x, 2, dim=1)
        h = self.net(x1)
        shift, scale = torch.chunk(h, 2, dim=1)
        scale = torch.sigmoid(scale + 2)
        y2 = x2 * scale + shift
        log_det = torch.sum(torch.log(scale).flatten(start_dim=1), dim=1)
        return torch.cat([x1, y2], dim=1), log_det

    def inverse(self, y):
        y1, y2 = torch.chunk(y, 2, dim=1)
        h = self.net(y1)
        shift, scale = torch.chunk(h, 2, dim=1)
        scale = torch.sigmoid(scale + 2)
        x2 = (y2 - shift) / scale
        return torch.cat([y1, x2], dim=1)

class LearnedPermutation(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(dim, dim))

    def forward(self, x):
        W = self.weight
        sign, logabsdet = torch.slogdet(W)
        log_det = logabsdet * x.shape[0]
        return x @ W.T, log_det

    def inverse(self, y):
        return y @ torch.inverse(self.weight).T

class NormalizingFlow(nn.Module):
    def __init__(self, base_dist, transforms):
        super().__init__()
        self.base_dist = base_dist
        self.transforms = nn.ModuleList(transforms)
        
    def forward(self, x):
        log_prob = self.base_dist.log_prob(x)
        for transform in self.transforms:
            x, ldj = transform(x)
            log_prob += ldj
        return x, log_prob
        
    def sample(self, num_samples):
        x = self.base_dist.sample((num_samples,))
        for transform in reversed(self.transforms):
            x = transform.inverse(x)
        return x

# Step 5: Train Flow Model
def train_flow(data):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data.to(device)
    train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)
    
    # Define base distribution (multivariate normal)
    base_dist = torch.distributions.Independent(
        torch.distributions.Normal(torch.zeros(6, device=device), torch.ones(6, device=device)),
        reinterpreted_batch_ndims=1
    )
    
    # Define flow layers
    transforms = []
    for _ in range(2):  # Use 2 coupling layers for small dataset
        transforms.append(AffineCoupling(6, hidden_dim=128))
        transforms.append(LearnedPermutation(6))
    
    flow_model = NormalizingFlow(base_dist, transforms).to(device)
    optimizer = optim.AdamW(flow_model.parameters(), lr=3e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, verbose=True)
    
    # Training loop
    epochs = 500
    batch_size = 128
    best_loss = float('inf')
    patience = 20
    wait = 0
    losses = []

    for epoch in range(epochs):
        idx = torch.randperm(len(train_data))[:batch_size]
        batch = train_data[idx]
        
        optimizer.zero_grad()
        _, log_prob = flow_model(batch)
        loss = -log_prob.mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(flow_model.parameters(), max_norm=1.0)
        optimizer.step()
        
        losses.append(loss.item())
        scheduler.step(loss.item())
        
        # Early stopping
        if loss.item() < best_loss:
            best_loss = loss.item()
            wait = 0
            torch.save(flow_model.state_dict(), "best_flow.pth")
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping triggered.")
                break
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
    
    # Generate samples
    with torch.no_grad():
        samples = flow_model.sample(1000).cpu().numpy()
    
    # Evaluate
    plot_comparison(data.cpu().numpy(), samples)
    evaluate_distributions(data.cpu().numpy(), samples)
    return samples

def plot_comparison(real_data, fake_data):
    df_real = pd.DataFrame(real_data, columns=["age", "fnlwgt", "education_num", "capital_gain", "capital_loss", "hours_per_week"])
    df_fake = pd.DataFrame(fake_data, columns=["age", "fnlwgt", "education_num", "capital_gain", "capital_loss", "hours_per_week"])
    
    for col in df_real.columns:
        plt.figure(figsize=(10, 4))
        sns.histplot(df_real[col], bins=30, kde=True, label="Real", stat="density", alpha=0.5)
        sns.histplot(df_fake[col], bins=30, kde=True, label="Generated", stat="density", alpha=0.5)
        plt.title(f"{col} Distribution Comparison")
        plt.legend()
        plt.show()

def evaluate_distributions(real_data, fake_data):
    for i, col in enumerate(["age", "fnlwgt", "education_num", "capital_gain", "capital_loss", "hours_per_week"]):
        w_dist = wasserstein_distance(real_data[:, i], fake_data[:, i])
        print(f"Feature '{col}', Wasserstein distance: {w_dist:.4f}")

# Run pipeline
data = load_adult_data()
samples = train_flow(data)
