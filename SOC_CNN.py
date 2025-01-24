import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import shap
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt
import random

# Check if GPU is available
"""
# Set random seeds for reproducibility
random.seed(42)
torch.manual_seed(42)
np.random.seed(42)

# If using GPU
torch.cuda.manual_seed_all(42)
"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class CNNRegressionModel(nn.Module):
    def __init__(self):
        super(CNNRegressionModel, self).__init__()
        # Using 1D convolution to treat each feature as a "pixel" of a 1-channel image
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 250, 128)  # Adjust size based on input features
        self.fc2 = nn.Linear(128, 1)  # Output is a single value for regression

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool1d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool1d(x, 2)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # No activation function here for regression
        return x
    
file_path = './SOC/OSSL_ARD.csv'  # Update this to your file path
data = pd.read_csv(file_path)
X = data.drop(columns=["SOC"])
y = data["SOC"]

# Normalize the features (optional but common in deep learning)
X = (X - X.mean()) / X.std()

# Convert X and y to numpy arrays and ensure correct alignment
X_values = X.values
y_values = y.values

# Ensure that X and y have the same number of samples
assert X_values.shape[0] == y_values.shape[0], "Number of samples in X and y must match"

# Reshape the data into the format (samples, channels, length) for Conv1d
# The 'length' here corresponds to the number of features (1000), and 'channels' = 1 because it's tabular data.
X_tensor = torch.tensor(X_values).float().view(-1, 1, 1000).to(device)  # Adjust to shape [samples, channels, features]
y_tensor = torch.tensor(y_values).float().view(-1, 1).to(device)

# Create DataLoader
dataset = TensorDataset(X_tensor, y_tensor)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Initialize the model, optimizer, and loss function
model = CNNRegressionModel().to(device)
#optimizer = optim.Adam(model.parameters(), lr=0.001)#optim.SGD(
optimizer = optim.SGD(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()  # Mean Squared Error loss for regression

# Training loop
epochs = 50
for epoch in range(epochs):
    model.train()
    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
    
    print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item()}')
# Predict and evaluate
model.eval()
all_preds = []
all_targets = []

with torch.no_grad():
    for data, target in test_loader:
        predictions = model(data)
        all_preds.append(predictions.cpu().numpy())
        all_targets.append(target.cpu().numpy())

# Combine predictions and targets into single arrays
all_preds = np.vstack(all_preds)
all_targets = np.vstack(all_targets)

# Compute evaluation metrics
rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
mae = mean_absolute_error(all_targets, all_preds)

print(f"RMSE: {rmse}")
print(f"MAE: {mae}")
#RMSE: 2.9973502159118652
#MAE: 1.6038691997528076
###shap
# Load a sample input
shap_sample_size = 500  # Choose a manageable subset size
background_data = X_tensor[:shap_sample_size]  # Use the first `shap_sample_size` samples as background
test_sample = X_tensor[train_size:train_size + 100]  # Take 10 samples from the test data to explain

# Create SHAP explainer
explainer = shap.GradientExplainer(model, background_data)

# Get SHAP values for the test samples
shap_values = explainer.shap_values(test_sample)


# Visualize the SHAP values
shap.initjs()
#test_sample_flat = test_sample.view(test_sample.size(0), -1).cpu().detach().numpy()
# Summarize SHAP values for a single test sample
# Reshape SHAP values and test samples to 2D
shap_values_reshaped = shap_values.squeeze(1) # Flatten spatial dimensions
test_sample_reshaped = test_sample.squeeze(1).cpu().detach().numpy()

# Feature names for visualization
feature_names = [f"Feature_{i}" for i in range(test_sample_reshaped.shape[1])]
plt.figure()
shap.summary_plot(shap_values_reshaped, test_sample_reshaped, feature_names=feature_names)
plt.savefig(("./SOC/"+"shap_summary_plot_cnn.png"), bbox_inches="tight", dpi=300)


