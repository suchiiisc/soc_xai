import xgboost as xgb
import shap
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.stats import chi2
from sklearn.covariance import EmpiricalCovariance
from sklearn.manifold import TSNE



file_path = './SOC/OSSL_ARD.csv'  # Update this to your file path
data = pd.read_csv(file_path)
X = data.drop(columns=["SOC"])
y = data["SOC"]


# Perform PCA
#pca = PCA(n_components = 2)
#X_pca = pca.fit_transform(X)

# Normalize the data
X_normalized = (X - X.mean()) / X.std()





X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


optimal_params = {
    'booster': 'dart',
    'lambda': 2.9346200249618713e-07,
    'alpha': 0.005579280483503804,
    'max_depth': 5,
    'eta': 0.17993106755258026,
    'gamma': 8.770851274533843e-06,
    'grow_policy': 'depthwise',
    'sample_type': 'uniform',
    'normalize_type': 'forest',
    'rate_drop': 3.96679972263458e-06,
    'skip_drop': 4.018363236629942e-05,
    'objective': 'reg:squarederror',
    'n_estimators': 582,
    'random_state': 42,
    "eval_metric": "mse",
    #'tree_method': 'gpu_hist',  # Use GPU if available
    #'predictor': 'gpu_predictor',
    'device': 'cuda'
}#RMSE: 3.836066530843889
#MAE: 1.9219335074290478

# Train XGBoost model
#model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=582,max_depth= 5, random_state=42, device="cuda")#792
#model = xgb.XGBRegressor(**optimal_params)#792
model = xgb.XGBRegressor(#booster='dart',
    reg_lambda=2.9346200249618713e-07,       # L2 regularization term on weights
    reg_alpha=0.005579280483503804,          # L1 regularization term on weights
    max_depth=5,                             # Maximum tree depth for base learners
    learning_rate=0.17993106755258026,       # Boosting learning rate (eta)
    gamma=8.770851274533843e-06,             # Minimum loss reduction required to make a further partition
    grow_policy='depthwise',                 # Tree growing policy
    sample_type='uniform',                   # Type of sampling algorithm
    normalize_type='forest',                 # Type of normalization algorithm
    rate_drop=3.96679972263458e-06,          # Dropout rate (only used with dart booster)
    skip_drop=4.018363236629942e-05,         # Probability of skipping the dropout procedure during a boosting iteration
    n_estimators=582,                        # Number of boosting rounds
    objective='reg:squarederror', 
    #eval_metric= "mse",           # Loss function for regression
    random_state=42 , device="cuda")#792
model.fit(X_train, y_train)

# Predict and evaluate
predictions = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
print(f"RMSE: {rmse}")
mae = mean_absolute_error(y_test, predictions)
print(f"MAE: {mae}")
#RMSE: 3.70393039622348
#MAE: 1.874640646182523

# Global explainer
# Initialize the SHAP explainer
explainer = shap.Explainer(model, X)


# Calculate SHAP values for all instances
shap_values = explainer(X)

# Summary plot
plt.figure()
shap.summary_plot(shap_values, X)
plt.savefig(("./SOC/"+"shap_summary_plot.png"), bbox_inches="tight", dpi=300)

plt.figure()
shap.summary_plot(shap_values, X, plot_type="bar")
plt.savefig(("./SOC/"+"shap_summary_barplot.png"), bbox_inches="tight", dpi=300)

# Dependence plot for a specific feature
# compute SHAP values
explainer1 = shap.TreeExplainer(model)
shap_values1 = explainer1.shap_values(X)
plt.figure()
shap.dependence_plot("502", shap_values1, X)
plt.savefig("./SOC/"+"shap_dependence_plot_Feature1.png", bbox_inches="tight", dpi=300)

k=1




