import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

# Dummy Data (Replace with your actual data)
X = pd.DataFrame(np.random.rand(100, 20), columns=[f'feature_{i}' for i in range(20)])
y = np.random.choice([0, 1], size=(100,))

# 1. Train Scaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save Scaler
with open("models/scaler_1.pkl", "wb") as f:
    pickle.dump(scaler, f)

# 2. Train PCA
pca = PCA(n_components=10)
X_pca = pca.fit_transform(X_scaled)

# Save PCA
with open("models/pca_1.pkl", "wb") as f:
    pickle.dump(pca, f)

# 3. Train Logistic Regression
logistic_model = LogisticRegression()
logistic_model.fit(X_pca, y)

# Save Logistic Regression Model
with open("models/tuned_logistic_regression_model_1.pkl", "wb") as f:
    pickle.dump(logistic_model, f)

print("All models saved successfully in 'models/' folder.")