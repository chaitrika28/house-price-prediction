import pandas as pd
import numpy as np
import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load dataset
data = fetch_california_housing(as_frame=True)
X = data.data
y = data.target  # Price in $100,000s

# 2. Explore
print(X.head())
print(X.describe())
print("Missing values:\n", X.isnull().sum())

# 3. No missing values here, but handle if using Kaggle dataset:
# X.fillna(X.median(), inplace=True)

# 4. Normalize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# 6. Train model
model = LinearRegression()
model.fit(X_train, y_train)

# 7. Evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2  = r2_score(y_test, y_pred)

print(f"MSE  : {mse:.4f}")
print(f"RMSE : {np.sqrt(mse):.4f}")
print(f"R²   : {r2:.4f}")

# 8. Save model and scaler
joblib.dump(model,  "model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("Model saved!")