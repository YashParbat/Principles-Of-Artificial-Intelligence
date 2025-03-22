#Load the data
from sklearn.datasets import fetch_california_housing
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, SGDRegressor


housing = fetch_california_housing(as_frame=True)
df = housing.frame

# Scatter plot
plt.scatter(df['MedInc'], df['MedHouseVal'], alpha=0.3)
plt.xlabel('Median Income ($10,000s)')
plt.ylabel('Median House Value ($100,000s)')
plt.title('Median Income vs House Value')
plt.show()

# Summary statistics
print(df[['MedInc', 'MedHouseVal']].describe())

X = df[['MedInc']]
y = df['MedHouseVal']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize (optional)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Batch Gradient Descent (Linear Regression)
lin_reg = LinearRegression()
lin_reg.fit(X_train_scaled, y_train)

# Stochastic Gradient Descent
sgd_reg = SGDRegressor(max_iter=1000, eta0=0.01, random_state=42)
sgd_reg.fit(X_train_scaled, y_train)

# Predictions
y_pred_batch = lin_reg.predict(X_test_scaled)
y_pred_sgd = sgd_reg.predict(X_test_scaled)

# Prediction for MedInc = 8.0 ($80,000)
sample = scaler.transform([[8.0]])
predicted_value_batch = lin_reg.predict(sample)
predicted_value_sgd = sgd_reg.predict(sample)

print(f"Batch GD Prediction for MedInc=8.0: {predicted_value_batch[0]:.2f} ($100,000s)")
print(f"SGD Prediction for MedInc=8.0: {predicted_value_sgd[0]:.2f} ($100,000s)")

plt.scatter(X_test, y_test, alpha=0.3, label='Actual Data')
plt.plot(X_test, y_pred_batch, color='r', label='Batch GD')
plt.plot(X_test, y_pred_sgd, color='g', linestyle='--', label='SGD')
plt.xlabel('Median Income ($10,000s)')
plt.ylabel('Median House Value ($100,000s)')
plt.legend()
plt.title('Regression Lines on Test Data')
plt.show()
