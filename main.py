import numpy as np
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Generate synthetic data for the inverse problem
np.random.seed(42)
A = np.random.randn(100, 50)  # Model matrix A (100 observations, 50 features)
x_true = np.random.randn(50)   # True solution vector
y = A.dot(x_true) + 0.1 * np.random.randn(100)  # Observations with some noise

# Split data into training and testing sets
A_train, A_test, y_train, y_test = train_test_split(A, y, test_size=0.2, random_state=42)

# L2 Regularization (Ridge Regression)
ridge = Ridge(alpha=1.0)  # Regularization strength lambda
ridge.fit(A_train, y_train)
x_ridge = ridge.coef_

# L1 Regularization (Lasso Regression)
lasso = Lasso(alpha=0.1)  # Regularization strength lambda
lasso.fit(A_train, y_train)
x_lasso = lasso.coef_

# Evaluate the solutions
y_pred_ridge = A_test.dot(x_ridge)
y_pred_lasso = A_test.dot(x_lasso)

print("Ridge Regression (L2) MSE:", mean_squared_error(y_test, y_pred_ridge))
print("Lasso Regression (L1) MSE:", mean_squared_error(y_test, y_pred_lasso))

print("Ridge Solution (first 10 coefficients):", x_ridge[:10])
print("Lasso Solution (first 10 coefficients):", x_lasso[:10])
