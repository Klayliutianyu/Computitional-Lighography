import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# Generate synthetic data for the inverse problem
np.random.seed(42)
A = np.random.randn(50, 50)  # Model matrix A (100 observations, 50 features)
x_true = np.random.randn(50, 50)   # True solution vector
y = A.dot(x_true) + 0.1 * np.random.randn(50, 50)  # Observations with some noise

# Split data into training and testing sets (optional)
# Here, we assume all data is used for training as it's an inverse problem scenario
A_train = A
y_train = y

# L2 Regularization (Ridge Regression)
# alpha = 1.0  # Regularization strength
# ridge = Ridge(alpha=alpha)  # Create a Ridge regression model with L2 regularization
# ridge.fit(A_train, y_train, )  # Fit the model on the training data

# L2 Regularization (Ridge Regression)
iters = 1000
lr = 0.0001
x_init = np.random.randn(50, 50)

for i in range(iters):
    grad = -2*np.matmul(A.T, (y-np.matmul(A, x_init)))+2*x_init
    x_init = x_init-lr*grad
    loss = (x_init-x_true).sum()
    print(loss)

# Retrieve the solution vector (estimated x_true)
# x_ridge = ridge.coef_

# Evaluate the solution (optional)
# y_pred = A_train.dot(x_ridge)
# mse = mean_squared_error(y_train, y_pred)

print("Estimated solution (x_ridge):", x_init)
print("True solution (x_true):", x_true)
# print("Mean Squared Error:", mse)

import matplotlib.pyplot as plt

plt.figure()
plt.subplot(121)
plt.imshow(x_init, cmap='gray')
plt.title('x_init')
plt.subplot(122)
plt.imshow(x_true, cmap='gray')
plt.title('x_true')
plt.show()
