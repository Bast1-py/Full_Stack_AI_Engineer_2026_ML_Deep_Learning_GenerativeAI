# task 1: implement the mathematical formula for linear regression
import numpy as np


# generate
np.random.seed(42)
X = 2 * np.random.randn(26750, 1)
y = 4 + 3 * X + np.random.randn(26750,1)
# add bias term to features matrix
X_b = np.c_[np.ones((26750, 1)), X]
# initialize parameter
theta = np.random.randn(2, 1)
learning_rate = 0.1
iterations = 1000


def predict(X, theta):
    return np.dot(X, theta)


# task 2: use gradient descent to optimize the model parameters
def gradient_descent(X, y, theta, learning_rate, iteration):
    m = len(y)
    for _ in range(iterations):
        gradients = (1/m) * np.dot(X.T, (np.dot(X, theta) - y))
        theta -= learning_rate * gradients
    return theta


# task 3: calculate evaluation metrics
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1 - (ss_res / ss_tot)

# perform gradient descent
theta_optimized = gradient_descent(X_b, y, theta, learning_rate, iterations)

# prediction and evaluations
y_pred = predict(X_b, theta_optimized)
mse = mean_squared_error(y, y_pred)
r2 = r_squared(y, y_pred)


print('Optimized Parameters (Theta): ', theta_optimized)
print('MSE: ', mse)
print('R2: ', r2)