import numpy as np

np.random.seed(42)
X = 2 * np.random.random((13700,1))
y = 4 + 3 * X + np.random.randn(13700,1)

x_b = np.c_[np.ones((13700,1)), X]

def stochastic_gradient_descent(X, y, theta, learning_rate, n_epocs):
    m = len(y)
    for epoch in range(n_epochs):
        for i in range(m):
            random_index = np.random.randint(m)
            xi = X[random_index:random_index+1]
            yi = y[random_index:random_index+1]
            gradients = 2 * xi.T @ (xi @ theta - yi)
            theta -= learning_rate * gradients
    return theta

theta = np.random.randn(2,1)
learning_rate = 0.01
n_epochs = 52

theta_opt = stochastic_gradient_descent(x_b, y, theta, learning_rate, n_epochs)
print('Optimized Parameters: ', theta_opt)