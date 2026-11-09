import sympy as sp
import numpy as np

# x,y = sp.symbols('x y')
# f = x**2 + 3*y**2 - 4*x*y
# grad_x = sp.diff(f,x)
# grad_y = sp.diff(f,y)
# print('Gradients: ')
# print('Grad X:', grad_x)
# print('Grad Y:', grad_y)

def gradient_descent(X, y, theta, learning_rate, iterations):
    m = len(y)
    for _ in range(iterations):
        predictions = np.dot(X, theta)
        errors = predictions - y
        gradients = (1/m) * np.dot(X.T, errors)
        theta -= learning_rate * gradients
    return theta

# sample data
X = np.array([[1,1], [1,2], [1,3]])
y = np.array([2.12,2.67,3.71])
theta = np.array([0.1, 0.1])
learning_rate = 0.1
iterations = 1000

# perform gradient descent
optimized_theta = gradient_descent(X, y, theta, learning_rate, iterations)
print('Optimized Parameters: ', optimized_theta)