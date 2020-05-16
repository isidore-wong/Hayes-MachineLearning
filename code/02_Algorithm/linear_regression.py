# _*_ coding:utf-8 _*_
import numpy as np
import matplotlib.pyplot as plt

"""
@author: Isidore
@email:616132717@qq.com
@file: linear_regression.py
@time: 2019/02/01  10:23
@version: 1.0
"""

"""
程序目的：to strengthen the cognition of gradient descent by plotting scatter and fitting
"""

# define the size of points dataset
m = 20

# the x-coordinate and dummy value(x0, x1) of the points
X0 = np.ones((m, 1))
X1 = np.arange(1, m+1).reshape(m, 1)
X = np.hstack((X0, X1))

# the y-coordinate of the points
y = np.array([
    3, 4, 5, 5, 2, 4, 7, 8, 11, 8, 12,
    11, 13, 13, 16, 17, 18, 17, 19, 21
]).reshape(m, 1)

# the learning rate alpha
alpha = 0.01

# the cost function J definition
def cost_funciton(theta, X, y):
    diff = np.dot(X, theta) - y
    return (1.0/2*m) * np.dot(np.transpose(diff), diff)

# the gradient of cost function J definition
def gradient_function(theta, X, y):
    diff = np.dot(X, theta) - y
    return (1./m) * np.dot(np.transpose(X), diff)

# perform gradient descent
def gradient_descent(X, y, alpha):
    theta = np.array([1, 1]).reshape(2, 1)
    gradient = gradient_function(theta, X, y)
    while not np.all(np.absolute(gradient) <= 1e-5):
        theta = theta - alpha * gradient
        gradient = gradient_function(theta, X, y)
    return theta

optimal_theta = gradient_descent(X, y, alpha)
predict_y = np.dot(X, optimal_theta)

print("the optimal value of theta:", optimal_theta)
print("the value of cost function:", cost_funciton(optimal_theta, X, y)[0, 0])
print("****************************************")

# plot scatter and fitting
plt.figure(1)
plt.subplot(221)
plt.plot(X1, y, "o")
plt.legend(["scatter"], loc='upper left', frameon=False)

plt.subplot(222)
plt.plot(X1, y, "bo",
         X1, predict_y, "r-")
plt.legend(["fitting"], loc='upper left', frameon=False)
plt.show()
