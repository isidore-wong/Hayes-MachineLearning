# _*_ coding:utf-8 _*_
from math import exp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

"""
@author: Isidore
@email:616132717@qq.com
@file: logistic_regression.py
@time: 2019/02/18  12:27
@version: 
"""

"""
程序目的：LR是经典的分类方法,处理二分类问题
"""

# loads iris data from the sklearn.datasets, return the data what we need
def create_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns = iris.feature_names)
    df['label'] = iris.target
    df.columns = ['sepal length', 'sepal width ', 'petal length ', 'petal width ', 'label']
    data = np.array(df.iloc[:100, [0, 1, -1]])
    # print(data)
    return data[:, :2], data[:, -1]

# set the data of X, Y, and set the proportion of train and test
X, y = create_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


# logistic regression
class LogisticReressionClassifier:
    def __init__(self, max_iter=200, learning_rate=0.01):
        self.max_iter = max_iter
        self.learning_rate = learning_rate

    def sigmoid(self, x):
        return 1 / (1 + exp(-x))

    def data_matrix(self, X):
        data_mat = []
        for d in X:
            data_mat.append([1.0, *d])
        return data_mat

    def fit(self, X, y):
        # label = np.mat(y)
        data_mat = self.data_matrix(X)  # m*n
        self.weights = np.zeros((len(data_mat[0]), 1), dtype=np.float32)

        for iter_ in range(self.max_iter):
            for i in range(len(X)):
                result = self.sigmoid(np.dot(data_mat[i], self.weights))
                error = y[i] - result
                self.weights += self.learning_rate * error * np.transpose([data_mat[i]])
        print('LogisticRegression Model(learning_rate={},max_iter={})'.format(self.learning_rate, self.max_iter))

    # def f(self, x):
    #     return -(self.weights[0] + self.weights[1] * x) / self.weights[2]

    def score(self, X_test, y_test):
        right = 0
        X_test = self.data_matrix(X_test)
        for x, y in zip(X_test, y_test):
            result = np.dot(x, self.weights)
            if (result > 0 and y == 1) or (result < 0 and y == 0):
                right += 1
        return right / len(X_test)

lr_clf = LogisticReressionClassifier()
lr_clf.fit(X_train, y_train)

# the test sets' prediction accroding to the train sets
lr_clf.score(X_test, y_test)

# the parameter of w
lr_clf.weights

# plot the figture of logistic regression
x_points = np.arange(4, 8)
y_points = -(lr_clf.weights[1] * x_points + lr_clf.weights[0] ) / lr_clf.weights[2]
plt.plot(x_points, y_points)

# lr_clf_show_graph()
plt.scatter(X[:50, 0], X[:50, 1], label='0')
plt.scatter(X[50: , 0], X[50: , 1], label='1')
plt.legend()


