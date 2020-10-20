# -*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib.pyplot as plt

def std_init(X):
    u = np.mean(X, axis = 0)
    sigma = np.std(X, axis = 0)
    X = (X - u) / sigma
    return X

def normal_equation(X, Y):
    theta = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), Y)
    return theta

data = np.loadtxt('./HW1/boston_house_price_dataset.txt')
X_data = data[:, 0:-1].reshape(data.shape[0], data.shape[1] - 1)
Y_data = data[:, -1].reshape(data.shape[0], 1)
#plt.scatter(X_data, Y_data)
#plt.show()

X_data = std_init(X_data)
X_train = np.hstack((np.ones((X_data.shape[0], 1)), X_data))
Y_train = Y_data

theta = normal_equation(X_train, Y_train)
print(theta)
loss = 1 / 2 / X_train.shape[0] * np.sum(np.power((np.dot(X_train, theta) - Y_train), 2))
print(loss)