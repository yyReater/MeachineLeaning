# -*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib.pyplot as plt
#from matplotlib.pyplot import savefig

def std_init(X):
    u = np.mean(X, axis = 0)
    sigma = np.std(X, axis = 0)
    X = (X - u) / sigma
    return X

def init_theta(shape):
    #np.random.seed(1)
    #theta = np.random.random(shape)
    theta = np.ones(shape)
    return theta

def compute_J(X, Y, theta):
    m = X.shape[0]
    loss = 1 / 2 / m * np.sum(np.power((np.dot(X, theta) - Y), 2))    
    return loss

def compute_gradient(X, Y, theta):
    m = X.shape[0]
    delta_theta = 1 / m * (np.dot(X.T, np.dot(X, theta) - Y))
    return delta_theta

def update_theta(theta, delta_theta, alpha):
    theta -= alpha * delta_theta
    return theta

def gradient_descent(X, Y, theta, limit = 0.001, alpha = 0.1):
    loss_history = []
    theta_history = []

    while True:
        loss = compute_J(X, Y, theta)
        delta_theta = compute_gradient(X, Y, theta)
        theta = update_theta(theta, delta_theta, alpha)

        loss_history.append(loss)
        theta_history.append(theta)

        if(np.max(np.abs(delta_theta)) < limit):
            break
        #print(np.max(np.abs(delta_theta)))
    
    return loss_history, theta_history, theta

data = np.loadtxt('./HW1/boston_house_price_dataset.txt')
X_data = data[:, 0:-1].reshape(data.shape[0], data.shape[1] - 1)
Y_data = data[:, -1].reshape(data.shape[0], 1)
#plt.scatter(X_data, Y_data)
#plt.show()

X_data = std_init(X_data)
X_train = np.hstack((np.ones((X_data.shape[0], 1)), X_data))
Y_train = Y_data
print(X_train)
theta = init_theta((X_train.shape[1], 1))
print('theta_init:', theta)
loss_init = compute_J(X_train, Y_train, theta)
print('loss_init:', loss_init)

loss_history, theta_history, theta = gradient_descent(X_train, Y_train, theta)
print(theta)
plt.plot(loss_history)
plt.show()
print("loss = ", loss_history[-1])