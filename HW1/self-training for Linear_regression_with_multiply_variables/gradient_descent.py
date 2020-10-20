# -*- coding: utf-8 -*-

import numpy as np
import math
import readin
import matplotlib.pyplot as plt

def gradient_descent(dataset):
    
    n = len(dataset[1]) - 1
    m = len(dataset)
    #print(m, n)

    x = np.mat(dataset)
    x = np.c_[np.ones(m),x]
    y = x[:, -1]
    x = np.delete(x, -1, axis = 1)

    theta = np.matrix(np.ones((n+1,1))) 
    alpha = 0.000001

    #tmp1 = map * theta - map[:, n+1]
    #cost = np.sum(np.multiply(tmp1, tmp1)) / 2 / m

    print(x.T.shape, x.shape, theta.shape, y.shape)

    yy = []
    xx = []
    i = 0
    while 1:
        delta_theta = x.T * (x * theta - y) * alpha / m
        #print(delta_theta[0][0],theta[0][0] - delta_theta[0][0])
        if (np.max(delta_theta) < 0.0001) and (np.min(delta_theta) > -0.0001):
            break
        theta -= delta_theta
        
        xx.append(i)
        i += 1
        yy.append(delta_theta[1,0])
    
    plt.scatter(xx, yy)
    plt.show()

    print(theta)
    #print(x * theta / 2 / m)
