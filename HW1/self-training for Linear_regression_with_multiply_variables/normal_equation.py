# -*- coding: utf-8 -*-
import numpy as np

def normal_equation(dataset):
    n = len(dataset[1]) - 1
    m = len(dataset)
    #print(m, n)

    x = np.mat(dataset)
    x = np.c_[np.ones(m),x]
    y = x[:, -1]
    x = np.delete(x, -1, axis = 1)

    theta = (x.T * x).I * x.T * y
    print(theta)
    #print(x * theta / 2 / m)