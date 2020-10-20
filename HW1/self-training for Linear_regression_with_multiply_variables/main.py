# -*- coding: utf-8 -*-

import readin
import gradient_descent
import normal_equation
import os

print(os.getcwd())
gradient_descent.gradient_descent(readin.readin())
normal_equation.normal_equation(readin.readin())

