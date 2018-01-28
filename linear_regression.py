# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import util_functions

# Load data
trainig_data = util_functions.load_data('./data_set_1/ex1data1.txt');
X_train = trainig_data[0]
Y_train = trainig_data[1]
m = len(X_train)

#Plot the data
util_functions.scatter_plot_data(X_train[:,1], Y_train)

#Some gradient descent settings
theta = np.zeros(2);
iterations = 1500;
alpha = 0.01;

#Compute Initial Cost
J = util_functions.computeCost(X_train, Y_train, theta, m);

#Run gradient descent algorithm
algorithm_param = util_functions.gradient_descent(X_train, Y_train, theta, alpha)
theta = algorithm_param[0]
J_history = algorithm_param[1]

#Plot the linear fit on the data
fitted_line = X_train.dot(theta)
util_functions.linear_fit_plot(X_train[:,1], fitted_line)

#Predit 
X_test = 3.5
predict = theta.dot(np.array([1, X_test]))
printResult = 'For the population of ' + str(X_test*10000) + ' the model predits a profit of ' + str(predict*10000)
print(printResult)