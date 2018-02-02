"""
Created on Wed Jan 28 14:10:45 2018

@author: Prashant
@github: github.com/prashant45
"""

import numpy as np
import util_functions

#Load data set 1 or 2
trainig_data = util_functions.load_data('./data_set_1/ex1data1.txt')

#Split the data into Train, Test
X_train = trainig_data[0][:90, :] #Population of a city
Y_train = trainig_data[1][:90, :] #Profit of food truck business
X_test = trainig_data[0][90:, :]
Y_test = trainig_data[1][90:, :]
m_train = len(X_train)
m_test = len(X_test)

#Plot the data
util_functions.scatter_plot_data(X_train[:,1], Y_train)

#Some gradient descent settings
theta = np.zeros((1,2))
iterations = 1500
alpha = 0.001

#Compute Initial Cost
J = util_functions.computeCost(X_train, Y_train, theta, m_train)

#Run gradient descent algorithm
algorithm_param = util_functions.gradient_descent(X_train, Y_train, theta, alpha)
theta = algorithm_param[0]
J_history = np.insert(algorithm_param[1], 0, J)

#Plot the linear fit on the data
fitted_line = np.dot(X_train, theta.T)
util_functions.linear_fit_plot(X_train[:,1], fitted_line, phase='training')

#Predit
predictions = np.dot(X_test, theta.T)
util_functions.scatter_plot_data(X_test[:,1], Y_test)
util_functions.linear_fit_plot(X_test[:,1], predictions, phase='testing')


#Plot the cost function
util_functions.cost_function_plot(J_history)

