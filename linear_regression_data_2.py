"""
Created on Wed Jan 28 14:10:45 2018

@author: Prashant
@github: github.com/prashant45
"""

import numpy as np
import util_functions

#Load training data set 2
trainig_data = util_functions.load_data('./data_set_2/train.csv')

#Split the data into Input and Output
X_train = trainig_data[0] #Population 
Y_train = trainig_data[1] #Profit of food truck business
m = len(X_train)

#Plot the data
util_functions.scatter_plot_data(X_train[:,1], Y_train)

#Some gradient descent settings
theta = np.zeros((1,2))
iterations = 1500
alpha = 0.0005

#Compute Initial Cost
J = util_functions.computeCost(X_train, Y_train, theta, m)

#Run gradient descent algorithm
algorithm_param = util_functions.gradient_descent(X_train, Y_train, theta, alpha)
theta = algorithm_param[0]
J_history = np.insert(algorithm_param[1], 0, J)

#Plot the linear fit on the data
fitted_line = np.dot(X_train, theta.T)
util_functions.linear_fit_plot(X_train[:,1], fitted_line, phase='training')

#Plot the cost function
util_functions.cost_function_plot(J_history)

#Load and predit for data_set_2
test_data = util_functions.load_data('./data_set_2/test.csv')
X_test = test_data[0]
Y_test = test_data[1]
m = len(X_test)

prediction_test = np.zeros((m,1))
prediction_test = np.dot(X_test, theta.T)

util_functions.scatter_plot_data(X_test[:,1], Y_test)
util_functions.linear_fit_plot(X_test[:,1], prediction_test, phase='testing')
