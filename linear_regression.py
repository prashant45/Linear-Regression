"""
Created on Wed Jan 28 14:10:45 2018

@author: Prashant
@github: github.com/prashant45
"""

import numpy as np
import util_functions

# Load data
trainig_data = util_functions.load_data('./data_set_1/ex1data1.txt')
X_train = trainig_data[0] #Population 
Y_train = trainig_data[1] #Profit of food truck business
m = len(X_train)

#Plot the data
util_functions.scatter_plot_data(X_train[:,1], Y_train)

#Some gradient descent settings
theta = np.zeros(2)
iterations = 1500
alpha = 0.001

#Compute Initial Cost
J = util_functions.computeCost(X_train, Y_train, theta, m)

#Run gradient descent algorithm
algorithm_param = util_functions.gradient_descent(X_train, Y_train, theta, alpha)
theta = algorithm_param[0]
J_history = algorithm_param[1]

#Plot the linear fit on the data
fitted_line = X_train.dot(theta)
util_functions.linear_fit_plot(X_train[:,1], fitted_line)

#Plot the cost function
util_functions.cost_function_plot(J_history)

#Predit 
X_test = 3.5
predict = theta.dot(np.array([1, X_test]))
printResult = 'For the population of ' + str(X_test*10000) + ' the model predits the food truck profit of ' + str(predict*10000)
print(printResult)
