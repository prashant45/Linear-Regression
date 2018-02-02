"""
Created on Wed Jan 28 14:10:45 2018

@author: Prashant
@github: github.com/prashant45
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_data(file_path):
    data_train = pd.read_csv(file_path, header=None)
    l = len(data_train)
    data_train.columns = ['input', 'output']
    data_train['extra feature'] = 1
    
    X_train = np.asarray(data_train[['extra feature', 'input']]).reshape(l,2)
    Y_train = np.asarray(data_train['output']).reshape(l,1)
    
    return [X_train, Y_train]

def scatter_plot_data(X, Y):
    plt.figure()
    plt.scatter(X, Y, label='Data', color='red')
    plt.xlabel('Population: (*10,000)')
    plt.ylabel('Profit: (*10,000 Euros)')
    plt.title('Linear Regression')
    
def linear_fit_plot(X, Y, phase):
    if (phase == 'training'):
        plt.plot(X, Y, label='Linear Fit', color='blue')
        plt.legend()
    elif (phase == 'testing'):
        plt.plot(X, Y, label='Linear Fit Predictions', color='blue')
        plt.legend()
    
def cost_function_plot(cost):
    plt.figure()
    plt.plot(cost, label='Cost')
    plt.xlabel('Number of Iterations')
    plt.ylabel('Cost')
    plt.title('Cost in Gradient Descent')
    
def computeCost(X, Y, theta, m):
    sum = 0;
    for i in range(m):
        sum += (np.dot(theta, X[i]) - Y[i]) ** 2
    
    return sum/(2*m)

def gradient_descent(X, Y, theta, alpha):
    iterations = 1500
    J_history = np.zeros(iterations).reshape(iterations, 1)
    m = len(Y)
    
    for i in range(iterations):
        A = 0
        B = 0
        
        for j in range(m):
            common_sum = theta.dot(X[j]) - Y[j]
            A += common_sum*X[j,0]
            B += common_sum*X[j,1]
            
        theta[0,0] -= (alpha*A)/m
        theta[0,1] -= (alpha*B)/m
        
        J_history[i] = computeCost(X, Y, theta, m)
        
    return theta, J_history
