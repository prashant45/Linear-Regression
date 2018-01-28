# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def load_data(file_path):
    data_train = pd.read_csv(file_path, header=None);
    data_train.columns = ['input', 'output']
    data_train['extra feature'] = 1
    
    X_train = data_train[['extra feature', 'input']].as_matrix()
    Y_train = data_train['output'].as_matrix()
    
    return [X_train, Y_train]

def scatter_plot_data(X, Y):
	plt.scatter(X, Y, label="Data", color="red")
	plt.xlabel('Input')
	plt.ylabel('Output')
	plt.title('Linear Regression')
    
def linear_fit_plot(X, Y):
	plt.plot(X, Y, label='Linear Fit', color='blue')
	plt.legend()
	plt.show()
    
def computeCost(X, Y, theta, m):
    sum = 0;
    for i in range(m):
        sum += (theta.dot(X[i]) - Y[i]) ** 2
    
    return sum/(2*m)

def gradient_descent(X, Y, theta, alpha):
    iterations = 1500
    J_history = np.zeros(iterations)
    m = len(Y)
    
    for i in range(iterations):
        A = 0
        B = 0
        
        for j in range(m):
            common_sum = theta.dot(X[j]) - Y[j]
            A += common_sum*X[j,0]
            B += common_sum*X[j,1]
            
        theta[0] -= (alpha*A)/m;
        theta[1] -= (alpha*B)/m;
        
        J_history[i] = computeCost(X, Y, theta, m)
        
    return theta, J_history
        
        
        
        
        
        
        