# -*- coding: utf-8 -*-
"""

@author: Zion
"""

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from cvxopt import matrix, solvers
from scipy import stats
from l1regls import *
import numpy as np
import datetime
import json #to manipulate yec output
import pandas as pd
import cvxpy as cp
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def dual_lasso(X1, X2, mu=7, gam=2):
    n = X1.size[1]
    print('debug', X1.size)
    print('debug', X2.size)

    #2) Define opt fcns & params
    e = matrix(np.random.rand(n,1))
    w = matrix(np.random.rand(n,1))
    K = X2.T #use X2 instead of a kernel
    #y = K*e

    # STEP 1: [DO THE L1 ALTERNATING NORM HERE]
    eps = 0.0001*np.ones(n)
    dw, de= matrix(0.5*np.ones(n)), matrix(0.5*np.ones(n)) #deltas to judge convergence
    #mu,gam = 100, 140 #experiment with these
    #mu,gam = 7, 2 #experiment with these

    w_old, e_old = w,e
    #repeat until within error tolerance
    #multiply by 1/mu to get the proper form
    while np.greater_equal(dw, eps).all() or np.greater_equal(de, eps).all():
        #fix e, find w
        w_new = l1regls((1/mu)*X1, (1/mu)*X2*e_old) #|X1'w - y| + u|w| 
        #fix w, find e
        e_new = l1regls((1/gam)*X2, (1/gam)*X1*w_new)
        #compute deltas
        dw = np.absolute(w_new-w_old)
        de = np.absolute(e_new-e_old)
        #reset vars        
        w_old, e_old = w_new, e_new

    w,e = w_new, e_new    
    #3) Solve with CVXOpt
    print('answer: ', np.round(np.array(w),2))
    print(np.round(np.array(e),2))

    corr = stats.pearsonr(X1*w, X2*e)[0]
    print('CORRELATION: ', corr)

    return w,e

def quad_prog(X1=None, X2=None):
    #define loss to plug into objective
    def loss_fn(X, Y, beta):
        return cp.norm2(X @ beta - Y)#**2 #@ for matrix multiplication
    
    #penalize mag; plugs into objective 
    def regularizer(beta):
        return cp.norm1(beta)
    
    #cvxpy will minimize over this objective
    '''change this up'''
    def objective_fn(X, Y, beta, lambd): #lambd = mu for us
        return loss_fn(X, Y, beta) + lambd * regularizer(beta)
    
    
    def mse(X, Y, beta):
        return (1.0 / X.shape[0]) * loss_fn(X, Y, beta).value
    
    
    def generate_data(m=100, n=20, sigma=5, density=0.2):
        "Generates data matrix X and observations Y."
        np.random.seed(1)
        beta_star = np.random.randn(n)
        idxs = np.random.choice(range(n), int((1-density)*n), replace=False)
        for idx in idxs:
            beta_star[idx] = 0
        X = np.random.randn(m,n)
        Y = X.dot(beta_star) + np.random.normal(0, sigma, size=m)
        return X, Y, beta_star
    
    m = 100
    n = 20
    sigma = 5
    density = 0.2
    
    if X1 == None or X2 == None:
        X, Y, _ = generate_data(m, n, sigma)
        X2, Y2, __ = generate_data(m, n, 7)
        X1 = matrix(X[:50, :]) #rename for my own variables
        X2 = matrix(X2[:50, :])
        
    mu,gam = 100, 140 #experiment with these
    
    #first minimize w (beta)
    beta = cp.Variable(n) #w
    alpha = np.random.rand(n,1) #e
    lambd = cp.Parameter(nonneg=True)
    '''seems like we dont need constraints + obj reg'''
    
    m = 50
    eps = 0.0001*np.ones(n)
    dbeta, dalpha= 0.5*np.ones(n), 0.5*np.ones(n) #deltas to judge convergence
    
    
    while np.greater_equal(dbeta, eps).all() or np.greater_equal(dalpha, eps).all():
        beta_old, alpha_old = beta, alpha
        
        constraints = [cp.norm(beta, 1) <= mu] #see thing
        problem = cp.Problem(cp.Minimize(objective_fn(X1, (X2@alpha).reshape(m,), beta, lambd)))#, constraints)
        lambd.value = mu
        problem.solve()
        #np.round(beta.value,2) #beta = x
        
        #second minimize alpha
        constraints = [cp.norm(alpha, 'inf') <= 1] #see thing
        problem = cp.Problem(cp.Minimize(objective_fn(X2, (X1@beta.value).reshape(m,), alpha, gam)), constraints)
        lambd.value = gam
        problem.solve()
        np.round(alpha,2)
    
        #assess convergence
        dbeta = np.absolute(beta.value-beta_old.value)
        dalpha = np.absolute(alpha-alpha_old)
    
    w = np.round(beta.value,2)
    e = np.round(alpha,2)
    print('w: ', w) #beta = x
    print('e: ', e) #beta = x
    
    #STEP 3: Compute correlation to check answer of Algo 6 implementation
    corr = stats.pearsonr(X1@beta.value, (X2@alpha).reshape(50,))[0]
    print('CORRELATION: ', corr)
    
    return w,e

def stock_conversion(vec, stock_basket):
    '''
    input: w, e vector
    output: list of corresponding stocks sorted in order of canonical weights
    '''
    soi = []#stocks of interest
    stock_weights = [] #(stock, weight)
    sort_dict = {} #store weights for sorting
    for i in range(len(vec)):
        if np.round(vec[i], 2) != 0:
            stock = stock_basket[i]
            sort_dict[stock] = abs(vec[i])
            stock_weights.append((stock, abs(vec[i])))
            soi.append(stock)
    soi.sort(key= lambda x: sort_dict[x], reverse = True)
    return soi

yrs = list(range(2015,2021)) #debug


def plot_variates(y1, y2, t1=None, t2=None, lab1=f'Stock Prices', lab2=f'Earnings'): 
    
    # Pull plot data
    if t1 == None:
        t1 = yrs #debug
        
    if t2 == None:
        t2 = yrs #debug
    fig, ax1 = plt.subplots()
    
    color = 'tab:red'
    ax1.set_xlabel('Quarters')
    ax1.set_ylabel(lab1, color=color)
    ax1.plot(y1, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    
    color = 'tab:blue'
    ax2.set_ylabel(lab2, color=color)  # we already handled the x-label with ax1
    ax2.plot(y2, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    print(f'{lab1} vs {lab2}') #DEBUG
    spc = 4
    plt.xticks([0, spc ,2*spc ,3*spc, 4*spc, 5*spc], yrs) #not show too many ticks
    plt.show()
