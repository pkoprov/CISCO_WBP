# -*- coding: utf-8 -*-
"""
@author: bstarly

"""

import scipy
import matplotlib.pyplot as plt
from scipy.fft import fft
import numpy as np
import pandas as pd


signal_length = 20 #[ seconds ]

def calc_euclidean(x, y):
    return np.sqrt(np.sum((x - y) ** 2))

def calc_mape(x,y):
    return np.mean(np.abs((x - y) / x))

plt.figure(figsize=(12, 10), dpi=300)


#read signal from a single file
dataf = pd.read_csv(f'./VF2_1/test0_no tool.csv')
y = dataf['3'].to_list()
sample_rate = len(y)/signal_length  # sampling rate [Hz]    
dt = 1.0/ sample_rate # time between two samples [s]
df = 1/ signal_length # frequency between points in frequency domain [Hz]
t = np.arange(0, signal_length , dt) #the time vector
n_t=len(t) # length of time vector

# plot input data y against time
plt.subplot (1, 1, 1)
plt.plot (t,y, label ='input data ')
plt.xlabel ('time [s]')
plt.ylabel ('signal ')
    
# read signal from multiple files
for i in range (1,8):
    dataf = pd.read_csv(f'test{str(i)}.csv')
    y = dataf['3'].to_list()
    sample_rate = len(y)/signal_length  # sampling rate [Hz]    
    dt = 1.0/ sample_rate # time between two samples [s]
    df = 1/ signal_length # frequency between points in frequency domain [Hz]
    t = np.arange(0, signal_length , dt) #the time vector
    n_t=len(t) # length of time vector

    # plot input data y against time
    plt.subplot (7, 1, i)
    plt.plot (t,y, label ='input data ')
    plt.xlabel ('time [s]')
    plt.ylabel ('signal ')
    
       
plt.show() #and display plot on screen


#FIND EUCLIDEAN AND MAPE SCORES between reference and test
colnames=['TIME', 'X', 'Y', 'Z', 'Avg'] 
refDF = pd.read_csv(f'test1.csv', names=colnames, skiprows=1)
size = refDF.shape[0]
s1 = refDF['Avg'][:size]

for i in range (2,8):
    dataf = pd.read_csv(f'test{str(i)}.csv', names=colnames, skiprows=1)
    s2 = dataf['Avg'][:size]
    euc_dist = calc_euclidean(s1, s2)
    mape_dist = calc_mape(s1, s2)
    if i!=2:
        pct_euc_change = abs(euc_dist - prev_euc_dist) / prev_euc_dist
        pct_mape_change = abs(mape_dist - prev_mape_dist) / prev_mape_dist
    else:
        pct_mape_change = 0
        pct_euc_change = 0
        
    print(f" Test {i}: Euclidean= {euc_dist}, %change={pct_euc_change} and MAPE = {mape_dist}, %change = {pct_mape_change}")
    prev_mape_dist = mape_dist
    prev_euc_dist = euc_dist
    
