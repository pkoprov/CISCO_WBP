# -*- coding: utf-8 -*-
"""
@author: bstarly

"""

import scipy
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, ifft
import numpy as np
import pandas as pd

# create signal
colnames=['TIME', 'X', 'Y', 'Z', 'Avg']
datafr1 = pd.read_csv("VF-2-1 wTool/22_01_07/2022_01_07 13-12-25_VF-2-1_with tool.csv", names=colnames, skiprows=1)
datafr2 = pd.read_csv("VF-2-1 wTool/22_01_07/2022_01_07 13-12-55_VF-2-1_with tool (not moving).csv", names=colnames, skiprows=1)
df = (datafr1, datafr2)
signal_length = 5 #[ seconds ]
plt.figure(figsize=(24, 12), dpi=100)

for i, dat in enumerate(df):

    x= dat['X'] - dat['X'].mean()
    y= dat['Y'] - dat['Y'].mean()
    z= dat['Z'] - dat['Z'].mean()

    # compute fourier transform
    for coord in ('x','y','z'):
        # here I am using globals() to call and create global variables
        globals()[coord] = np.array(globals()[coord])
        globals()[f'f{coord}'] = fft(globals()[coord])

    size = dat.shape[0]
    sample_rate = size /signal_length # sampling rate [Hz]
    dt = 1.0/ sample_rate # time between two samples [s]
    df = 1/ signal_length # frequency between points in frequency domain [Hz]

    # here I am using globals() to call and create global variables
    globals()[f't{i}'] = np.linspace(0, signal_length , size) #the time vector
    globals()[f'n_t{i}'] = len(globals()[f't{i}']) # length of time vector
    globals()[f'accel{i}'] = (x,y,z)
    globals()[f'fft_accel{i}'] = (fx,fy,fz)
    globals()[f'freqs{i}'] = df * np.arange(0 ,(globals()[f'n_t{i}']-1)/2. , dtype ='d') #d= double precision float
    globals()[f'n_freq{i}'] = len ( globals()[f'freqs{i}'] )

    plt.subplot(2, 2, i*2+1)
    plt.plot(globals()[f't{i}'], globals()[f'accel{i}'][0], label='input data ')
    plt.xlabel('time [s]')
    plt.ylabel(f'{colnames[1]} signal')
    plt.subplot(2, 2, i*2+2)
    plt.plot(globals()[f'freqs{i}'], abs(globals()[f'fft_accel{i}'][0][0: globals()[f'n_freq{i}']]))
    plt.xlabel('frequency [Hz]')
    plt.ylabel('abs(DFT( signal ))')
    label = ('abs( fourier transform )')
    plt.xticks(np.arange(min(globals()[f'freqs{i}']), max(globals()[f'freqs{i}']) + 1, 10.0))


# save plot to disk
plt.savefig ('fft1.png')
plt.show() #and display plot on screen

arrfreqs = fftfreq(len(f))
print(arrfreqs.min(), arrfreqs.max())

# Find the peak in the coefficients
idx = np.argmax(np.abs(f))
freq = arrfreqs[idx]
freq_in_hertz = abs(freq * sample_rate )
print(freq_in_hertz)

newf = np.abs(f)
newf.sort()
print(newf[-5:]* sample_rate)
