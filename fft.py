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
datafr = pd.read_csv(r".\VF-2_2\VF-2-2 wTool\2021_12_23 21-09-59_VF-2-2_with tool.csv", names=colnames, skiprows=1)
y= datafr['Y'] - datafr['Y'].mean()
size = datafr.shape[0]
signal_length = 12 #[ seconds ]
sample_rate = size /signal_length # sampling rate [Hz]
dt = 1.0/ sample_rate # time between two samples [s]
df = 1/ signal_length # frequency between points in frequency domain [Hz]
t = np.arange(0, signal_length , dt) #the time vector
n_t = len(t) # length of time vector

# compute fourier transform
y = np.array(y)
f = fft(y)

# work out meaningful frequencies in fourier transform

freqs = df * np.arange(0 ,(n_t-1)/2. , dtype ='d') #d= double precision float
n_freq = len ( freqs )

# plot input data y against time
plt.figure(figsize=(24, 12), dpi=80)
plt.subplot (2, 1, 1)
plt.plot (t,y, label ='input data ')
plt.xlabel ('time [s]')
plt.ylabel ('signal ')

# plot frequency spectrum
plt.subplot (2 ,1 ,2)
plt.plot (freqs ,abs(f[0: n_freq]))
label =('abs( fourier transform )')
plt.xlabel ('frequency [Hz]')
plt.ylabel ('abs(DFT( signal ))')
plt.xticks(np.arange(min(freqs), max(freqs)+1, 10.0))

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