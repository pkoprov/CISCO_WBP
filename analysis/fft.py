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
datafr1 = pd.read_csv("VF-2-1 wTool/2022_01_07 13-12-25_VF-2-1_with tool.csv", names=colnames, skiprows=1)
datafr2 = pd.read_csv("VF-2-1 wTool/2022_01_07 13-12-55_VF-2-1_with tool (not moving).csv", names=colnames, skiprows=1)
df_tuple = (datafr1, datafr2)

signal_length = 5 #[ seconds ]
rownames = ("4000 RPM", "0 RPM")
fig, big_axes = plt.subplots( 2,1, sharey=True)
for n, big_ax in enumerate(big_axes):
    big_ax.set_title(rownames[n], fontsize=16)

    # Turn off axis lines and ticks of the big subplot
    big_ax.tick_params(labelcolor=(1.,1.,1., 0.0), top='off', bottom='off', left='off', right='off')
    # removes the white frame
    big_ax._frameon = False

for i, dat in enumerate(df_tuple):

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
    globals()[f'accel{i}'] = (x,y,z) # tuple with acceleration data
    globals()[f'fft_accel{i}'] = (fx,fy,fz) # tuple with frequencies data
    globals()[f'freqs{i}'] = df * np.arange(0 ,(globals()[f'n_t{i}']-1)/2. , dtype ='d') #d= double precision float
    globals()[f'n_freq{i}'] = len ( globals()[f'freqs{i}'] )

    # plot input data y against time
    fig.add_subplot(len(df_tuple), 2, i*2+1)
    plt.plot(globals()[f't{i}'], globals()[f'accel{i}'][0], label='input data ')
    plt.title('Time domain')
    plt.xlabel('time [s]')
    plt.ylabel(f'{colnames[1]} signal')


    # plot frequency spectrum
    fig.add_subplot(len(df_tuple), 2, i*2+2)
    plt.plot(globals()[f'freqs{i}'], abs(globals()[f'fft_accel{i}'][0][0: globals()[f'n_freq{i}']]))
    plt.xlabel('frequency [Hz]')
    plt.ylabel('abs(DFT( signal ))')
    plt.title('Frequency spectrum')
    plt.xticks(np.arange(min(globals()[f'freqs{i}']), max(globals()[f'freqs{i}']) + 1, 10.0))


# save plot to disk
plt.savefig ('VF-2-1 wTool/22_01_07_spindle only/fft1.png')
plt.show() #and display plot on screen

arrfreqs = fftfreq(len(fft_accel0[0]))
print(arrfreqs.min(), arrfreqs.max())

# Find the peak in the coefficients
idx = np.argmax(np.abs(fft_accel0[0]))
freq = arrfreqs[idx]
freq_in_hertz = abs(freq * sample_rate )
print(f"Dominant frequency is {round(freq_in_hertz,2)} Hz")
