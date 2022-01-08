# -*- coding: utf-8 -*-
"""
@author: bstarly

"""
import os
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, ifft
import numpy as np
import pandas as pd

# create signal
colnames=['TIME', 'X', 'Y', 'Z']
df_list = []
folders = ["VF-2-1 wTool/22_01_07", 'VF-2-1 wTool/22_01_07']
for folder in folders:
    for file in os.listdir(folder):
        try:
            df_list.append(pd.read_csv(f'{folder}/{file}', names=colnames, skiprows=1))
        except:
            pass

signal_length = 5 #[ seconds ]
colnames = ("With tool", "Without tool")
fig, big_axes = plt.subplots( 1,2, sharey=True)
for n, big_ax in enumerate(big_axes):
    big_ax.set_title(colnames[n], fontsize=16)

    # Turn off axis lines and ticks of the big subplot
    big_ax.tick_params(labelcolor=(1.,1.,1., 0.0), top='off', bottom='off', left='off', right='off')
    # removes the white frame
    big_ax._frameon = False

for i, dat in enumerate(df_list):

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

    # plot frequency spectrum
    fig.add_subplot(int(len(df_list)/2), 2, i + 1)
    plt.plot(globals()[f'freqs{i}'], abs(globals()[f'fft_accel{i}'][0][0: globals()[f'n_freq{i}']]))
    plt.xlabel('frequency [Hz]')
    plt.ylabel('abs(DFT( signal ))')
    plt.xticks(np.arange(min(globals()[f'freqs{i}']), max(globals()[f'freqs{i}']) + 1, 10.0))


# save plot to disk
plt.savefig ('fft1.png')
plt.show() #and display plot on screen

for i in range(6):
    arrfreqs = fftfreq(len(globals()[f'fft_accel{i}'][0]))
    # print(arrfreqs.min(), arrfreqs.max())
    # Find the peak in the coefficients
    idx = np.argmax(np.abs(globals()[f'fft_accel{i}'][0]))
    freq = arrfreqs[idx]
    sample_rate = globals()[f'fft_accel{i}'][0].shape[0] / signal_length
    freq_in_hertz = abs(freq * sample_rate )
    print(f"Dominant frequency is {round(freq_in_hertz,2)} Hz")
