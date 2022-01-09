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
datafr1 = pd.read_csv("VF-2-1 wTool/22_01_08/2022_01_08 16-28-21_VF-2-1_with tool.csv", names=colnames, skiprows=1)
datafr2 = pd.read_csv("VF-2-1 wTool/22_01_08/2022_01_08 16-28-45_VF-2-1_with tool.csv", names=colnames, skiprows=1)
datafr3 = pd.read_csv("VF-2-1 wTool/22_01_08/2022_01_08 16-29-07_VF-2-1_with tool.csv", names=colnames, skiprows=1)
df_tuple = (datafr1, datafr2,datafr3)

signal_length = 16 #[ seconds ]

# create a figure window with 3 rows
rownames = ("1st attempt", "2nd attempt", '3rd attempt')
fig, big_axes = plt.subplots(3,1, sharey=True)
for n, big_ax in enumerate(big_axes):
    big_ax.set_title(rownames[n], fontsize=16, y=1.05)
    # Turn off axis lines and ticks of the big subplot
    big_ax.tick_params(labelcolor=(1.,1.,1., 0.0), top='off', bottom='off', left='off', right='off')
    # removes the white frame
    big_ax._frameon = False

# create plots
for i, dat in enumerate(df_tuple):

    x= dat['X'] - dat['X'].mean()
    y= dat['Y'] - dat['Y'].mean()
    z= dat['Z'] - dat['Z'].mean()

    # compute fourier transform
    for coord in ('x','y','z'):
        # here I am using globals() to call and create global variables
        globals()[f'{coord}{i}'] = np.array(globals()[coord])
        globals()[f'f{coord}{i}'] = fft(globals()[f'{coord}{i}'])

    size = dat.shape[0]
    sample_rate = size /signal_length # sampling rate [Hz]
    dt = 1.0/ sample_rate # time between two samples [s]
    df = 1/ signal_length # frequency between points in frequency domain [Hz]

    # here I am using globals() to call and create global variables
    globals()[f't{i}'] = np.linspace(0, signal_length , size) #the time vector
    globals()[f'n_t{i}'] = len(globals()[f't{i}']) # length of time vector
    globals()[f'accel{i}'] = (globals()[f'x{i}'],globals()[f'y{i}'],globals()[f'z{i}']) # tuple with acceleration data
    globals()[f'fft_accel{i}'] = (globals()[f'fx{i}'],globals()[f'fy{i}'],globals()[f'fz{i}']) # tuple with frequencies data
    globals()[f'freqs{i}'] = df * np.arange(0 ,(globals()[f'n_t{i}']-1)/2. , dtype ='d') #d= double precision float
    globals()[f'n_freq{i}'] = len ( globals()[f'freqs{i}'] )

    # plot input data y against time
    fig.add_subplot(len(df_tuple), 2, i*2+1)
    plt.plot(globals()[f't{i}'], globals()[f'accel{i}'][0], label='input data ')
    plt.xlabel('time [s]')
    plt.ylabel(f'{colnames[1]} signal')
    if i == 0:
        plt.title('Time domain')


    # plot frequency spectrum
    fig.add_subplot(len(df_tuple), 2, i*2+2)
    plt.plot(globals()[f'freqs{i}'], abs(globals()[f'fft_accel{i}'][0][0: globals()[f'n_freq{i}']]))
    plt.xlabel('frequency [Hz]')
    plt.ylabel('abs(DFT( signal ))')
    plt.xticks(np.arange(min(globals()[f'freqs{i}']), max(globals()[f'freqs{i}']) + 1, 10.0),rotation = 45)
    if i == 0:
        plt.title('Frequency spectrum')


def plot_time(attempt, ax:str, dt: np.array):
    # plots accel timeseries
    if ax.lower()=='x':
        ax = 0
    elif ax.lower()=='y':
        ax = 1
    elif ax.lower() == 'z':
        ax = 2
    else:
        print("Incorrect axis name")
        return

    data_range = (dt * sample_rate).astype(int)  # converts time range to data points range

    plt.subplot(3,2,attempt*2+1)
    plt.plot(globals()[f't{attempt}'][data_range[0]:data_range[1]],
             globals()[f'accel{attempt}'][ax][data_range[0]:data_range[1]], label='input data ')
    plt.xlabel('time [s]')
    plt.ylabel(f'X signal')
    if attempt == 0:
        plt.title('Time domain')


def plot_fft(attempt, ax:str,dt: np.array):
    # plots accel fft
    ax = ax.lower()
    if ax =='x':
        ax_ind = 0
    elif ax =='y':
        ax_ind = 1
    elif ax == 'z':
        ax_ind = 2
    else:
        print("Incorrect axis name")
        return

    data_range = (dt * sample_rate).astype(int)  # converts time range to data points range

    plt.subplot(3, 2, attempt*2+2)
    snippet_length = dt[1] - dt[0]
    fft_accel_ax = fft(globals()[f"{ax}{attempt}"][data_range[0]:data_range[1]])  # tuple with frequencies data
    freqs0X = 1 / snippet_length * np.arange(0, (data_range[1] - data_range[0] - 1) / 2.,
                                             dtype='d')  # d= double precision float
    n_freq0X = len(freqs0X)
    plt.plot(freqs0X, abs(fft_accel_ax[0: n_freq0X]))
    plt.xlabel('frequency [Hz]')
    plt.ylabel('abs(DFT( signal ))')
    plt.title('Frequency spectrum')
    plt.xticks(np.arange(min(freqs0X), max(freqs0X) + 1, 10.0), rotation=45)


# choose time period
dt = np.array([0,3])
for i in range(3):
    plt.subplot(3,2,i*2+1) # pick corresponding subplot
    # draw lines
    for x in dt:
        plt.vlines(x, plt.ylim()[0]*0.9, plt.ylim()[1]*0.9, color = 'purple')


plt.figure()
axis = "z"
for i in range(3):
    plot_time(i, axis, dt)
    plot_fft(i,axis, dt)



#
#
# # save plot to disk
# plt.savefig ('VF-2-1 wTool/22_01_07/fft1.png')
# plt.show() #and display plot on screen

arrfreqs = fftfreq(len(fft_accel0X))
print(arrfreqs.min(), arrfreqs.max())

# Find the peak in the coefficients
idx = np.argmax(np.abs(fft_accel0X))
freq = arrfreqs[idx]
sample_rate = round(globals()[f't{attempt}'].size /signal_length)
freq_in_hertz = abs(freq * sample_rate )
print(f"Dominant frequency is {round(freq_in_hertz,2)} Hz")
