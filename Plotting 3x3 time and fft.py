# -*- coding: utf-8 -*-
"""
@author: bstarly

"""
import os
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, ifft
import numpy as np
import pandas as pd

# plots vertical lines of specific time period
def draw_dt(move_name):
    dt = np.array(move_dict[move_name]) # pick a move from movement dictionary
    for x in dt:
        plt.vlines(x, plt.ylim()[0]*0.9, plt.ylim()[1]*0.9, color = 'red')

plt.ioff() # turn off the otput of figures
machine="UR-5e_1"
# create signal
colnames=['TIME', 'X', 'Y', 'Z']
# dictionary containing moves and their time period
move_dict = {"X_fwd":[0,3],"Y_fwd":[3,4.8], "X_rvs": [4.8,7.4], "Y_rvs": [7.4,9.1], "X_half-fwd":[9.1,10.8],
            "Z_dwn": [10.8, 11.9], "Z_up": [11.9, 13], "RPM": [13,15]}

# folder = f'./{machine} wTool' # for CNC
folder = f'./{machine}' # for robot
for n,file in enumerate(os.listdir(folder)):
    if '.csv' in file:
        globals()[f"datafr{n}"] = pd.read_csv(f'{folder}/{file}', names=colnames, skiprows=1)

df_tuple = (datafr0, datafr1,datafr2)

signal_length = 16 #[ seconds ]

# create a figure window with 3 rows
rownames = ("1st attempt", "2nd attempt", '3rd attempt')
fig, big_axes = plt.subplots(3,1, sharey=True,figsize=(20,15), dpi=200)
fig.suptitle(machine, fontsize='xx-large')
for n, big_ax in enumerate(big_axes):
    big_ax.set_title(rownames[n], fontsize=16, y=1.05)
    # Turn off axis lines and ticks of the big subplot
    big_ax.tick_params(labelcolor=(1.,1.,1., 0.0), top='off', bottom='off', left='off', right='off')
    # removes the white frame
    big_ax._frameon = False

# create plots of the whole movement set
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

    # draw plot names only at the first row
    if i == 0:
        plt.title('Time domain', y=1.1)
    for move, val in move_dict.items():
        # insert move names
        if i == 0: plt.text((val[0] + val[1]) / 2, plt.ylim()[1] * 1.05, move, ha='center')
        draw_dt(move)

    # plot frequency spectrum
    fig.add_subplot(len(df_tuple), 2, i*2+2)
    plt.plot(globals()[f'freqs{i}'], abs(globals()[f'fft_accel{i}'][0][0: globals()[f'n_freq{i}']]))
    plt.xlabel('frequency [Hz]')
    plt.ylabel('abs(DFT( signal ))')
    plt.xticks(np.arange(min(globals()[f'freqs{i}']), max(globals()[f'freqs{i}']) + 1, 10.0),rotation = 50)
    if i == 0:
        plt.title('Frequency spectrum')

if not os.path.isdir(f"{folder}/plots"):
    os.makedirs(f"{folder}/plots")
plt.savefig(f"{folder}/plots/{machine} full movement.png")


# function to plot time series
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


# function to plot fft
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
    plt.xticks(np.arange(min(freqs0X), max(freqs0X) + 1, 10.0), rotation=45)
    if attempt == 0:
        plt.title('Frequency spectrum')


# function to plot time series and fft for specific movement and sensor axis
def plot_xyz(move_name):
    dt = np.array(move_dict[move_name])
    for axis in 'xyz':

        for i in range(3):
            plot_time(i, axis, dt)
            plot_fft(i,axis, dt)
        name = f"{machine}_{move}_{axis.capitalize()}axis of sensor"
        plt.suptitle(name, fontsize='xx-large')
        # check if the folder "plots" exist
        if not os.path.isdir(f"{folder}/plots"):
            os.makedirs(f"{folder}/plots")
        plt.savefig(f"{folder}/plots/{name}")
        plt.clf()

plt.figure(figsize=(20,15),dpi=200)
# plot all moves on for each of sensor's axis
for move, val in move_dict.items():
    plot_xyz(move)



#
#
# # save plot to disk
# plt.savefig ('VF-2_1 wTool/22_01_07/fft1.png')
# plt.show() #and display plot on screen
#
# arrfreqs = fftfreq(len(fft_accel0X))
# print(arrfreqs.min(), arrfreqs.max())
#
# # Find the peak in the coefficients
# idx = np.argmax(np.abs(fft_accel0X))
# freq = arrfreqs[idx]
# sample_rate = round(globals()[f't{attempt}'].size /signal_length)
# freq_in_hertz = abs(freq * sample_rate )
# print(f"Dominant frequency is {round(freq_in_hertz,2)} Hz")
