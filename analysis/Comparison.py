import os, math
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, ifft
import numpy as np
import pandas as pd


machine1="UR-5e_1"
# create signal
colnames=['TIME', 'X', 'Y', 'Z']
folder1 = f'./{machine1}' # for CNC
for n,file in enumerate(os.listdir(folder1)):
    if '.csv' in file and n==0:
        globals()[f"datafr0"] = pd.read_csv(f'{folder1}/{file}', names=colnames, skiprows=1)

machine2="UR-5e_2"
folder2 = f'./{machine2}' # for CNC
for n,file in enumerate(os.listdir(folder2)):
    if '.csv' in file and n==0:
        globals()[f"datafr1"] = pd.read_csv(f'{folder2}/{file}', names=colnames, skiprows=1)


df_list = [globals()[df] for df in globals() if 'datafr' in df]

signal_length = 12 #[ seconds ]

plt.figure()

i=1

dat=df_list[i]

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
# fig.add_subplot(len(df_list), 2, i * 2 + 1)
plt.plot(globals()[f't{i}'], globals()[f'accel{i}'][0], label=machine1)
plt.xlabel('time [s]')
plt.ylabel(f'{colnames[1]} signal')

shift = 450
plt.plot(globals()[f't{i}'][shift:], globals()[f'accel{i}'][0][:-shift], label=machine2)
plt.legend()

