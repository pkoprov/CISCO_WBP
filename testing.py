import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.fft import rfft, rfftfreq,irfft
import os


VF_2_1_nt_1 =  pd.read_csv(r'C:\Users\pkoprov\Desktop\test0_no tool.csv')
VF_2_1_nt_2 =  pd.read_csv(r'C:\Users\pkoprov\Desktop\test1_no tool.csv')
VF_2_2_nt_1 =  pd.read_csv(r'C:\Users\pkoprov\Desktop\test02_no tool.csv')
VF_2_2_nt_2 =  pd.read_csv(r'C:\Users\pkoprov\Desktop\test12_no tool.csv')
VF_2_1_bt_1 =  pd.read_csv(r'C:\Users\pkoprov\Desktop\test0_With big tool.csv')
VF_2_1_bt_2 =  pd.read_csv(r'C:\Users\pkoprov\Desktop\test1_With big tool.csv')
VF_2_2_bt_1 =  pd.read_csv(r'C:\Users\pkoprov\Desktop\test02_With big tool.csv')
VF_2_2_bt_2 =  pd.read_csv(r'C:\Users\pkoprov\Desktop\test12_With big tool.csv')

data = [[VF_2_1_nt_1, VF_2_2_nt_1,VF_2_1_nt_2,VF_2_2_nt_2],[VF_2_1_bt_1,VF_2_2_bt_1,VF_2_1_bt_2,VF_2_2_bt_2]]
data_lab = [['VF-2_1 with no tool.1','VF-2_2 with no tool.1','VF-2_1 with no tool.2','VF-2_2 with no tool.2'],
            ['VF-2_1 with Big tool.1','VF-2_2 with Big tool.1','VF-2_1 with Big tool.2','VF-2_2 with Big tool.2']]
fig, ax = plt.subplots(4,2)
plt.subplots_adjust(hspace=1)
for i, d in enumerate(data):
    for j, dat in enumerate(d):
        ax[j,i].plot(range(dat.shape[0]),dat.iloc[:,1])
        ax[j,i].set_title(data_lab[i][j])

test1_1 = pd.read_csv(r'C:\Users\pkoprov\Desktop\test1_no tool.csv')
test1_0 = pd.read_csv(r'C:\Users\pkoprov\Desktop\test1_With big tool.csv')
test1_2 = pd.read_csv(r'C:\Users\pkoprov\Desktop\test12_With big tool.csv')


ax[0].plot(range(test1_0.shape[0]),test1_0.iloc[:,1])
ax[1].plot(range(900,test1_1.shape[0]+900),test1_1.iloc[:,1])
ax[2].plot(range(test1_2.shape[0]),test1_2.iloc[:,1])
plt.figure()
plt.plot(test2.iloc[:,0],test2.iloc[:,1])
plt.plot(test3.iloc[:,0],test3.iloc[:,1])
plt.plot(test4.iloc[:,0],test4.iloc[:,1])
plt.plot(test5.iloc[:,0],test5.iloc[:,1])
plt.plot(test6.iloc[:,0],test6.iloc[:,1])


SAMPLE_RATE = test1.shape[0]/20  # Hertz
DURATION = 20  # Seconds
N = SAMPLE_RATE * DURATION
normalized = np.float_((test1['0']/test1['0'].max())*9.8)

fig, ax = plt.subplots(2,1)
plt.subplots_adjust(hspace=1)
ax[0].set_title('Time series')
ax[0].plot(np.arange(0,20, 20/N), normalized)
ax[0].set_xlabel('Time')
ax[0].set_ylabel('Amplitude')


# from scipy.io.wavfile import write
#
# # Remember SAMPLE_RATE = 44100 Hz is our playback rate
# write("mysinewave.wav", SAMPLE_RATE, normalized_tone)


yf = rfft(normalized)
xf = rfftfreq(int(N), 1/SAMPLE_RATE)
ax[1].set_title('FFT')
ax[1].plot(xf, np.abs(yf))
ax[1].set_ylim(0,1000)
ax[1].set_xlabel('Frequencies')
ax[1].set_ylabel('FFT Amplitude')

# change frequency cutoff and plot the results
fig, ax = plt.subplots(5,1)
plt.subplots_adjust(hspace=1)
for i in range(5):
    sig_fft_filtered = yf.copy()
    cut_off = (i+1)*50
    sig_fft_filtered[np.abs(xf) > cut_off] = 0
    filtered = irfft(sig_fft_filtered)
    ax[i].plot(np.arange(0,20, 20/N), filtered)
    ax[i].set_title(f'Cut off > {cut_off} ')


plt.figure(figsize = (12, 6))
plt.subplot(121)
plt.stem(xf, np.abs(yf), 'b', \
         markerfmt=" ", basefmt="-b")
plt.title('Before filtering')
plt.xlim(-10, xf.max())
# plt.ylim(1, 800)
plt.xlabel('Frequency (Hz)')
plt.ylabel('FFT Amplitude')
plt.subplot(122)
plt.stem(xf, np.abs(sig_fft_filtered), 'b', \
         markerfmt=" ", basefmt="-b")
plt.title('After filtering')
plt.xlim(1, cut_off)
plt.ylim(0, 800)
plt.xlabel('Frequency (Hz)')
plt.ylabel('FFT Amplitude')
plt.tight_layout()
plt.show()

# get rid of small amplitudes AKA noise
x = normalized.copy()
argmin = x[:2516].min()
argmax = x[:2516].max()
x[(x>=(argmin)) & (x<=(argmax))] = np.mean([argmax, argmin])
fig, ax = plt.subplots(2,1)
plt.subplots_adjust(hspace=1)
for i in range(2):
    if i == 0:
        ax[i].set_title('Time series of full data')
        ax[i].plot(np.arange(0,20, 20/N), normalized)
    else:
        ax[i].set_title('Time series without ambient noise')
        ax[i].plot(np.arange(0, 20, 20 / N), x)
    ax[i].set_xlabel('Time')
    ax[i].set_ylabel('Amplitude')


fig, ax = plt.subplots(6, 1)
fig.suptitle("UR-5e_1")
plt.subplots_adjust(hspace=0)
i=0
for file in os.listdir(r'C:\Users\pkoprov\Desktop\UR-5e_1'):
    try:
        test = pd.read_csv(fr'C:\Users\pkoprov\Desktop\UR-5e_1\{file}')
        ax[i].plot(test.iloc[:,0],test.iloc[:,1])
        ax[i].set_title(file[:19].replace('_','/').replace('-',':'))
        i+=1
    except:
        pass

