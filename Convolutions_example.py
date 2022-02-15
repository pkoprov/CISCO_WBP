import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train = pd.read_csv("train.csv", header=None)
train = np.array(train)
s = np.zeros([4,4])
s1 = np.zeros([4,4])
plt.figure()
for k in range(16):
    for j in range(4):
        for i in range(4):
            a=np.convolve(np.flip(train[k]),train[4*j+i])
            s[j,i]=max(a)
    S = np.max(s)
    s1 = s/S

    plt.subplot(4,4,k+1)
    plt.imshow(s1,extent=[0, 1, 0, 1])
    plt.title(f'point of impact {k}')

u1_16 = np.array(pd.read_csv("u1.csv", header=None))
u2_16 = np.array(pd.read_csv("u2.csv", header=None))

u1 = np.zeros([4,4])
u2 = np.zeros([4,4])
for j in range(4):
    for i in range(4):
        a1 = np.convolve(np.flip(u1_16[0]), train[4*j+i])
        a2 = np.convolve(np.flip(u2_16[0]), train[4*j+i])
        u1[j, i] = max(a1)
        u2[j, i] = max(a2)

mu1 = np.max(u1)
mu2 = np.max(u2)
U1 = u1 / mu1
U2 = u2 / mu2

plt.figure()
plt.subplot(1,2,1)
plt.imshow(u1,extent=[0, 1, 0, 1])
plt.title(f'Mistery point1')
plt.subplot(1,2,2)
plt.imshow(u2,extent=[0, 1, 0, 1])
plt.title(f'Mistery point2')