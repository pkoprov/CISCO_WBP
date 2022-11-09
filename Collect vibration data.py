import time
from mpu9250_jmdev.registers import *
from mpu9250_jmdev.mpu_9250 import MPU9250
import matplotlib.pyplot as plt
import pandas as pd
import os
from datetime import datetime

mpu = MPU9250(
    
    address_mpu_master=MPU9050_ADDRESS_68, # In 0x68 Address
    address_mpu_slave=None, 
    bus=1,
    gfs=GFS_1000, 
    afs=AFS_8G, 
    mfs=AK8963_BIT_16, 
    mode=AK8963_MODE_C100HZ)

mpu.configure() # Apply the settings to the registers.

print("Start Calibrating")
# collect an ambient vibration
calibration_data = []
start = time.time()
while time.time() - start < 2:
    calibration_data.append(mpu.readAccelerometerMaster()[0])

# calculate the mean and SD of ambient vibration
mean = sum(calibration_data) / len(calibration_data)
sd = (sum([(i - mean) ** 2 for i in calibration_data]) / len(calibration_data)) ** 0.5

print("Calibration is done")

def measurement_steady():
    start_flag = False
    data = []
        
    while True:
        # read the input
        ax, ay, az = mpu.readAccelerometerMaster()
        # check if the values are exceeding the threshold
        if not start_flag:
            if ax > mean + 4 *sd or ax < mean - 4 *sd:
                start_flag = True
                print("Start reading")
            else:
                continue

        data.append([ax, ay, az])
        if not (ax > mean + 4*sd or ax < mean - 4*sd):
            n += 1
            if n == 500:
                print("Stop reading")
                break
        else:
            n = 0
    return (data)


n = int(input("How many times? "))
i = 1
duration = int(input("Approximate duration of move in sec? "))

asset_name = input("Name of machine? ")
try:
    os.mkdir(f'./data/Kernels/{datetime.now().date().strftime("%Y_%m_%d")}')
except FileExistsError:
    pass
try:
    os.mkdir(f'./data/Kernels/{datetime.now().date().strftime("%Y_%m_%d")}/{asset_name}')
except FileExistsError:
    pass
folder = f'./data/Kernels/{datetime.now().date().strftime("%Y_%m_%d")}/{asset_name}'

while i < n+1:
    print(f"Collecting sample {i}")
    sample_name = f"{asset_name}_{round(time.time())}"
    data = measurement_steady()
    df = pd.DataFrame(data[:-100])
    if len(df) < 500*duration:
        continue
    
    df.to_csv(f"{folder}/{sample_name}.csv")
    fig = plt.figure(sample_name)
    fig.set_size_inches(18.5,10)
    plt.plot(data)
    plt.plot(data)
    plt.savefig(f"{folder}/{sample_name}.png")
    plt.close()
    print(f"Created {sample_name}")
    i += 1

print("Done Collecting samples")