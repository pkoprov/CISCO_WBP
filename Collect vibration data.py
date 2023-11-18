import time
import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from mpu9250_jmdev.registers import *
from mpu9250_jmdev.mpu_9250 import MPU9250

# Constants
MPU_ADDRESS = MPU9050_ADDRESS_68
GFS_SETTING = GFS_1000
AFS_SETTING = AFS_4G
MFS_SETTING = AK8963_BIT_16
MODE_SETTING = AK8963_MODE_C100HZ
CALIBRATION_DURATION = 2
THRESHOLD_MULTIPLIER = 4
STOP_COUNT = 500
BUS = int(input("Which printer?\n1. 1 material\n3. Multimaterial\n>>")) if os.path.exists(f'/dev/i2c-3') else 1

def initialize_sensor():
    mpu = MPU9250(
        address_mpu_master=MPU_ADDRESS,
        address_mpu_slave=None, 
        bus=BUS,
        gfs=GFS_SETTING, 
        afs=AFS_SETTING, 
        mfs=MFS_SETTING, 
        mode=MODE_SETTING)
    mpu.configure()
    return mpu

def calibrate_sensor(mpu, duration=CALIBRATION_DURATION):
    calibration_data = []
    start = time.time()
    while time.time() - start < duration:
        calibration_data.append(mpu.readAccelerometerMaster()[0])
    mean = sum(calibration_data) / len(calibration_data)
    sd = (sum([(i - mean) ** 2 for i in calibration_data]) / len(calibration_data)) ** 0.5
    return mean, sd

def measurement_steady(mpu, mean, sd):
    start_flag = False
    data = []
    n = 0

    while True:
        ax, ay, az = mpu.readAccelerometerMaster()
        if not start_flag:
            if abs(ax - mean) > THRESHOLD_MULTIPLIER * sd:
                start_flag = True
                print("Start reading")
                start = time.time()
            else:
                continue

        timestamp = time.time()
        data.append([timestamp - start, ax, ay, az])
        if abs(ax - mean) > THRESHOLD_MULTIPLIER * sd:
            n = 0
        else:
            n += 1
            if n == STOP_COUNT:
                print("Stop reading")
                break
    return data

def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

def collect_samples(mpu, mean, sd, n, duration, asset_name):
    folder = os.path.join('data', 'Kernels', datetime.now().date().strftime("%Y_%m_%d"))
    create_directory(folder)
    asset_folder = os.path.join(folder, asset_name)
    create_directory(asset_folder)

    i = 1
    while i <= n:
        print(f"Collecting sample {i}")
        sample_name = f"{asset_name}_{round(time.time())}"
        data = measurement_steady(mpu, mean, sd)
        df = pd.DataFrame(data[:-STOP_COUNT])
        if len(df) < STOP_COUNT * duration:
            continue

        df.to_csv(os.path.join(asset_folder, f"{sample_name}.csv"), index=False)
        fig = plt.figure(sample_name)
        fig.set_size_inches(18.5, 10)
        plt.plot(df.iloc[:, 1:])
        plt.savefig(os.path.join(asset_folder, f"{sample_name}.png"))
        plt.close()
        print(f"Created {sample_name}")
        i += 1

def main():
    mpu = initialize_sensor()
    print("Start Calibrating")
    mean, sd = calibrate_sensor(mpu)
    print("Calibration is done")

    n = int(input("How many times? "))
    duration = float(input("Approximate duration of move in sec? "))
    asset_name = input("Name of machine? ")

    collect_samples(mpu, mean, sd, n, duration, asset_name)
    print("Done Collecting samples")

if __name__ == "__main__":
    main()

