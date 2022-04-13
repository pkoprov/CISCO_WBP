import struct
import sys
import time

from machine import I2C, Pin

from mpu6550 import MPU6500

# addresses 
icl_id = 1
sda = Pin(18)
scl = Pin(19)

# create the I2C
i2c = I2C(id=icl_id, scl=scl, sda=sda)

# Scan the bus
m = MPU6500(i2c)

start = time.time()
calibration_data = []
while time.time() - start <= 1:
    calibration_data.append(m.acceleration[1])
mean = sum(calibration_data) / len(calibration_data)
sd = (sum([(i - mean) ** 2 for i in calibration_data]) / len(calibration_data)) ** 0.5

# main loop
while True:
    # read the input
    ax, ay, az = m.acceleration
    if ax[1] > mean + 5 * sd or ax[1] < mean - 5 * sd:
        try:
            sys.stdout.write(struct.pack('3d', ax, ay, az) + '\n'.encode())
        except:
            print("Not working!")
            time.sleep(1)
