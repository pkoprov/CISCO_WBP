import struct
import sys
import time
import micropython
import gc

from machine import I2C, Pin

from mpu6500 import MPU6500


gc.collect()
# addresses
icl_id = 1
sda = Pin(18)
scl = Pin(19)

# create the I2C
i2c = I2C(id=icl_id, scl=scl, sda=sda)

# Scan the bus
m = MPU6500(i2c)

led = Pin(25, Pin.OUT)
led.value(1)
