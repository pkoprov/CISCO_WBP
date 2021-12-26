from machine import I2C, Pin, reset
from mpu9250 import MPU9250
import sys, time, struct

# addresses 
icl_id = 1
sda = Pin(18)
scl = Pin(19)

# create the I2C
i2c = I2C(id=icl_id, scl=scl, sda=sda)

# Scan the bus
m = MPU9250(i2c)
# main loop
while True:
    # read the input
    CMD = sys.stdin.buffer.readline()
    if CMD == None:
        continue
    elif 'exit' in CMD.decode():
        reset()
    else:
        # check if the input is integer
        try:
            time_del = int(CMD)
        except (ValueError):
            print("Incorrect time data type")
            continue
        #write to file
#         with open('buf.csv','w') as file:
        n=0 # count of measurements
            # set the duration of data collection
        start_time = time.time()
        while (time.time()- start_time) < (time_del+2):
            ax, ay, az = m.acceleration
#                 file.write('%sl\n' % m.acceleration)
            n += 1
            sys.stdout.write(struct.pack('3d', ax,ay,az)+'||'.encode())
    #             print('')
    #             print(m.acceleration)
        print('stop',n)

print('end of code')