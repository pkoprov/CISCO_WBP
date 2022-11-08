import serial
import pandas as pd
import csv
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

ser = serial.Serial("/dev/ttyACM0", baudrate = 115200, timeout = 0.1)

n = 10
name = "Signal"
#while True:
for i in range(n):
    try:
        ser.flushInput()
        while True:
            dat = ser.readall().decode().split('\r\n')
            if dat[0] and len(dat) > 1000:
                dat = [list(map(float, i.split(' '))) for i in dat[:-100]]
    #             print(len(dat))
                df = pd.DataFrame(dat)
                #df.to_csv(f"./Kernels/{name}{i}.csv")
                break


        plt.plot(range(df.shape[0]), df.iloc[:,0])
        
        kernel_df = pd.read_csv(r"/home/pi/Desktop/Vibration_Patterns/Kernels/Kernel12_Backtrace_Motion.csv")
        if len(df) in range(len(kernel_df)-300, len(kernel_df)+300):
            try:
                if len(df) >= len(kernel_df):
                    list1 = df[0][0:len(kernel_df)]
                    list2 = kernel_df['0']
                    pc = pearsonr(list1, list2)[0]
                    if pc > 0.06:
                        print(f"pearson corr = {pc}. Authentication Done")
                        df.to_csv(f"./Kernels/{name}_{time.strftime('%Y_%m_%d %H:%M:%S', time.localtime(time.time()))}.csv")
                        print(f"Saved {name} at {time.strftime('%Y_%m_%d %H:%M:%S', time.localtime(time.time()))}\n")
                elif len(df) < len(kernel_df):
                    list1 = kernel_df['0'][0:len(df)]
                    list2 = df['0']
                    pc = pearsonr(list1, list2)[0]
                    if pc > 0.06:
                        print(f"pearson corr = {pc}. Authentication Done")
                        df.to_csv(f"./Kernels/{name}_{time.strftime('%Y_%m_%d %H:%M:%S', time.localtime(time.time()))}.csv")
                        print(f"Saved {name} at {time.strftime('%Y_%m_%d %H:%M:%S', time.localtime(time.time()))}\n")
            except:
                pass
                
        else:        
            print(f"Discarded unwanted data: {name} at {time.strftime('%Y_%m_%d %H:%M:%S', time.localtime(time.time()))}\n")
    except:
        n = 6
plt.savefig(f"./Kernels/{name}{i}.png")

