import matplotlib.pyplot as plt
import pandas as pd
import serial

ser = serial.Serial("/dev/ttyACM1", baudrate=115200, timeout=0.1)

n = 6
name = "Kernel1"
for i in range(n):
    try:
        ser.flushInput()
        while True:
            dat = ser.readall().decode().split('\r\n')
            if dat[0] and len(dat) > 1000:
                dat = [list(map(float, i.split(' '))) for i in dat[:-100]]
                #             print(len(dat))
                df = pd.DataFrame(dat)
                df.to_csv(f"./Kernels/{name}{i}.csv")
                break

        plt.plot(range(df.shape[0]), df.iloc[:, 0])

        print(f"Created {name}{i}")
    except:
        n = 6
plt.savefig(f"./Kernels/{name}{i}.png")

