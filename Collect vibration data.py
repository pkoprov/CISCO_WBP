import serial, time, struct
import pandas as pd


ser = serial.Serial("/dev/ttyACM0")
dur = input("Set duration as an int: ")+'\n'
for j in range(6):
    input("ready?")

    df=[]
    ser.flushInput()
    ser.write(dur.encode())
    start_time = time.time()
    while (time.time()- start_time) < int(dur):
        
        reading = ser.read(4096)
        df.append(reading)
            
    print("Time is out")
    x=b''.join(df)
    readings = x.split(b'||')
    l = 0
    y=[]
    for i in readings:
        try:
            y.append(struct.unpack('3d',i))
        except:
            l+=1
    print("errors",l)

    timestamp = time.strftime("%Y_%m_%d %H:%M:%S")
    pd.DataFrame(y).to_csv(f'{timestamp}_VF-2-1_with tool.csv')
