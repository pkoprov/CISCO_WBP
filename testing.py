import matplotlib.pyplot as plt
import pandas as pd
import os


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

