import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


ur_data = pd.read_csv(r"C:\Users\pkoprov\PycharmProjects\Vibration_Patterns\data\Kernels\UR\UR_x.csv")
ur_labels = pd.unique(ur_data["0"])

target = ur_labels[0]

target_df = ur_data[ur_data["0"] == target].iloc[:, 1:].to_numpy()
scaled_df = target_df - np.mean(target_df, axis=0)

plt.plot(scaled_df[6])



train_df, test_df = train_test_split(target_df, test_size=0.2, random_state=42)


