import os

import pandas as pd
from sklearn import metrics, svm
from sklearn.model_selection import train_test_split

# read data and create dataframes
length = 3100

coord_list = ['all', 'x', 'y', 'z']
# create global variables to store x,y,z and xyz data
for i in range(4):
    globals()[f'df_UR5_{coord_list[i]}'] = pd.DataFrame()

home = "data/Kernels/5_7_2022"
for folder in os.listdir(home):
    # if "_ex" in folder:
    if os.path.isdir(f"{home}/{folder}"):
        for file in os.listdir(f"{home}/{folder}"):
            if '.csv' in file:
                df = pd.read_csv(f"{home}/{folder}/{file}")
                type = pd.Series(file[:7])
                X = df.iloc[:length, 1]
                Y = df.iloc[:length, 2]
                Z = df.iloc[:length, 3]
                all_coord_df = pd.concat([X, Y, Z, type], ignore_index=True)
                x_coord_df = pd.concat([X, type], ignore_index=True)
                y_coord_df = pd.concat([Y, type], ignore_index=True)
                z_coord_df = pd.concat([Z, type], ignore_index=True)
                df_UR5_all = pd.concat([df_UR5_all, all_coord_df], axis=1, ignore_index=True)
                df_UR5_x = pd.concat([df_UR5_x, x_coord_df], axis=1, ignore_index=True)
                df_UR5_y = pd.concat([df_UR5_y, y_coord_df], axis=1, ignore_index=True)
                df_UR5_z = pd.concat([df_UR5_z, z_coord_df], axis=1, ignore_index=True)
##################################################################################################
# ____________________________________________ SVM ____________________________________________
##################################################################################################
data = df_UR5_all.transpose().iloc[:, :-1].to_numpy()
targets = df_UR5_all.transpose().iloc[:, -1].to_numpy()
# targets[targets == "UR-5e-1"] = 1
targets[targets != "UR-5e-4"] = "Not correct"

X_train, X_test, y_train, y_test = train_test_split(data, targets, test_size=0.3)  # 70% training and 30% test

clf = svm.SVC(kernel='linear')  # Linear Kernel

# Train the model using the training sets
clf.fit(X_train, y_train)

# Predict the response for test dataset
y_pred = clf.predict(X_test)

print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
metrics.confusion_matrix(y_test, y_pred)
metrics.ConfusionMatrixDisplay.from_predictions(y_test, y_pred)

##################################################################################################
