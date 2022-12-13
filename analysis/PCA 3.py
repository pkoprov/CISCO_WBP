import numpy as np
import pandas as pd
import random
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


# train_data = pd.read_csv(r"C:\Users\pkoprov\PycharmProjects\Vibration_Patterns\data\Kernels\5_7_2022\df_UR5_all_train.csv.csv", low_memory=False).iloc[:, 1:]
#
#
# test_data = pd.read_csv(r"C:\Users\pkoprov\PycharmProjects\Vibration_Patterns\data\Kernels\5_7_2022\df_UR5_all_test.csv.csv", low_memory=False).iloc[:, 1:]

df = pd.read_csv(r"C:\Users\pkoprov\PycharmProjects\Vibration_Patterns\data\Kernels\2022_11_09\VF-2_all.csv", low_memory=False)
target = "VF-2_2"

target_df = df[df["0"] == target]

# ind = random.sample(target_df.index.to_list(), 22)
ind = random.sample(target_df.index.to_list(), 22)
train_target_df = target_df.loc[ind]
test_target_df = target_df.drop(ind)

false_df = df.drop(target_df.index)
ind = random.sample(false_df.index.to_list(), train_target_df.shape[0])
train_false_df = false_df.loc[ind]
train_false_df["0"] = "false"
test_false_df = false_df.drop(ind)
test_false_df["0"] = "false"

train_data = pd.concat([train_target_df, train_false_df])
test_data = pd.concat([test_target_df, test_false_df])

# U, S, V = np.linalg.svd(train_data.iloc[:, 1:].to_numpy(), full_matrices=False)

test_data["0"].loc[test_data["0"] == target] = 1
test_data["0"].loc[test_data["0"] == "false"] = -1
test_data["0"] = test_data["0"].astype(int)



############### Isolation Forest #####################
clf = IsolationForest(random_state=0).fit(train_data.iloc[:22, 1:])
score = clf.score_samples(test_data.iloc[:, 1:])
clf.offset_ = np.mean(clf.score_samples(test_data.iloc[:, 1:])[:8])
confusion_matrix(test_data.iloc[:, 0], clf.predict(test_data.iloc[:, 1:]))





df["0"].loc[df["0"] != target] = -1
df["0"].loc[df["0"] == target] = 1
df["0"] = df["0"].astype(int)

confusion_matrix(df.iloc[:, 0], clf.predict(df.iloc[:, 1:]))


############### OCSVM #####################
from sklearn.svm import OneClassSVM
clf = OneClassSVM(gamma='auto').fit(train_data.iloc[:22, 1:])
score = clf.score_samples(test_data.iloc[:, 1:])
decision = np.mean(clf.score_samples(test_data.iloc[:, 1:])[:8])
pred = []
for i in clf.score_samples(test_data.iloc[:, 1:]):
    if i < decision:
        pred.append(-1)
    else:
        pred.append(1)
confusion_matrix(test_data.iloc[:, 0], pred)