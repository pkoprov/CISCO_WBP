import pandas as pd

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn import linear_model


df = pd.read_csv(f"data/Kernels/2023_02_07/Prusa_merged.csv")
labels = pd.unique(df["asset"])
target_df = df.loc[df["asset"] ==labels[7]]
X_train = target_df.iloc[:22, 1:]
y_train = target_df["asset"][:22]
X_test = df.iloc[:, 1:]
y_test = np.full((X_test.shape[0],), -1)
y_test[np.where(df["asset"] == labels[7])[0]] = 1


nu = 1
accuracy = 0
j = 0
for i in range(1, 101):
    model = linear_model.SGDOneClassSVM(random_state= 123,nu=i/100)
    model.fit(X_train)
    y_pred_test = model.predict(X_test)
    conf = confusion_matrix(y_test, y_pred_test)
    if conf[0][1]== 0 and conf[1][0] <= 1:
        accuracy = (conf[0][0]+conf[1][1])/np.sum(conf)
        nu = i/100
        print(f"with nu = {i / 100}, accuracy = {accuracy}")
        if accuracy >= 1:
            j += 1
            print(j)
        else:
            j = 0
        if j >= 5 and accuracy ==1:
            break
    # print(confusion_matrix(y_test, y_pred_test))


model = linear_model.SGDOneClassSVM(random_state= 123,nu=1)
model.fit(X_train)


model.offset_ = 0.9*min(model.score_samples(X_train))
y_pred_test = model.predict(X_test)
confusion_matrix(y_test, y_pred_test)

plt.plot(model.score_samples(X_test), ".")
plt.hlines(model.offset_, 0, len(X_test), color="red")
plt.scatter(np.where(y_test == 1)[0], model.score_samples(X_test)[y_test == 1], color="red")
