from cgi import test
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import OneClassSVM
from sklearn.model_selection import train_test_split

plt.ion()

df = pd.read_csv(r"data\Kernels\2023_11_25\Bambu_merged.csv")

label = "Bambu_S"
target_data = df[df["asset"] == label]
non_target_data = df[df["asset"] != label]

train_df, test_df = train_test_split(target_data, test_size=0.2, random_state=123)
train_ind = train_df.index

test_df = pd.concat([test_df, non_target_data])
test_ind = test_df.index
y_test = test_df["asset"]

train_df.drop(["asset"], axis=1, inplace=True)
test_df.drop(["asset"], axis=1, inplace=True)

model = OneClassSVM(kernel="rbf", nu=0.1)
model.fit(train_df)
train_scores = model.score_samples(train_df)
test_scores = model.score_samples(test_df)

# Plotting training scores with blue circle markers
plt.plot(train_ind, train_scores, 'o', color='blue',
            fillstyle='none', label='train')
# Plotting test scores for the correct label with blue circle markers
plt.plot(test_ind[y_test == label], test_scores[y_test ==
            label], 'o', color='blue', label='test target')
# Plotting test scores for the incorrect label with red circle markers
plt.plot(test_ind[y_test != label], test_scores[y_test !=
            label], 'o', color='red', label='test other')
