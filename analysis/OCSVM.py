from matplotlib import gridspec
from matplotlib.pylab import f
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import OneClassSVM
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV,train_test_split
import numpy as np
import os
import sys

sys.path.append(os.getcwd())
from analysis.plot_errors_from_FDA import confusion_matrix, plot_confusion_matrix, plot_errors, metrics
from data.merge_X_all import folders_to_process, START

plt.ion()


def scoring_function(estimator, X, y):
    test_scores = 1 / estimator.score_samples(X)
    threshold = np.percentile(test_scores[y == label], 90)

    # Make predictions based on the threshold
    predictions = np.where(test_scores <= threshold, label, -label)

    # Calculate the F1 score
    f1 = f1_score(y, predictions, pos_label=label)
    # Calculate Sensitivity 
    cm = confusion_matrix(test_scores, test_scores, y, label, threshold)
    sensitivity = cm[0]/(cm[0]+cm[2])

    return sensitivity

param_grid = {
    'kernel': ['rbf', 'poly', 'sigmoid'],
    'gamma': ['scale', 'auto'],
    'nu': np.linspace(0.01, 0.5, 10)}
grid_search = GridSearchCV(OneClassSVM(), param_grid, scoring=scoring_function)


df = pd.read_csv(r"data\Kernels\2023_11_25\Bambu_merged.csv")

labels = df["asset"]
unique_labels = labels.unique()
label = unique_labels[0]
target_data = df[labels == label]
non_target_data = df[labels != label]

train_df, test_df = train_test_split(
    target_data, test_size=0.25, random_state=123)
train_ind = train_df.index

test_df = pd.concat([test_df, non_target_data])
test_ind = test_df.index
y_test = test_df["asset"]

train_df.drop(["asset"], axis=1, inplace=True)
test_df.drop(["asset"], axis=1, inplace=True)


# Perform grid search
grid_search.fit(train_df)
print("Best parameters:", grid_search.best_params_)

model = grid_search.best_estimator_

train_scores = 1/model.score_samples(train_df)
test_scores = 1/model.score_samples(test_df)
test_target_scores = test_scores[y_test==label]
test_non_target_scores = test_scores[y_test!=label]



plt.figure(figsize=[10, 5])
gs = gridspec.GridSpec(1, 2, width_ratios=[7, 2.5])
plt.subplot(gs[0])

plot_errors(labels, unique_labels, label, train_ind, test_ind, y_test, test_scores, train_scores, 'top')

# remove line that has label "threshold"
ax = plt.gca()  # Get current axes
for child in ax.get_children():
    if child.get_label() == 'threshold':
        child.remove()
perc =90
threshold = np.percentile(test_target_scores, perc)
plt.title(f"OneClassSVM performance for {label} with {perc}th perentile threshold")
plt.axhline(threshold, color='red', linestyle = '--', label='threshold')

gs_right = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[1], hspace=0.4)

# Upper plot for confusion matrix
plt.subplot(gs_right[0])
cm = confusion_matrix(test_target_scores, test_scores, y_test, label, threshold)
plot_confusion_matrix(cm)
# Lower plot for metrics
ax_met = plt.subplot(gs_right[1])
met = metrics(cm)  # Calculate metrics
# Display metrics using text
metrics_text = f"Sensitivity: {met[0]:.2f}\nSpecificity: {met[1]:.2f}\nPrecision: {met[2]:.2f}\nF1: {met[3]:.2f}"
ax_met.text(0.5, 0.5, metrics_text, horizontalalignment='center',
            verticalalignment='center', transform=ax_met.transAxes, fontsize=12)
ax_met.axis('off')  # Optionally turn off axis if not needed

plt.tight_layout()



typ = label[:-2] if "UR" not in label else "UR"
folders = folders_to_process(typ)

folders = folders[folders.index(START[typ])+1:]
day = 0
for folder in folders:
    print(folder)
    day += 1
    try:
        verif = pd.read_csv(fr"data\Kernels\{folder}\Bambu_merged.csv")
        train_folder = folders[folders.index(folder)-1]
        train_df = pd.read_csv(fr"data\Kernels\{train_folder}\{label}\{label}_merged_new.csv")
    except FileNotFoundError:
        continue

    train_df.drop(["asset"], axis=1, inplace=True)
    grid_search.fit(train_df)
    model = grid_search.best_estimator_

    verif_labels = verif["asset"]
    verif.drop(["asset"], axis=1, inplace=True)
    verif.columns = verif.columns.astype(float)
    verif_scores = 1/model.score_samples(verif)
    verif_target_scores = verif_scores[verif_labels==label]
    verif_non_target_scores = verif_scores[verif_labels!=label]
    cm = confusion_matrix(verif_target_scores, verif_scores, verif_labels, label, threshold)
    met = metrics(cm)
    print(verif_target_scores)
    days = np.repeat(day, len(verif_target_scores))
    plt.plot(days, verif_target_scores, "o", color="blue")
    plt.plot(days, verif_non_target_scores, "o", color="red")

plt.axhline(threshold, color="black", linestyle="--")
plt.tight_layout()





