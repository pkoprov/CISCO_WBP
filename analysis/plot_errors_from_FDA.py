# Standard library imports
import os
import pickle
from multiprocessing import Pool
import warnings
import itertools
import threading
import sys

# Third-party imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.model_selection import train_test_split
from skfda.preprocessing.dim_reduction import FPCA
from skfda.misc.metrics import l2_distance, l2_norm

# Local imports
sys.path.append(os.getcwd())
from analysis.FDA import Sample


# asset = "1"
# label = "VF-2_1"
# key = 'top'
# plt.ion()


# Constants
ASSET_CHOICES = {'1': 'VF-2', '2': 'UR', '3': 'Prusa', '4': 'Bambu'}
FIGURES_DIR = "analysis/figures"
DATA_FILES = {"VF-2": r"data\Kernels\2023_11_18\VF-2_merged.csv", "UR": r"data\Kernels\2023_11_21\UR_merged.csv",
              "Prusa": r"data\Kernels\PRUSA\Prusa_merged.csv", "Bambu": r"data\Kernels\2023_11_25\Bambu_merged.csv"}


# Suppress specific warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def load_data(asset=None):
    """Loads and preprocesses the data for the given asset."""

    data = pd.read_csv(DATA_FILES[asset])
    sample = Sample(data)

    return sample


def save_model(model, filename):
    """Saves the model to a file."""
    with open(filename, 'wb') as f:
        pickle.dump(model, f)


def load_model(filename):
    """Loads a model from a file."""
    with open(filename, 'rb') as f:
        return pickle.load(f)


def error_threshold(train_scores):
    return np.quantile(train_scores, 0.95)


def softmax(x, train):
    return np.exp(x) / np.sum(np.exp(train), axis=0)


def plot_errors(labels, unique_labels, label, train_ind, test_ind, y_test, test_scores, train_scores, key):

    # Plotting training scores with blue circle markers
    plt.plot(train_ind, train_scores, 'o', color='blue',
             fillstyle='none', label='train')
    # Plotting test scores for the correct label with blue circle markers
    plt.plot(test_ind[y_test == label], test_scores[y_test ==
             label], 'o', color='blue', label='test target')
    # Plotting test scores for the incorrect label with red circle markers
    plt.plot(test_ind[y_test != label], test_scores[y_test !=
             label], 'o', color='red', label='test other')

    # Calculating the error threshold based on the training scores
    err_thresh = error_threshold(train_scores)

    # Drawing a horizontal line representing the error threshold
    plt.hlines(err_thresh, 0, len(labels), linestyle='--',
               color='red', label='threshold')
    # Setting the title of the plot indicating the label and the key used
    plt.title(f"Errors for {label} using {key} curves")
    # Displaying the legend of the plot
    plt.legend(loc='best')

    # Drawing vertical dashed lines to indicate label changes
    vlines = [(labels == label).idxmax() -
              0.5 for label in unique_labels] + [test_ind.max()+0.5]
    # Calculate midpoints
    midpoints = [(vlines[i] + vlines[i + 1]) /
                 2 for i in range(len(vlines) - 1)]
    y_pos = plt.gca().get_ylim()[1]  # Slightly below the top edge

    for mid, label in zip(midpoints, unique_labels):
        plt.text(mid, y_pos, label, ha='center', va='bottom')
    plt.vlines(vlines[1:-1], plt.gca().get_ylim()[0], plt.gca().get_ylim()[
               1], color='black', linestyle='--')


def l2_errors_threaded(results_dict, fd_dict, train_ind, test_ind, key, target_idx, label):
    test_errors, train_errors = l2_errors(
        fd_dict, train_ind, test_ind, key, target_idx, label)
    # Save results in dictionary
    results_dict[key] = (test_errors, train_errors)


def l2_errors(fd_dict, train_ind, test_ind, key, target_idx, label):
    # Extract training and testing data for the given 'key' from feature dictionary
    train = fd_dict[key][train_ind]
    test = fd_dict[key][test_ind]
    typ = label[:-2] if "UR" not in label else "UR"
    # Check if a model for the given label and key has already been saved to avoid re-fitting
    model_name = os.path.join(os.path.dirname(DATA_FILES[typ]),label,f"{label}_{key}_fpca.pkl")
    if not os.path.exists(model_name):
        # If the model doesn't exist, fit a new FPCA (Functional Principal Component Analysis) model on target index data
        print(f"Fitting FPCA to train {label} {key} data")
        fpca_clean = FPCA(n_components=1)
        fpca_clean.fit(fd_dict[key][target_idx])
    else:
        # If the model exists, load the FPCA model from the saved file
        fpca_clean = load_model(model_name)["model"]

    # Perform FPCA transformation and inverse transformation to get the reconstructed training set
    train_set_hat = fpca_clean.inverse_transform(fpca_clean.transform(train))

    # Calculate the L2 distance between the original and reconstructed training set,
    # normalized by the L2 norm of the original training set
    train_errors = l2_distance(train_set_hat, train) / l2_norm(train)

    # Perform FPCA transformation and inverse transformation to get the reconstructed testing set
    test_set_hat = fpca_clean.inverse_transform(fpca_clean.transform(test))
    # Calculate the L2 distance between the original and reconstructed testing set,
    # normalized by the L2 norm of the original testing set
    test_errors = l2_distance(test_set_hat, test) / l2_norm(test)

    # Return the normalized test errors, train errors, and the fitted or loaded FPCA model
    return test_errors, train_errors


def main(label, fd_dict, labels, unique_labels, indices):
    target_idx = indices[labels == label]
    train_ind, _, _, _ = train_test_split(
        target_idx, target_idx, test_size=0.2, random_state=123)
    test_ind = indices.difference(train_ind)
    y_test = labels.loc[test_ind].values

    threads = []
    results_dict = {}  # Use a dictionary to enforce order based on key

    for key in ['top', 'bottom']:
        t = threading.Thread(target=l2_errors_threaded, args=(
            results_dict, fd_dict, train_ind, test_ind, key, target_idx, label))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()
    both = {'train': [], 'test': []}
    plt.figure(figsize=[10, 5])
    # Process results in a specific order
    for n, key in enumerate(['top', 'bottom']):
        test_scores, train_scores = results_dict[key]

        plt.subplot(2, 1, n + 1)
        plot_errors(labels, unique_labels, label, train_ind,
                    test_ind, y_test, test_scores, train_scores, key)

        both['train'].append(train_scores)
        both['test'].append(test_scores)
    plt.savefig(f"{FIGURES_DIR}\{label}.png")

    train_scores = np.mean(both['train'], axis=0)
    test_scores = np.mean(both['test'], axis=0)

    plt.figure(figsize=[10, 5])
    gs = gridspec.GridSpec(1, 2, width_ratios=[7, 2.5])
    plt.subplot(gs[0])

    plot_errors(labels, unique_labels, label, train_ind, test_ind,
                y_test, test_scores, train_scores, 'both')

    # Create a nested GridSpec for the right part
    gs_right = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=gs[1], hspace=0.4)

    # Upper plot for confusion matrix
    plt.subplot(gs_right[0])
    cm = confusion_matrix(train_scores, test_scores, y_test, label)
    plot_confusion_matrix(cm)  # Your function to plot confusion matrix

    # Lower plot for metrics
    ax_met = plt.subplot(gs_right[1])
    met = metrics(cm)  # Calculate metrics
    # Display metrics using text
    metrics_text = f"Sensitivity: {met[0]:.2f}\nSpecificity: {met[1]:.2f}\nPrecision: {met[2]:.2f}\nF1: {met[3]:.2f}"
    ax_met.text(0.5, 0.5, metrics_text, horizontalalignment='center',
                verticalalignment='center', transform=ax_met.transAxes, fontsize=12)
    ax_met.axis('off')  # Optionally turn off axis if not needed

    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}\{label}_both_FPCA1.png")


def confusion_matrix(train_scores, test_scores, y_test, label, err_thresh=None):

    err_thresh = error_threshold(train_scores) if err_thresh is None else err_thresh

    act_pos = np.where(y_test != label)[0]
    act_neg = np.where(y_test == label)[0]
    pred_pos = np.where(test_scores > err_thresh)[0]
    pred_neg = np.where(test_scores <= err_thresh)[0]

    TP = len(np.intersect1d(pred_pos, act_pos))
    TN = len(np.intersect1d(pred_neg, act_neg))
    FP = len(np.intersect1d(pred_pos, act_neg))
    FN = len(np.intersect1d(pred_neg, act_pos))

    confusion_matrix = np.array([[TP, FN], [FP, TN]])
    return confusion_matrix


def metrics(confusion_matrix):
    TP = confusion_matrix[0][0]
    FN = confusion_matrix[0][1]
    FP = confusion_matrix[1][0]
    TN = confusion_matrix[1][1]

    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    precision = TP / (TP + FP)
    F1 = 2 * (precision * sensitivity) / (sensitivity + precision)
    return sensitivity, specificity, precision, F1


def wrapper_plot_basis(label, fd_dict, labels, unique_labels, indices):
    return main(label, fd_dict, labels, unique_labels, indices)


def plot_confusion_matrix(cm):

    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title("Confusion matrix")

    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks([0, 1], ['Positive', 'Negative'])
    plt.yticks([0, 1], ['Positive', 'Negative'])

    thresh = cm.max() / 2

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center", fontsize=12,
                 color="white" if cm[i, j] > thresh else "black")

    plt.vlines(0.5, 1.5, 0.5, color='black')
    plt.hlines(0.5, -0.5, 1.5, color='black')


if __name__ == '__main__':
    asset = input("""
    Which asset do you want to plot?
    Options:
    1. VF-2
    2. UR
    3. Prusa
    4. Bambu
    > """)

    asset = ASSET_CHOICES.get(asset.upper(), None)
    if not asset:
        raise ValueError("Invalid asset choice")

    sample = load_data(asset)
    labels = sample.labels
    unique_labels = labels.unique()
    indices = sample.index
    fd_dict = sample.FData()

    with Pool(len(unique_labels)) as p:
        args = [(label, fd_dict, labels, unique_labels, indices)
                for label in unique_labels]
        p.starmap(wrapper_plot_basis, args)
