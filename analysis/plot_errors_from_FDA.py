import os
import pickle
import threading
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing import Pool
import matplotlib.gridspec as gridspec 
from skfda.misc.metrics import l2_distance, l2_norm
from sklearn.model_selection import train_test_split
from FDA import Sample


warnings.simplefilter(action='ignore', category=FutureWarning)

def softmax(x, train):
    return np.exp(x) / np.sum(np.exp(train), axis=0)


def plot_errors(labels, unique_labels, label, train_ind, test_ind, y_test, test_errors, train_errors, key):
    train_scores = softmax(train_errors, train_errors)
    test_scores = softmax(test_errors, train_errors)
    plt.plot(train_ind, train_scores, 'o', color='blue', fillstyle='none', label='train')
    plt.plot(test_ind[y_test == label], test_scores[y_test == label], 'o', color='blue', label='test target')
    plt.plot(test_ind[y_test != label], test_scores[y_test != label], 'o', color='red', label='test other')

    err_thresh = np.percentile(train_scores, 95)
    plt.hlines(err_thresh, 0, len(labels), linestyle='--', color='red', label='threshold')
    plt.title(f"Errors for {label} using {key} curves")
    plt.legend()
    plt.vlines([(labels == label).idxmax() for label in unique_labels],
               plt.gca().get_ylim()[0], plt.gca().get_ylim()[1], color='black',
               linestyle='--', label='label change')

    for lbl in unique_labels:
        plt.text((labels == lbl).idxmax()+10, plt.gca().get_ylim()[1]*0.99, lbl)


def l2_errors_threaded(results_dict, fd_dict, train_ind, test_ind, key):
    test_errors, train_errors = l2_errors(fd_dict, train_ind, test_ind, key)
    results_dict[key] = (test_errors, train_errors)  # Save results in dictionary


def l2_errors(fd_dict, train_ind, test_ind, key):
    train = fd_dict[key][train_ind]
    test = fd_dict[key][test_ind]
    # target_curve_mean = train.mean()
    # tcm = target_curve_mean.data_matrix.reshape(-1)
    # knots = find_extreme_grid(tcm, key=key)

    # basis = BSpline(knots=knots)
    # print(f"Fitting basis to {key} train data")
    # train_basis = train.to_basis(basis)
    # print(f"Fitting basis to {key} test data")
    # test_basis = test.to_basis(basis)
    print(f"Calculating L2 distance for {key} test data")
    test_errors = l2_distance(test, train.mean()) / l2_norm(train.mean())
    print(f"Calculating L2 distance for {key} train data")
    train_errors = l2_distance(train, train.mean()) / l2_norm(train.mean())
    return test_errors, train_errors


def main(label, fd_dict, labels, unique_labels, indices):
    target_idx = indices[labels == label]
    train_ind,_,_,_ = train_test_split(target_idx, target_idx, test_size=0.3, random_state=123)
    test_ind = indices.difference(train_ind)
    y_test = labels.loc[test_ind].values

    threads = []
    results_dict = {}  # Use a dictionary to enforce order based on key

    for key in ['top', 'bottom']:
        t = threading.Thread(target=l2_errors_threaded, args=(results_dict, fd_dict, train_ind, test_ind, key))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()
    both = {'train': [], 'test': []}
    plt.figure(figsize=[34.4, 13.27])
    # Process results in a specific order
    for n, key in enumerate(['top', 'bottom']):
        test_errors, train_errors = results_dict[key]
        plt.subplot(2, 1, n + 1)
        plot_errors(labels, unique_labels, label, train_ind, test_ind, y_test, test_errors, train_errors, key)
        both['train'].append(train_errors)
        both['test'].append(test_errors)
    plt.savefig(f"analysis\\figures\\{label}.png")

    train_errors = np.mean(both['train'], axis=0)
    test_errors = np.mean(both['test'], axis=0)
    plt.figure(figsize=[34.4, 13.27])
    gs = gridspec.GridSpec(1, 2, width_ratios=[7, 2.5])
    plt.subplot(gs[0])
    plot_errors(labels, unique_labels, label, train_ind, test_ind, y_test, test_errors, train_errors, 'both')
    cm = confusion_matrix(train_errors, test_errors, y_test, label)
    
    plt.subplot(gs[1])
    plot_confusion_matrix(cm)
    # plot metrics under confusion matrix centered at the middle of the plot horizontally and put under the confusion matrix
    met = metrics(cm)
    plt.text(0.5, -0.25, f"Sensitivity: {met[0]:.2f}\nSpecificity: {met[1]:.2f}\nPrecision: {met[2]:.2f}\nF1: {met[3]:.2f}",
              horizontalalignment='center', transform=plt.gca().transAxes, fontsize=14)
    plt.savefig(f"analysis\\figures\\{label}_both.png")

def confusion_matrix(train_errors, test_errors, y_test, label):
        train_scores = softmax(train_errors, train_errors)
        test_scores = softmax(test_errors, train_errors)
        err_thresh = np.percentile(train_scores, 95)
        act_pos = np.where(y_test!=label)[0]
        act_neg = np.where(y_test==label)[0]
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
    plt.xticks([0, 1], ['Negative', 'Positive'])
    plt.yticks([0, 1], ['Negative', 'Positive'])
    import itertools
    thresh = cm.max() / 2

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center", fontsize=20,
                 color="white" if cm[i, j] > thresh else "black")
    
    plt.vlines(0.5, 1.5, 0.5, color='black')
    plt.hlines(0.5, -0.5, 1.5, color='black')
    


if __name__ == '__main__':
    asset = input("""
    Which asset do you want to plot?
    Options:
    1. VF
    2. UR
    3. Prusa
    > """)

    match asset.upper():
        case '1' | 'VF':
            asset = 'VF'
        case '2' | 'UR':
            asset = 'UR'
        case '3' | 'PRUSA':
            asset = 'Prusa'

    data = pd.read_csv(rf'data\Kernels\2023_02_07\{asset}_merged.csv')
    sample = Sample(data)
    labels = sample.labels
    unique_labels = labels.unique()
    indices = sample.index

    pickled_data = f'data/{asset}_data.pkl'
    if not os.path.exists(pickled_data):
        fd_dict = sample.FData()
        with open(pickled_data, 'wb') as f:
            pickle.dump(fd_dict, f)
    else:
        with open(pickled_data, 'rb') as f:
            fd_dict = pickle.load(f)

    with Pool(len(unique_labels)) as p:
        args = [(label, fd_dict, labels, unique_labels, indices) for label in unique_labels]
        p.starmap(wrapper_plot_basis, args)
