import os
import pickle
import threading
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing import Pool
from skfda.representation.basis import BSpline
from skfda.misc.metrics import l2_distance, l2_norm, linf_distance, linf_norm
from FDA import Sample, find_extreme_grid

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
    plt.title(f"Errors for {label} using {key} samples")
    plt.legend()
    plt.vlines([(labels == label).idxmax() for label in unique_labels],
               plt.gca().get_ylim()[0], plt.gca().get_ylim()[1], color='black',
               linestyle='--', label='label change')

    for lbl in unique_labels:
        plt.text((labels == lbl).idxmax()+10, plt.gca().get_ylim()[1]*0.95, lbl)


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
    np.random.seed(0)
    target_idx = indices[labels == label]
    train_ind = target_idx[0] + np.random.choice(target_idx.shape[0], 25, replace=False)
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
    plot_errors(labels, unique_labels, label, train_ind, test_ind, y_test, test_errors, train_errors, 'both')
    plt.savefig(f"analysis\\figures\\{label}_both.png")

    




def wrapper_plot_basis(label, fd_dict, labels, unique_labels, indices):
    return main(label, fd_dict, labels, unique_labels, indices)


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
