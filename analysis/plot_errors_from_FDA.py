from FDA import Sample, find_extreme_grid
from skfda.representation.basis import BSpline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from multiprocessing import Pool
import threading
from skfda.misc.metrics import l2_distance, l2_norm
import pickle
import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def plot_errors(labels, unique_labels, label, train_ind, test_ind, y_test, test_errors, train_errors, key):
    plt.plot(train_ind,np.log(train_errors), 'o', color = 'blue', fillstyle='none', label = 'train')
    plt.plot(test_ind[y_test==label], np.log(test_errors[y_test==label]), 'o', color = 'blue', label = 'test target')
    plt.plot(test_ind[y_test!=label], np.log(test_errors[y_test!=label]), 'o', color = 'red', label = 'test other')
    
    err_thresh = np.percentile(train_errors,99)
    plt.hlines(np.log(err_thresh), 0, len(labels), linestyle = '--', color = 'red', label = 'threshold')
    plt.title(f"Errors for {label} using {key} samples")
    plt.legend()
    plt.vlines([(labels == label).idxmax() for label in unique_labels],
                plt.gca().get_ylim()[0], plt.gca().get_ylim()[1], color = 'black',
                 linestyle = '--', label='label change')
    # add text to very label change
    for label in unique_labels:
        plt.text((labels == label).idxmax()+10, plt.gca().get_ylim()[1]-0.05, label)


from multiprocessing import Pool

def main(label, fd_dict, labels,unique_labels, indices):
    np.random.seed(0)
    # Define train and test indices
    target_idx = indices[labels == label]
    train_ind = target_idx[0]+np.random.choice(target_idx.shape[0], 25, replace=False)
    test_ind = indices.difference(train_ind)
    y_test = labels.loc[test_ind].values
    
    plt.figure(figsize=[34.4 , 13.27])
    
    threads = []
    results = []

    for key in ['top', 'bottom']:
        t = threading.Thread(target=l2_errors_threaded, args=(results, fd_dict, train_ind, test_ind, key))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    for n, ((test_errors, train_errors), key) in enumerate(results):
        plt.subplot(2,1,n+1)
        plot_errors(labels, unique_labels, label, train_ind, test_ind, y_test, test_errors, train_errors, key)
    plt.savefig(f"analysis\\figures\\{label}.png")

def l2_errors_threaded(results, fd_dict, train_ind, test_ind, key):
    test_errors, train_errors = l2_errors(fd_dict, train_ind, test_ind, key)
    results.append(((test_errors, train_errors), key))


def l2_errors(fd_dict, train_ind, test_ind, key):
    train = fd_dict[key][train_ind]
    test = fd_dict[key][test_ind]
    target_curve_mean = train.mean()
    tcm = target_curve_mean.data_matrix.reshape(-1)
    knots = find_extreme_grid(tcm, key = key) 

    basis = BSpline(knots=knots)
    print(f"Fitting basis to {key} train data")
    train_basis = train.to_basis(basis)
    print(f"Fitting basis to {key} test data")
    test_basis = test.to_basis(basis)
    print(f"Calculating L2 distance for {key} test data")
    test_errors = l2_distance(test_basis,train_basis.mean())/l2_norm(train_basis.mean())
    print(f"Calculating L2 distance for {key} train data")
    train_errors = l2_distance(train_basis,train_basis.mean())/l2_norm(train_basis.mean())
    return test_errors,train_errors


def wrapper_plot_basis(label, fd_dict, labels,unique_labels, indices):
    return main(label, fd_dict, labels,unique_labels, indices)


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
        args = [(label, fd_dict, labels,unique_labels, indices) for label in unique_labels]
        p.starmap(wrapper_plot_basis, args)
