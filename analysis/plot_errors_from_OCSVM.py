from matplotlib import gridspec
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from multiprocessing import Pool
from sklearn.model_selection import train_test_split

sys.path.append(os.getcwd())
from analysis.plot_errors_from_FDA import *


plt.ion()


def plot_scores(labels, unique_labels, label, train_ind, test_ind, y_test, test_scores, train_scores, test_target_scores):
    plot_errors(labels, unique_labels, label, train_ind,
                test_ind, y_test, test_scores, train_scores, '')
    # remove line that has label "threshold"
    ax = plt.gca()  # Get current axes
    for child in ax.get_children():
        if child.get_label() == 'threshold':
            child.remove()
    perc = 80
    threshold = np.percentile(test_target_scores, perc)
    plt.title(
        f"OCSVM performance for {label} with {perc}th perentile threshold")
    plt.axhline(threshold, color='red', linestyle='--', label='threshold')


def main(label, sample, labels, unique_labels, indices):

    target_idx = indices[labels == label]
    train_ind, _, _, _ = train_test_split(
        target_idx, target_idx, test_size=0.2, random_state=123)
    test_ind = indices.difference(train_ind)
    y_test = labels.loc[test_ind].values

    typ = label[:-2] if "UR" not in label else "UR"
    model_name = os.path.join(os.path.dirname(
        DATA_FILES[typ]), label, f"{label}_ocsvm.pkl")
    ocsvm = load_model(model_name)['model']
    scores = 1/ocsvm.score_samples(sample.drop(["asset"], axis=1))

    train_scores = scores[train_ind]
    test_scores = scores[test_ind]
    test_target_scores = test_scores[y_test == label]
    threshold = np.percentile(test_target_scores, 80)

    plt.figure(figsize=[10, 5])
    gs = gridspec.GridSpec(1, 2, width_ratios=[7, 2.5])
    plt.subplot(gs[0])

    plot_scores(labels, unique_labels, label, train_ind, test_ind,
                y_test, test_scores, train_scores, test_target_scores)

    gs_right = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=gs[1], hspace=0.4)

    # Upper plot for confusion matrix
    plt.subplot(gs_right[0])
    cm = confusion_matrix(test_target_scores, test_scores,
                          y_test, label, threshold)
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
    plt.savefig(f"{FIGURES_DIR}\{label}_OCSVM.png")


def wrapper_plot_basis(label, sample, labels, unique_labels, indices):
    return main(label, sample, labels, unique_labels, indices)


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

    sample = pd.read_csv(DATA_FILES[asset])
    labels = sample["asset"]
    unique_labels = labels.unique()
    indices = sample.index

    with Pool(len(unique_labels)) as p:
        args = [(label, sample, labels, unique_labels, indices)
                for label in unique_labels]
        p.starmap(wrapper_plot_basis, args)
