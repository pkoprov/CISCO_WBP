import numpy as np
from skfda.preprocessing.dim_reduction import FPCA
from skfda.misc.metrics import l2_distance, l2_norm
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys

sys.path.append(os.getcwd())
from analysis.FDA import Sample
from data.read_merge_align_write import select_files
from analysis.plot_errors_from_FDA import load_model, save_model, error_threshold
from data.merge_X_all import folders_to_process


plt.ion()


def predict_error(fpca, x):
    x_hat = fpca["model"].inverse_transform(fpca["model"].transform(x))
    return l2_distance(x_hat, x) / l2_norm(x)


def plot_errors():
    file = select_files(r".\data\Kernels")[0]
    folder = os.path.dirname(file).split("/")[-1]

    df = pd.read_csv(file)
    sample = Sample(df)
    fd = sample.FData()

    assets = sample['asset'].unique()
    # Get indices of rows corresponding to assets
    asset_indices = {asset: np.where(sample['asset'] == asset)[
        0] for asset in assets}

    models = {}
    results = {asset: {'old': [], 'new': []} for asset in assets}
    for asset in assets:
        total_error, thresh = apply_models(file, folder, fd, models, asset)
        n = 1
        plt.figure()
        for version, err in total_error.items():

            mean_err = np.array(err).mean(axis=0)
            min_thresh = min(thresh[version])
            results[asset][version] = min_thresh - \
                mean_err[asset_indices[asset]]

            plt.subplot(2, 1, n)
            plt.title(" ".join([asset, version, "model"]))
            plt.plot(mean_err, "o")
            plt.hlines(min_thresh, 0, len(mean_err), color="red")
            # plt.hlines(1, 0, len(mean_err), color="red", linestyle="--")
            labels = sample['asset']

            unique, count = np.unique(labels, return_counts=True)

            for lbl, c in zip(unique, count):
                plt.text((labels == lbl).idxmax(),
                         plt.gca().get_ylim()[1]*0.95, lbl)
            plt.vlines([(labels == label).idxmax()-0.5 for label in unique],
                       plt.gca().get_ylim()[0], plt.gca().get_ylim()[1], color='black', linestyle='--', label='label change')
            n += 1
        plt.suptitle(f"{asset} data")
    return results

#  TODO: check this function


def apply_models(file, folder, fd, models, asset):
    folder_list = folders_to_process(
        asset[:-2]) if "UR" not in asset else folders_to_process("UR")
    new_model_dir = None
    old_model_dir = folder_list[0]
    if folder == folder_list[0]:
        print("Using old model")
        new_model_dir = old_model_dir
    else:
        for i, folder_name in enumerate(folder_list[folder_list.index(folder)-1::-1]):
            model_file = os.path.join(
                "/".join(file.split('/')[:-2]), folder_name, asset, f"{asset}_top_fpca.pkl")
            if os.path.exists(model_file):
                new_model_dir = folder_name
                break
            else:
                print("Training model for", asset, folder_name)
                train_model(os.path.join(r".\data\Kernels",
                            folder_name, asset, f"/{asset}_merged_new.csv"))
                new_model_dir = folder_name
                break
        print("Using model from", asset, new_model_dir)
    models[asset] = {
        "top": load_model(fr'data\Kernels\{new_model_dir}\{asset}\{asset}_top_fpca.pkl'),
        "bottom": load_model(fr'data\Kernels\{new_model_dir}\{asset}\{asset}_bottom_fpca.pkl')
    }

    total_error = {'old': [], 'new': []}
    thresh = {'old': [], 'new': []}
    for n, lim in enumerate(['top', 'bottom']):
        old_model = load_model(
            fr'data\Kernels\{old_model_dir}\{asset}\{asset}_{lim}_fpca.pkl')
        old_err = predict_error(old_model, fd[lim])
        new_model = load_model(
            fr'data\Kernels\{new_model_dir}\{asset}\{asset}_{lim}_fpca.pkl')
        new_err = predict_error(new_model, fd[lim])
        thresh['old'].append(old_model["threshold"])
        thresh['new'].append(new_model["threshold"])

        total_error["old"].append(old_err)
        total_error["new"].append(new_err)
    return total_error, thresh


def find_updated_model_dir(file):
    asset = os.path.dirname(file).split("\\")[-1]
    model_dir = None
    folders = folders_to_process(
        asset[:-2]) if "UR" not in asset else folders_to_process("UR")
    fin_idx = folders.index(file.split("\\")[-3])
    for folder in folders[:fin_idx][::-1]:
        model_dir = os.path.join(r'.\data\Kernels', folder, asset)
        if os.path.exists(model_dir):
            return model_dir
    if model_dir is None:
        print("No training data found")
        return


def apply_model(file, model_dir=None, model="fpca"):
    df = pd.read_csv(file)
    # get indices of labels
    assets = df['asset'].unique()
    # Get indices of rows corresponding to assets
    asset_indices = {asset: np.where(df['asset'] == asset)[
        0] for asset in assets}
    if model.lower() == "fpca":
        sample = Sample(df)
        fd = sample.FData()
    else:
        sample = df
        sample.drop(["asset"], axis=1, inplace=True)
        sample.columns = sample.columns.astype(float)

    if not model_dir:
        model_dir = find_updated_model_dir(file)

    result = {asset: {'scores': [], 'threshold': []} for asset in assets}
    for model_name in [m for m in os.listdir(model_dir) if f"{model}.pkl" in m]:
        model_path = os.path.join(model_dir, model_name)
        print("Using model", model_path)
        saved_model = load_model(model_path)
        scores = predict_error(saved_model, fd[model_path.split(
            "_")[-2]]) if model == "fpca" else 1/saved_model['model'].score_samples(sample)
        for i in asset_indices:
            result[i]['scores'].append(scores[asset_indices[i]])
            result[i]['threshold'].append(saved_model["threshold"])

    return result


def test_new_data(file=None):
    if file is None:
        print("Select the file to test")
        file = select_files(r".\data\Kernels")[0]
        print("Selected file is ", file)

    result = apply_model(file)
    asset = list(result.keys())[0]

    total_err = np.mean((result[asset]['errors']), axis=0)
    total_thresh = np.mean((result[asset]['threshold']), axis=0)
    plt.plot(total_err, "o")
    plt.hlines(total_thresh, 0, len(total_err), color="red")
    TP = np.sum(total_err > total_thresh)
    FP = np.sum(total_err < total_thresh)
    print("TP:", TP)
    print("FP:", FP)


def train_model(path=None):

    try:
        train_df = pd.read_csv(path)
    except FileNotFoundError:
        print("Select dataset to train on")
        train_file = select_files(r".\data\Kernels")[0]
        train_df = pd.read_csv(train_file)

    train_sample = Sample(train_df)
    train_fd = train_sample.FData()
    model_dict = {}
    for key in ['top', 'bottom']:
        model = FPCA(n_components=1)
        print(f"Fitting {key} model")
        model.fit(train_fd[key])
        train_set_hat = model.inverse_transform(model.transform(train_fd[key]))
        error = l2_distance(
            train_set_hat, train_fd[key]) / l2_norm(train_fd[key])
        thresh = error_threshold(error)
        save_as = f"{os.path.dirname(path)}\{os.path.basename(os.path.dirname(path))}_{key}_fpca.pkl"
        model_dict[key] = {"model": model, "threshold": thresh}
        save_model(model_dict[key], save_as)
        print(f"Saved model to {save_as}")

    return model_dict


if __name__ == '__main__':
    cmd = input("Test data or plot errors?\n(t/p)\n>>> ")
    match cmd:
        case 't':
            test_new_data()
        case 'p':
            plot_errors()

    plt.show()
    plt.waitforbuttonpress()
