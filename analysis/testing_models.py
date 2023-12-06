import os
import pandas as pd
import matplotlib.pyplot as plt
from skfda.misc.metrics import l2_distance, l2_norm
from skfda.preprocessing.dim_reduction import FPCA
# from analysis.plotting import shift_for_maximum_correlation
import numpy as np

try:
    from FDA import Sample
    from read_merge_align_write import select_files
    from plot_errors_from_FDA import load_model, save_model, error_threshold
except ModuleNotFoundError:
    from analysis.FDA import Sample
    from analysis.read_merge_align_write import select_files
    from analysis.plot_errors_from_FDA import load_model, save_model, error_threshold


plt.ion()


def predict_error(fpca, x):
    x_hat = fpca["model"].inverse_transform(fpca["model"].transform(x))
    return l2_distance(x_hat, x) / l2_norm(x)


def plot_errors():
    file = select_files(r".\data\Kernels")[0]
    dir = os.path.dirname(file)

    folder_list = os.listdir("data\Kernels")
    start_idx = folder_list.index("2023_11_18")
    stop_idx =  folder_list.index("2023_12_04")+1
    folder_list = folder_list[start_idx:stop_idx]
    dir_idx = folder_list.index(dir.split('/')[-1])

    df = pd.read_csv(file)
    sample = Sample(df)
    fd = sample.FData()

    assets = sample.iloc[:, 0].unique()
    assets = [f"{i}: {asset}" for i, asset in enumerate(assets)]
    assets_str = '\n'.join(assets)
    cmd = input(f"Which asset is a target?\n{assets_str}\n>>> ")
    asset = assets[int(cmd)].split(":")[1].strip()
    i=0
    model_dir = None
    while model_dir is None or not os.path.exists(model_dir):
        i+=1
        model_dir = os.path.join("/".join(dir.split('/')[:-1]), folder_list[dir_idx-i], asset)


    total_error = {'old': [], 'new': []}
    thresh = {'old': [], 'new': []}
    for n, lim in enumerate(['top', 'bottom']):
        old_model = load_model(fr'data\Kernels\2023_12_04\{asset}\{asset}_{lim}_fpca.pkl')
        old_err = np.log(1/predict_error(old_model, fd[lim]))
        new_model = load_model(fr'{model_dir}/{asset}_{lim}_fpca.pkl')
        new_err = np.log(1/predict_error(new_model, fd[lim]))
        thresh['old'].append(old_model["threshold"])
        thresh['new'].append(new_model["threshold"])

        total_error["old"].append(old_err)
        total_error["new"].append(new_err)
    n = 1
    for version, err in total_error.items():
        plt.subplot(2, 1, n)
        plt.title(" ".join([asset, version, "model"]))
        mean_err = np.array(err).mean(axis=0)
        mean_thresh = np.array(thresh[version]).mean()
        plt.plot(mean_err, "o")
        plt.hlines(mean_thresh, 0, len(mean_err), color="red")
        plt.hlines(1, 0, len(mean_err), color="red", linestyle="--")
        labels = sample['asset']

        unique, count = np.unique(labels, return_counts=True)

        for lbl, c in zip(unique, count):
            plt.text((labels == lbl).idxmax(),
                     plt.gca().get_ylim()[1]*0.95, lbl)
        plt.vlines([(labels == label).idxmax()-0.5 for label in unique],
                   plt.gca().get_ylim()[0], plt.gca().get_ylim()[1], color='black', linestyle='--', label='label change')
        n += 1
    plt.suptitle(f"{asset} data")
        # plt.save(fr"analysis\figures\{asset}_{file[-7:-4]}_data_{version}_model.png")


VERSION_CHOICES = {'0': 'old', '1': 'new'}


def test_data():
    print("Select the file to test")
    file = select_files(r".\data\Kernels")[0]
    print("Selected file is ", file)
    dir = os.path.dirname(file)
    file = os.path.basename(file)

    df = pd.read_csv(os.path.join(dir, file))
    sample = Sample(df)
    fd = sample.FData()
    asset = dir.split("/")[-1]
    dir_list = os.listdir(r".\data\Kernels")

    i=0
    train_fold = None
    while train_fold is None or not os.path.exists(train_fold):
        i+=1
        train_fold_idx = dir_list.index(dir.split('/')[-2])-i
        train_fold = os.path.join(
            "/".join(dir.split('/')[:-2]), os.listdir(r".\data\Kernels")[train_fold_idx], asset)
    
    print(f"Using {train_fold} for training")
          
    model_top = load_model(fr'{train_fold}/{asset}_top_fpca.pkl') if os.path.exists(
        fr'{train_fold}/{asset}_top_fpca.pkl') else train_model(asset, train_fold)["top"]
    model_bottom = load_model(fr'{train_fold}/{asset}_bottom_fpca.pkl') if os.path.exists(
        fr'{train_fold}/{asset}_bottom_fpca.pkl') else train_model(asset, train_fold)["bottom"]


    top_err = np.log(1/predict_error(model_top, fd['top']))
    bottom_err = np.log(1/predict_error(model_bottom, fd['bottom']))
    total_err = np.mean((top_err, bottom_err), axis=0)
    total_thresh = np.mean((model_top["threshold"], model_bottom["threshold"]))
    plt.plot(total_err, "o")
    plt.hlines(total_thresh, 0, len(total_err), color="red")
    plt.hlines(1, 0, len(total_err), color="red", linestyle="--")
    TP = np.sum(total_err > total_thresh)
    FP = np.sum(total_err < total_thresh)
    print("TP:", TP)
    print("FP:", FP)


def train_model(asset, train_fold):
    
    try:
        train_df = pd.read_csv(train_fold+f"/{asset}_merged_new.csv")
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
        thresh = error_threshold(np.log(1/error))
        save_as = f"{train_fold}\{asset}_{key}_fpca.pkl"
        model_dict[key] = {"model": model, "threshold": thresh}
        save_model(model_dict[key], save_as)
        print(f"Saved model to {save_as}")
        
    return model_dict


if __name__ == '__main__':
    cmd = input("Test data or plot errors?\n(t/p)\n>>> ")
    match cmd:
        case 't':
            test_data()
        case 'p':
            plot_errors()
    
    plt.show()
    plt.waitforbuttonpress()
