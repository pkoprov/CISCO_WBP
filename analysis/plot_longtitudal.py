import os
import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.getcwd())
from analysis.testing_models import apply_model, find_updated_model_dir
from data.merge_X_all import START, folders_to_process
from analysis.plot_errors_from_FDA import load_model


def process_data(model, asset, version):
    typ = asset[:-2] if "UR" not in asset else "UR"
    dir_list = folders_to_process(typ)

    results = {}

    for folder in dir_list[1:]:

        file = os.path.join(r".\data\Kernels", folder, asset, f"merged_X.csv")
        if version == "old":
            model_dir = START[typ]
            model_dir = os.path.join(r".\data\Kernels", model_dir, asset)
        else:
            model_dir = find_updated_model_dir(file)
            file = os.path.join(r".\data\Kernels", folder, f"{typ}_merged.csv")

        try:
            prediction = apply_model(file, model_dir, model)

        except FileNotFoundError:
            print("File not found", file)
            continue

        for key in prediction.keys():
            if key not in results:
                results[key] = {'scores': [], 'threshold': []}
            results[key]['scores'].append(
                np.mean(prediction[key]['scores'], axis=0))
            results[key]['threshold'].append(
                np.mean(prediction[key]["threshold"], axis=0))

    for key in results.keys():
        results[key]['scores'] = np.concatenate(results[key]['scores'])

    return results


def plot_results(model, asset, n, version):
    results = process_data(model, asset, version)
    typ = asset[:-2] if "UR" not in asset else "UR"
    col = ['blue', 'red'] if "UR" not in asset else ['blue', 'red', 'green']
    dir_list = folders_to_process(typ)
    plt.subplot(len(ASSETS[typ]), 1, n+1)

    for key, col in zip(results.keys(), col):
        days = np.arange(1.33, len(results[key]['scores'])/3+1.33, 1/3)
        plt.plot(days, results[key]['scores'], ".", label=key, color=col)

    days = np.arange(1.33, len(results[asset]['scores'])/3+1.33, 1/3)
    slope, intercept = np.polyfit(days, results[asset]["scores"], 1)
    trend_line = slope * days + intercept
    plt.plot(days, trend_line, label='trend Line for '+asset, color="orange")

    if version == "old":
        threshold = np.mean([load_model(os.path.join(r".\data\Kernels", dir_list[0], asset, f"{asset}_{lim}_fpca.pkl"))["threshold"] for lim in [
                            "top", "bottom"]]) if model == 'fpca' else load_model(os.path.join(r".\data\Kernels", dir_list[0], asset, f"{asset}_ocsvm.pkl"))["threshold"]
        plt.axhline(threshold, color="black",
                    linestyle="--", label="old threshold")
        plt.title(f"Scores for {asset} old {model.upper()} model on new data")

    else:
        threshold = np.array(results[asset]['threshold'])
        thresh_per_day = np.repeat(threshold, 3)
        plt.plot(days, thresh_per_day, label="Updated threshold",
                 color="black", linestyle="--")
        plt.title(f"Scores for {asset} updated {model.upper()} models on new data")

    plt.xlabel("Days")
    plt.ylabel("L2 error")
    plt.xticks(days[1::3], np.arange(1, results[asset]
               ['scores'].shape[0]/3+1).astype(int))
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))


def model_performance(model, version):
    cmd = input(
        "Which asset do you want to plot?\n1. VF-2\n2. UR\n3. Bambu\n>> ")
    typ = list(ASSETS.keys())[int(cmd)-1]

    plt.figure(figsize=[10, 2.5*len(ASSETS[typ])])
    for n, asset in enumerate(ASSETS[typ]):
        plot_results(model, asset, n, version)

    plt.tight_layout()
    plt.waitforbuttonpress()


ASSETS = {"VF-2": ["VF-2_1", "VF-2_2"], "UR": ["UR10e_A",
                                               "UR5e_N", "UR5e_W"], "Bambu": ["Bambu_M", "Bambu_S"]}

if __name__ == "__main__":
    prog = input("Model to run:\n1. old\n2. updated\n>> ")
    model = input("Model to run:\n1. FPCA\n2. OCSVM\n>> ")
    model_performance('fpca' if model == '1' else 'ocsvm', "old" if prog == "1" else "updated")
