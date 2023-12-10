import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
sys.path.append(os.getcwd())


from analysis.plot_errors_from_FDA import load_model
from data.merge_X_all import START, folders_to_process
from analysis.testing_models import predict_error, apply_model, find_updated_model
from analysis.FDA import Sample


def process_data(asset, version):
    typ = asset[:-2] if "UR" not in asset else "UR"
    dir_list = folders_to_process(typ)

    results = {}

    for folder in dir_list[1:]:
        
        file = os.path.join(r".\data\Kernels", folder,asset, f"merged_X.csv")
        if version == "old":
            if "VF" in typ:
                model_dir = START["VF-2"]
            elif "UR" in typ:
                model_dir = START["UR"]
            elif "Bambu" in typ:
                model_dir = START["Bambu"]
            model_dir = os.path.join(r".\data\Kernels", model_dir, asset)
        else:
            model_dir = find_updated_model(file)
            file = os.path.join(r".\data\Kernels", folder, f"{typ}_merged.csv")

        try:          
            prediction = apply_model(file, model_dir)
            
        except FileNotFoundError:
            print("File not found", file)
            continue

        for key in prediction.keys():
            if key not in results:
                results[key] = {'errors':[], 'threshold':[]}
            results[key]['errors'].append(np.mean(prediction[key]['errors'], axis=0))
            results[key]['threshold'].append(np.mean(prediction[key]["threshold"], axis = 0))

    for key in results.keys():
        results[key]['errors'] = np.concatenate(results[key]['errors'])
    
    return results


def plot_results(asset, n, version):
    results = process_data(asset, version)
    typ = asset[:-2] if "UR" not in asset else "UR"
    col = ['blue', 'red'] if "UR" not in asset else ['blue', 'red', 'green']
    dir_list = folders_to_process(typ)
    plt.subplot(len(ASSETS[typ]),1, n+1)

    for key, col in zip(results.keys(),col):   
        days = np.arange(1.33,len(results[key]['errors'])/3+1.33, 1/3)
        plt.plot(days,results[key]['errors'], ".", label=key, color=col)

    days = np.arange(1.33,len(results[asset]['errors'])/3+1.33, 1/3)
    slope, intercept = np.polyfit(days, results[asset]["errors"], 1)
    trend_line = slope * days + intercept
    plt.plot(days, trend_line, label='Trend Line for '+asset)

    if version == "old":
        threshold = np.mean([load_model(os.path.join(r".\data\Kernels", dir_list[0], asset, f"{asset}_{lim}_fpca.pkl"))["threshold"] for lim in ["top", "bottom"]])
        plt.axhline(threshold, color="black", linestyle = "--", label="old threshold")
        plt.title(f"Error for {asset} old model on new data")

    else:
        threshold = np.array(results[asset]['threshold'])
        thresh_per_day = np.repeat(threshold, 3)
        plt.plot(days, thresh_per_day, label="Updated threshold", color = "black", linestyle = "--")
        plt.title(f"Error for {asset} updated models on new data")
    
    plt.xlabel("Days")
    plt.ylabel("L2 error")
    plt.xticks(days[1::3],np.arange(1, results[asset]['errors'].shape[0]/3+1).astype(int))
    plt.legend(loc = "upper right")

def model_performance(version):
    cmd = input("Which asset do you want to plot?\n1. VF-2\n2. UR\n3. Bambu\n>> ")
    typ = list(ASSETS.keys())[int(cmd)-1]

    plt.figure(figsize = [10,2.5*len(ASSETS[typ])])
    for n, asset in enumerate(ASSETS[typ]):
        plot_results(asset, n, version)

    plt.tight_layout()
    plt.waitforbuttonpress()

ASSETS = {"VF-2":["VF-2_1", "VF-2_2"], "UR": ["UR10e_A", "UR5e_N","UR5e_W"], "Bambu":["Bambu_M", "Bambu_S"]}

if __name__ == "__main__":
    prog = input("Model to run:\n1. old\n2. updated\n>> ")
    model_performance("old" if prog == "1" else "updated")
