import os
import sys
import threading
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
sys.path.append(os.getcwd())


from analysis.plot_errors_from_FDA import load_model
from data.merge_X_all import START, folders_to_process
from analysis.testing_models import predict_error
from analysis.plot_errors_from_FDA import ASSET_CHOICES
from analysis.read_merge_align_write import select_files
from analysis.FDA import Sample


def old_model_test(typ=None, file=None, asset=None):
    if not typ:
        typ = input("""
        Which asset do you want to plot?
        Options:
        1. VF-2
        2. UR
        3. Prusa
        4. Bambu
        > """)

        typ = ASSET_CHOICES.get(typ.upper(), None)
        if not typ:
            raise ValueError("Invalid asset choice")
    
    if not file:
        print("Select file to test")
        file = select_files(r".\data\Kernels")[0]
    
    sample = Sample(pd.read_csv(file))
    fd  = sample.FData()
    
    # get indices of labels
    assets = sample['asset'].unique()
    asset_indices = {asset: np.where(sample['asset'] == asset)[0] for asset in assets}  # Get indices of rows corresponding to assets
    result = {asset:[] for asset in assets}

    if "VF" in typ:
        dir = START["VF-2"]
    elif "UR" in typ:
        dir = START["UR"]
    elif "Bambu" in typ:
        dir = START["Bambu"]
    
    if not asset:
        print("Which asset do you want to test?")
        for i, asset in enumerate(assets):
            print(i+1, asset)
        asset = assets[int(input("> "))-1]

    model_dir = os.path.join("data","Kernels", dir, asset)
    
    for model_path in [m for m in os.listdir(model_dir) if "fpca.pkl" in m]:
        model_path = os.path.join(model_dir, model_path)
        print("Using model", model_path)
        model = load_model(model_path)
        error = predict_error(model, fd[model_path.split("_")[-2]])
        for i in asset_indices:
            result[i] = error[asset_indices[i]]

    return result

def process_data(asset):
    typ = asset[:-2] if "UR" not in asset else "UR"
    dir_list = folders_to_process(typ)

    results = {}

    for folder in dir_list[1:-1]:
        
        file = os.path.join(r".\data\Kernels", folder,asset, f"merged_X.csv")
        try:
            errors = old_model_test(typ, asset = asset, file=file)
        except FileNotFoundError:
            print("File not found", file)
            continue

        for asset_er in errors:
            if asset_er not in results:
                results[asset_er] = []
            results[asset_er].append(errors[asset_er])

    for asset_er in results:
        results[asset_er] = np.concatenate(results[asset_er])
    
    return results


def plot_results(asset, n):
    results = process_data(asset)
    typ = asset[:-2] if "UR" not in asset else "UR"
    dir_list = folders_to_process(typ)
    plt.subplot(len(ASSETS[typ]),1, n+1)
    days = np.arange(1.33,len(results[asset])/3+1.33, 1/3)
    for asset_er in results:
        plt.plot(days,results[asset_er], "o", label=asset_er)
        slope, intercept = np.polyfit(days, results[asset_er], 1)
        trend_line = slope * days + intercept
        plt.plot(days, trend_line, label='Trend Line for'+asset_er)

    threshold = np.mean([load_model(os.path.join(r".\data\Kernels", dir_list[0], asset, f"{asset}_{lim}_fpca.pkl"))["threshold"] for lim in ["top", "bottom"]])
    plt.axhline(threshold, color="black", linestyle = "--", label="old threshold")

    plt.legend(loc = "upper left")
    plt.title(f"Error for {asset} old model on new data")
    plt.xlabel("Days")
    plt.ylabel("L2 error")

    plt.xticks(days[1::3],np.arange(1, results[asset].shape[0]/3+1).astype(int))



ASSETS = {"VF-2":["VF-2_1", "VF-2_2"], "UR": ["UR10e_A", "UR5e_N","UR5e_W"], "Bambu":["Bambu_M", "Bambu_S"]}

typ = "Bambu"
plt.figure(figsize = [10,2.5*len(ASSETS[typ])])

for n, asset in enumerate(ASSETS[typ]):
    plot_results(asset, n)

plt.tight_layout()
plt.waitforbuttonpress()
plt.subplot(212)
plt.legend(loc = "upper right")