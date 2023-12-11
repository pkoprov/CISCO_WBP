import os
import pickle
import sys
import pandas as pd
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.model_selection import GridSearchCV,train_test_split
sys.path.append(os.getcwd())
from data.merge_X_all import folders_to_process, START
from analysis.testing_models import train_model
from analysis.read_merge_align_write import select_files
from analysis.plot_errors_from_FDA import confusion_matrix, load_model, save_model



def main(model="fpca", force=False):
    folders = folders_to_process("VF-2")
    for folder in folders:
        path = os.path.join(r"data\Kernels", folder)
        print(path)
        for subf in [subf for subf in os.listdir(path) if os.path.isdir(os.path.join(path, subf))]:
            print(subf)
            if "VF" in subf:
                typ = "VF-2"
            elif "Bambu" in subf:
                typ = "Bambu"
            elif "UR" in subf:
                typ = "UR"
            else:
                print("Unknown type")
                typ = input("Type the asset type:\n>>")

            data_path = os.path.join(path, subf, f"{subf}_merged_new.csv") if folder != START[typ] else os.path.join(path, subf,"merged_X.csv")
            if model.lower() == 'fpca':
                if sum([True for model in os.listdir(os.path.dirname(data_path)) if "_fpca.pkl" in model]) == 2 and not force:
                    print(subf, "model already exists")
                    continue
                train_model(data_path)
            elif model.lower() == 'ocsvm':
                if os.path.exists(os.path.join(os.path.dirname(data_path), subf+"_ocsvm.pkl")) and not force:
                    print(subf, "model already exists")
                    continue
                train_OCSVM_model(data_path)

def train_OCSVM_model(data_path):
    print("Training OCSVM model for ", data_path, "...")
    try:
        train_df = pd.read_csv(data_path)
    except FileNotFoundError:
        print("Select dataset to train on")
        train_file = select_files(r".\data\Kernels")[0]
        train_df = pd.read_csv(train_file)

    label = train_df["asset"].unique()[0]

    train_df.drop(["asset"], axis=1, inplace=True)
    train_df, test_df = train_test_split(train_df, test_size=0.25, random_state=123)
    
    param_grid = {
    'kernel': ['rbf', 'poly', 'sigmoid'],
    'gamma': ['scale', 'auto'],
    'nu': np.linspace(0.01, 0.5, 10)}

    grid_search = GridSearchCV(OneClassSVM(), param_grid, scoring = custom_scorer(label),n_jobs=-1)

    grid_search.fit(train_df)
    model = grid_search.best_estimator_
    grid_search.best_params_
    test_scores = 1/model.score_samples(test_df)
    threshold = np.percentile(test_scores, 90)
    model = {"model": model, "threshold": threshold}
    save_as = os.path.join(os.path.dirname(data_path), label+"_ocsvm.pkl")
    save_model(model, save_as)


def custom_scorer(label):
    def scoring_function(estimator, X, y):
        test_scores = 1 / estimator.score_samples(X)
        threshold = np.percentile(test_scores[y == label], 90)

        # Calculate Sensitivity 
        cm = confusion_matrix(test_scores, test_scores, y, label, threshold)
        sensitivity = cm[0]/(cm[0]+cm[2])

        return sensitivity
    return scoring_function


if __name__ == "__main__":
    cmd = input("Train FPCA or OCSVM model?\n(f/o)\n>>> ")
    if cmd.lower() == 'f':
        model = "fpca"
    elif cmd.lower() == 'o':
        model = "ocsvm"
    else:
        print("Unknown command")
        sys.exit()
    main(model)