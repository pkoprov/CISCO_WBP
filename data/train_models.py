import os
import sys
sys.path.append(os.getcwd())
from data.merge_X_all import folders_to_process, START
from analysis.testing_models import train_model


if __name__ == "__main__":
    folders = folders_to_process("VF-2")
    for folder in folders:
        print(os.path.join(r"data\Kernels", folder))
        for subf in [subf for subf in os.listdir(os.path.join(r"data\Kernels", folder)) if os.path.isdir(os.path.join(r"data\Kernels", folder, subf))]:
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

            path = os.path.join(r"data\Kernels", folder, subf, f"{subf}_merged_new.csv") if folder != START[typ] else os.path.join(r"data\Kernels", folder, subf,"merged_X.csv")
            if sum([True for model in os.listdir(os.path.dirname(path)) if "_fpca.pkl" in model]) == 2:
                print(subf, "model already exists")
                continue
            train_model(path)
