import os
import sys
sys.path.append(os.getcwd())
from data.merge_X_all import folders_to_process, START
from analysis.testing_models import train_model


def main(force=False):
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

            model_path = os.path.join(path, subf, f"{subf}_merged_new.csv") if folder != START[typ] else os.path.join(path, subf,"merged_X.csv")
            if sum([True for model in os.listdir(os.path.dirname(model_path)) if "_fpca.pkl" in model]) == 2 and not force:
                print(subf, "model already exists")
                continue
            train_model(model_path)


if __name__ == "__main__":
    main()