
import sys
import os
import re

# Get the path of the parent directory of the current working directory
sys.path.append(os.getcwd())
from data.read_merge_align_write import create_dataset


START = {"VF-2": "2023_11_18", "UR": "2023_11_21",
         "Bambu": "2023_11_25", "Prusa": r"data\Kernels\PRUSA\Prusa_merged.csv"}


def folders_to_process(typ):
    date_pattern = re.compile(r"\d{4}_\d{2}_\d{2}")
    folder_list = [d for d in os.listdir(
        r".\data\Kernels") if date_pattern.match(d)]
    init_idx = folder_list.index(START[typ])
    fin_idx = len(folder_list)
    return folder_list[init_idx:fin_idx]


def main(force=False):
    folders = folders_to_process("VF-2")
    for folder in folders:
        print(os.path.join(r"data\Kernels", folder))
        for subf in os.listdir(os.path.join(r"data\Kernels", folder)):
            goal = os.path.join(r"data\Kernels", folder, subf)
            if os.path.isdir(goal):
                if os.path.exists(os.path.join(goal, "merged_X.csv")) and not force:
                    print(subf, "already exists")
                    continue
                create_dataset(goal)


if __name__ == "__main__":
    main()
