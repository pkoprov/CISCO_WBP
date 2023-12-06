import sys
import os

# Get the path of the parent directory of the current working directory
sys.path.append( os.getcwd())

from analysis.read_merge_align_write import create_dataset

folders = os.listdir(r"data\Kernels")
init_idx = folders.index("2023_11_18")
fin_idx = folders.index(datetime.now().strftime("%Y_%m_%d"))
folders = folders[init_idx:fin_idx+1]

for folder in folders:
    print(os.path.join(r"data\Kernels",folder))
    for subf in os.listdir(os.path.join(r"data\Kernels",folder)):
        goal = os.path.join(r"data\Kernels",folder,subf)
        if os.path.isdir(goal):
            if os.path.exists(os.path.join(goal,"merged_X.csv")):
                print("already exists")
                continue
            create_dataset(goal)
