import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from analysis.plot_errors_from_FDA import load_data, load_model
from analysis.testing_models import apply_model

plt.ion()


typ = "UR"

sample = load_data(typ)
fd = sample.FData()
labels = sample.labels
unique_labels = labels.unique()
indices = sample.index
fd_dict = sample.FData()


# Figure 5
# Plotting the curves in separate subplots
fig, axs = plt.subplots(len(unique_labels), 1, figsize=(10, 5))


for i, label in enumerate(unique_labels):
    label_indices = indices[labels == label]
    axs[i].plot(sample.numeric.columns, sample.numeric.loc[label_indices].T,
                label="data", color="black", alpha=0.03)
    axs[i].set_title(label)
    fd["top"][label_indices].mean().plot(
        axes=axs[i], color="red", linewidth=2, label="top")
    fd["bottom"][label_indices].mean().plot(
        axes=axs[i], color="blue", linewidth=2, label="bottom")
    axs[i].set_ylim(-0.2, 0.2)


plt.tight_layout()

label = 'UR5e_N'
# Figure 6
target_idx = indices[labels == label]
train_ind, _, _, _ = train_test_split(
    target_idx, target_idx, test_size=0.2, random_state=123)
test_ind = indices.difference(train_ind)
y_test = labels.loc[test_ind].values

key = 'top'
train = fd_dict[key][train_ind]
test = fd_dict[key][test_ind]
model_name = fr"data\Kernels\2023_11_21\{label}\{label}_top_fpca.pkl"
fpca_clean = load_model(model_name)["model"]
train_set_hat = fpca_clean.inverse_transform(fpca_clean.transform(train))
test_hat = fpca_clean.inverse_transform(fpca_clean.transform(test))
plt.figure(figsize=(8, 7.5))
plt.subplot(311)
train.mean().plot(plt.gca(), label=f"original curve", color="blue")
train_set_hat.mean().plot(plt.gca(), label=f"reconstructed curve", color="orange")
plt.title(f"Mean train {key} curve for {label}")
plt.legend(loc='best')
plt.subplot(312)
n = np.random.choice(test_ind[y_test != label])
test[n].plot(plt.gca(), label=f"original curve", color="blue")
test_hat[n].plot(plt.gca(), label=f"reconstructed curve", color='red')
plt.title(
    f"Random test {key} curve ({labels[test_ind[n]]}) projected on {label} {key} curve FPC")
plt.legend(loc='best')
plt.subplot(313)
n = np.random.choice(test_ind[y_test == label])
test[n].plot(plt.gca(), label=f"original curve", color="blue")
test_hat[n].plot(plt.gca(), label=f"reconstructed curve", color='orange')
plt.title(
    f"Random test {key} curve ({labels[test_ind[n]]}) projected on {label} {key} curve FPC")
plt.legend(loc='best')
plt.tight_layout()


#  plot data that has been impacted by load/noize
for asset in ["VF-2_1", "UR10e_A", "UR5e_N", "Bambu_M"]:
    match asset:
        case "VF-2_1":
            file = r"data\Kernels\2023_12_08\VF-2_1\with payload\merged_X.csv"
        case "UR10e_A":
            file = r'data\Kernels\2023_12_08\UR10e_A\with payload\merged_X.csv'
        case "UR5e_N":
            file = r'data\Kernels\2023_12_08\UR5e_N\with payload\merged_X.csv'
        case "Bambu_M":
            file = r'data\Kernels\2023_12_08\Bambu_M\with noize\merged_X.csv'

    model_dir = fr"data\Kernels\2023_12_07\{asset}"

    model_threshold = []
    for key in ["top", "bottom"]:
        model_threshold.append(load_model(
            fr"{model_dir}\{asset}_{key}_fpca.pkl")["threshold"])
    model_threshold = np.mean(model_threshold)

    result_w_load = apply_model(file, model_dir, model="fpca")
    errors_w_load = np.mean(result_w_load[asset]["scores"], axis=0)

    typ = asset[:-2] if "UR" not in asset else "UR"
    other_file = '\\'.join(file.split('\\')[:-3] + [typ+'_merged.csv'])
    result_wo_load = apply_model(other_file, model_dir, model="fpca")
    errors_wo_load_a = np.mean(result_wo_load.pop(asset)["scores"], axis=0)
    errors_wo_load_o = {asset: np.mean(result_wo_load[asset]["scores"], axis=0) for asset in [
        key for key in result_wo_load.keys()]}

    plt.figure(figsize=(8, 3))
    plt.plot(errors_w_load, 'v', label=f"{asset} with payload", color="blue")
    plt.plot(errors_wo_load_a, 'o', fillstyle="none",
             label=f" {asset} without payload", color="blue")
    [plt.plot(errors_wo_load_o[asset], 'o',
              label=f"{asset} without payload", color="red") for asset in errors_wo_load_o]
    plt.axhline(model_threshold, color="black", label="threshold")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title(
        f"Errors for {asset} with and without payload using updated model")
    plt.tight_layout()
