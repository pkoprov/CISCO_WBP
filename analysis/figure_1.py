import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from analysis.plot_errors_from_FDA import load_data, load_model

plt.ion()

asset = "UR"
label = "UR5e_N"
key = 'top'
MODEL_DIR = "analysis/models"
FIGURES_DIR = "analysis/figures"
DATA_DIR = r"data\train_datasets"
PICKLED_DATA_DIR = "data"

sample = load_data(asset)
fd = sample.FData()
labels = sample.labels
unique_labels = labels.unique()
indices = sample.index
fd_dict = sample.FData()


## Figure 5
# Plotting the curves in separate subplots
fig, axs = plt.subplots(len(unique_labels), 1, figsize=(10, 6))


for i, label in enumerate(unique_labels):
    label_indices = indices[labels == label]
    axs[i].plot(sample.numeric.columns, sample.numeric.loc[label_indices].T, label="data", color="black", alpha=0.03)
    axs[i].set_title(label)
    fd["top"][label_indices].mean().plot(axes=axs[i], color="red", linewidth=2, label="top")
    fd["bottom"][label_indices].mean().plot(axes=axs[i], color="blue", linewidth=2, label="bottom")
    axs[i].set_ylim(-0.2, 0.2)


plt.tight_layout()


## Figure 6
target_idx = indices[labels == label]
train_ind, _, _, _ = train_test_split(
    target_idx, target_idx, test_size=0.2, random_state=123)
test_ind = indices.difference(train_ind)
y_test = labels.loc[test_ind].values
train = fd_dict[key][train_ind]
test = fd_dict[key][test_ind]
model_name = f"{MODEL_DIR}\{label}_{key}_fpca_old.pkl"
fpca_clean = load_model(model_name)["model"]
train_set_hat = fpca_clean.inverse_transform(fpca_clean.transform(train))
test_hat = fpca_clean.inverse_transform(fpca_clean.transform(test))
plt.subplot(311)
train.mean().plot(plt.gca(), label=f"original train {label} {key} mean curve")
train_set_hat.mean().plot(plt.gca(), label=f"reconstructed train {label} {key} mean curve")
plt.title(f"Mean train {key} curve for {label}")
plt.legend()
plt.subplot(312)
test[0].plot(plt.gca(), label=f"original {labels[test_ind[0]]} {key} curve")
test_hat[0].plot(plt.gca(), label=f"reconstructed {labels[test_ind[0]]} {key} curve")
plt.title(f"First test {key} curve ({labels[test_ind[0]]}) projected on {label} {key} curve FPC")
plt.legend()
plt.subplot(313)
test[-1].plot(plt.gca(), label=f"original {labels[test_ind[-1]]} {key} curve")
test_hat[-1].plot(plt.gca(), label=f"reconstructed {labels[test_ind[0]]} {key} curve")
plt.title(f"Last test {key} curve ({labels[test_ind[-1]]}) projected on {label} {key} curve FPC")
plt.legend()
plt.tight_layout()