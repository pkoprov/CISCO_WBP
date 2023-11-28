import pandas as pd
import matplotlib.pyplot as plt
from skfda.misc.metrics import l2_distance, l2_norm
# from analysis.plotting import shift_for_maximum_correlation
import numpy as np
try:
    from FDA import Sample
    from read_merge_align_write import select_files, prepare_sample
    from plot_errors_from_FDA import load_model
except ModuleNotFoundError:
    from analysis.FDA import Sample
    from analysis.read_merge_align_write import select_files, prepare_sample
    from analysis.plot_errors_from_FDA import load_model


# plt.ion()


def predict_error(fpca, x):
    x_hat = fpca["model"].inverse_transform(fpca["model"].transform(x))
    return l2_distance(x_hat, x) / l2_norm(x)


# benchmark = pd.read_csv(select_files()[0], index_col=0)
# benchmark.plot( ax = plt.gca())

# # grid = benchmark.index.to_series()
# # grid.index.name = None
# grid = pd.Series(np.round(np.arange(0, 8.5, 0.001),3), name = "Time")
# df = pd.read_csv(select_files()[0]).iloc[:,:2]
# x = prepare_sample(df)
# x.index = x["Time"]
# x.drop("Time", axis=1, inplace=True)

# x = x.merge(grid, on="Time", how="right")
# len = 8500
# x = x.sort_values(by="Time").reset_index(drop=True).iloc[:len, :]
# x.fillna(x.iloc[:,1].median(), inplace=True)

# x.plot(x= "Time", ax = plt.gca(), legend=False)

# x_sync,_ = shift_for_maximum_correlation(benchmark.iloc[:,0], x.iloc[:,1])
# plt.plot(grid,x_sync, color="red")

# x_df = pd.DataFrame(x.iloc[:,1]).T
# x_df.columns = grid
# x_df.reset_index(inplace=True, drop=True)
# x_df.insert(0, "asset", "VF-2_1")


# x_fd = Sample(x_df).FData()
# x_fd["top"].plot(axes=plt.gca(), color = "blue")
# x_fd["bottom"].plot(axes=plt.gca(), color = "blue")



# fpca = load_model(select_files()[0])

# err = predict_error(fpca, x_fd['bottom'])

# if err > fpca["threshold"]*1.15:
#     print("Intruder")
# else:
#     print("Normal")


def plot_errors():
    df = pd.read_csv(select_files()[0])
    sample = Sample(df)
    fd = sample.FData()

    assets = sample.iloc[:,0].unique()
    assets = [f"{i}: {asset}" for i, asset in enumerate(assets)]
    assets_str = '\n'.join(assets)
    cmd = input(f"Which asset is a target?\n{assets_str}\n>>> ")
    asset = assets[int(cmd)].split(":")[1].strip()
    plt.suptitle(asset)
    total_error = []
    for n,lim in enumerate(['top', 'bottom']):
        model  = load_model(fr'D:/Users/pkoprov/PycharmProjects/Vibration_Patterns/analysis/models/{asset}_{lim}_fpca.pkl')
        err = predict_error(model, fd[lim])
        plt.subplot(2,1,n+1)
        plt.plot(err,"o")
        plt.hlines(model["threshold"], 0, len(err), color="red")
        plt.title(lim)
        total_error.append(err)

    mean_err = np.array(total_error).mean(axis=0)
    plt.figure()
    plt.plot(mean_err,"o")
    plt.hlines(model["threshold"], 0, len(err), color="red")

if __name__ == '__main__':
    plot_errors()
    plt.show()
    plt.waitforbuttonpress()
