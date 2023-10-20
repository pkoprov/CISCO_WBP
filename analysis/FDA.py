import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import skfda
from skfda.representation.basis import BSpline


plt.ion()
data = pd.read_csv(r'data\Kernels\2023_02_07\VF_merged.csv')
# convert columns names  to float
sample = data.iloc[0,1:].astype(float)
sample.index = sample.index.astype(float)

def figure():

    manager = plt.get_current_fig_manager()
    manager.window.wm_geometry("-5000+1")
    plt.pause(0.5)
    manager.window.state('zoomed')
    plt.show(block=False)

figure()
sample.plot()


def top_bottom(sample: pd.Series):
    ''' Function to find the top and bottom shape of a sample'''
    positive = sample[sample > 0]
    negative = sample[sample < 0]
    top,bottom = [0], [0]

    for var, fun, lst  in zip([positive, negative], [np.argmax, np.argmin], [top, bottom]):
        # var, fun, lst  = positive, np.argmax, top
        n = 0
        while n < sample.shape[0]/1000:
            try:
                chunk = var.loc[n:round(n+0.01, 2)]
                max_ind = fun(chunk)
                ms = chunk.index[max_ind]
                if ms not in lst:
                    lst.append(chunk.index[max_ind])
            except:
                pass
        
            n = round(n+0.01, 2)
            # print(n, lst)
    return top,bottom

top, bottom = top_bottom(sample)

# plot top and bottom shapes
for var in [top, bottom]:
    sample[var[0]] = 0
    plt.plot(var, sample[var], marker=".")

# interpolate the missing x values
def fit_missing(sample):
    limits = top_bottom(sample)
    lim_dict = {'top': [], 'bottom': []}
    for lim, key in zip(limits, lim_dict):
        data_matrix = sample[lim].values
        data_matrix[0] = 0
        # create a grid
        x = np.arange(0, sample.index[-1]+0.001, 0.001)
        x = np.round(x, 3)
        # interpolate the data
        y = np.interp(x, lim, data_matrix)
        lim_dict[key] = y
    return lim_dict


dat = sample.loc[:5.0]

lim_dict = fit_missing(dat)

grid = np.round(np.arange(0, dat.index[-1]+0.001, 0.001),3)
y = lim_dict['top']
curve = skfda.FDataGrid(y, grid_points=grid)

curve.plot(axes = plt.gca(), color = "blue")

basis = basis = BSpline(knots=grid, order =2)
basis_curve = curve.to_basis(basis)
basis_curve.plot(axes = plt.gca(), color="red")






plt.gca().get_lines()[-1].remove()

from skfda.preprocessing.dim_reduction import FPCA
from skfda.exploratory.visualization import FPCAPlot

dataset = skfda.datasets.fetch_growth()
fd = dataset['data']
y = dataset['target']
fd.plot()
fpca_discretized = FPCA(n_components=2)
fpca_discretized.fit(fd)
fpca_discretized.components_.plot()
basis = skfda.representation.basis.BSplineBasis(n_basis=7)
basis_fd = fd.to_basis(basis)
basis_fd.plot()
fpca = FPCA(n_components=2)
fpca.fit(basis_fd)
fpca.components_.plot()

FPCAPlot(
    basis_fd.mean(),
    fpca.components_,
    factor=30,
    fig=plt.figure(figsize=(6, 2 * 4)),
    n_rows=2,
).plot()


from skfda.misc.metrics import l2_distance, l2_norm
from skfda.preprocessing.dim_reduction import FPCA


target = data.loc[data['asset']=="VF-2_2", "0.0":]
np.random.seed(0)
train_ind = target.index[0] + np.random.choice(27, 25, replace=False)
train = target.loc[train_ind, :]
test = pd.concat([target.drop(train_ind), data.loc[data['asset']!="VF-2_2", "0.0":]])

train_set = skfda.representation.FDataGrid(train.values, train.columns.astype(float))
test_set = skfda.representation.FDataGrid(test.values, train.columns.astype(float))

test_set_labels = data['asset'].loc[test.index].values



fpca_clean = FPCA(n_components=10)
fpca_clean.fit(train_set)
train_set_hat = fpca_clean.inverse_transform(
    fpca_clean.transform(train_set)
)

err_train = l2_distance(
    train_set,
    train_set_hat
) / l2_norm(train_set)

test_set_hat = fpca_clean.inverse_transform(
    fpca_clean.transform(test_set)
)
err_test = l2_distance(
    test_set,
    test_set_hat
) / l2_norm(test_set)

err_thresh = err_train.max()*1.25

print('Flagged outliers: ')
print(test_set_labels[err_test >= err_thresh])
print('Flagged nonoutliers: ')
print(test_set_labels[err_test < err_thresh])

np.quantile(err_train, 0.99)