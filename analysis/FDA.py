import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import skfda
from skfda.representation.basis import BSpline
from tqdm import tqdm


def figure():

    manager = plt.get_current_fig_manager()
    manager.window.wm_geometry("-5000+1")
    plt.pause(0.5)
    manager.window.state('zoomed')
    plt.show(block=False)


# Function to check if a string can be converted to a float
def is_convertible_to_float(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


plt.ion()
data = pd.read_csv(r'data\Kernels\2023_02_07\VF_merged.csv')
# convert columns names  to float
sample = data.iloc[0,1:].astype(float)
sample.index = sample.index.astype(float)

class Sample(pd.DataFrame):

    def __init__(self, data):
        super().__init__(data)
        self.labels = self.iloc[:,0]
        self.columns = [float(col) if is_convertible_to_float(col) else col for col in self.columns]
        self.numeric = self.iloc[:,1:].astype(float)
        self._top_bottom = None  # Initialize the private attribute
        self._top_bottom_filled = None # Initialize the private attribute
        self.grid = self.numeric.columns
        

    @property
    def top_bottom(self):
        ''' Property returning the indexes of top and bottom of a sample in 0.01 windows'''
        if self._top_bottom is None:
            self._calculate_top_bottom()
        return self._top_bottom

    def _calculate_top_bottom(self):
        ''' Function to find the indexes of top and bottom of a sample in 0.01 windows'''
        positive = self.numeric[self.numeric > 0]
        negative = self.numeric[self.numeric < 0]
        self._top_bottom = {key:pd.DataFrame() for key in ['top', 'bottom']}
        final_n = round(self.numeric.columns[-1], 2)
        for var, key in zip([positive, negative], self._top_bottom):
            for n in tqdm(np.round(np.arange(0, final_n, 0.01), 2), desc=f"Processing {key}"):
                try:
                    chunk = var.loc[:, n:round(n + 0.01, 2)]
                    ms = chunk.idxmax(axis=1) if key == "top" else chunk.idxmin(axis=1)
                    concatenated = pd.concat([self._top_bottom[key].iloc[:, -1:], ms], axis=1, ignore_index=True)
                    unique = concatenated.apply(lambda row: pd.Series(row.unique()), axis=1)
                    self._top_bottom[key] = pd.concat([self._top_bottom[key], unique.iloc[:, -1]], axis=1, ignore_index=True)
                except Exception as e:
                    pass
    
    @property
    def top_bottom_filled(self):
        ''' Property returning the indexes of top and bottom of a sample in 0.01 windows'''
        if self._top_bottom_filled is None:
            self._calculate_top_bottom_filled()
        return self._top_bottom_filled

    def _calculate_top_bottom_filled(self):
        '''Funtion to interpolate the missing x values'''
        self._top_bottom_filled = {'top': [], 'bottom': []}
        for key in self.top_bottom:
            lim = self.top_bottom[key].apply(lambda row: np.interp(self.grid, row.dropna(), self.numeric.loc[row.name, row.dropna()]), axis=1)
            self._top_bottom_filled[key] = pd.DataFrame(lim.values.tolist(), index=lim.index, columns=self.grid)
        return self._top_bottom_filled

sample = Sample(data)


limits = sample.top_bottom


figure()

# plot top and bottom shapes
for i in range(2):
    for var in ['top', 'bottom']:
        col = limits[var].loc[i].dropna()
        plt.plot(col, sample.loc[i,col], marker=".")


limits  = sample.top_bottom_filled

# plot top and bottom shapes
for i in range(2):
    for var in ['top', 'bottom']:
        col = limits[var].loc[i]
        col.plot()


def to_basis(sample, lim:"top" or "bottom", plot = False):
    lim_dict = fit_missing(sample)

    grid = np.round(np.arange(0, sample.index[-1]+0.001, 0.001),3)
    y = lim_dict[lim]
    curve = skfda.FDataGrid(y, grid_points=grid) 
    basis = BSpline(knots=top)
    basis_curve = curve.to_basis(basis)
    if plot:
        basis_curve.plot(color="red")
        curve.plot(axes = plt.gca(), color = "blue", alpha = 0.5)
    return basis_curve



basis_curve = to_basis(sample, "bottom")




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