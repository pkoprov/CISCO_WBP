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


def find_extreme_grid(array:np.array, key = 'top'):
    ''' Function to find the indexes of top and bottom of a sample in 0.01 windows'''
    extreme = []
    for n in tqdm(np.round(np.arange(0, array.shape[0], 10), 2), desc=f"Processing {key}"):
                try:
                    chunk = array[n:n + 10]
                    pos = chunk.argmax() if key == "top" else chunk.argmin() 
                    if n == 0:
                        extreme.append(pos)
                        continue
                    pos = int(n+pos)                    
                    extreme.append(pos) if pos != extreme[-1] else extreme
                except Exception as e:
                    pass
    grid  = np.array(extreme)/1000
    return grid


class Sample(pd.DataFrame):

    def __init__(self, data):
        super().__init__(data)
        self.labels = self.iloc[:,0]
        self.columns = [float(col) if is_convertible_to_float(col) else col for col in self.columns]
        self.numeric = self.iloc[:,1:].astype(float)
        self._top_bottom = None  # Initialize the private attribute
        self._top_bottom_filled = None # Initialize the private attribute
        self.grid = self.numeric.columns
        self._FData = {}
        

    @property
    def top_bottom_idx(self):
        ''' Property returning the indexes of top and bottom of a sample in 0.01 windows'''
        if self._top_bottom is None:
            self._calculate_top_bottom_idx()
        return self._top_bottom

    def _calculate_top_bottom_idx(self):
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
        ''' Property returning the DataFrames of top and bottom of a sample with filled values'''
        if self._top_bottom_filled is None:
            self._calculate_top_bottom_filled()
        return self._top_bottom_filled

    def _calculate_top_bottom_filled(self):
        '''Funtion to interpolate the missing x values'''
        self._top_bottom_filled = {'top': [], 'bottom': []}
        for key in self.top_bottom_idx:
            # linearly interpolate and add 0 as a starting point
            lim = self.top_bottom_idx[key].apply(lambda row: np.interp(self.grid, [0]+row.dropna().tolist(), [0] + self.numeric.loc[row.name, row.dropna()].tolist()), axis=1)
            df = pd.DataFrame(lim.values.tolist(), index=lim.index, columns=self.grid)
            df[0] = 0
            self._top_bottom_filled[key] = df
        return self._top_bottom_filled
    

    def FData(self, row = 'all', lim = 'all', plot = False):

            '''Function to convert the top and bottom shapes to a FData representation'''
            match row:
                case int():
                    row = np.array(row).flatten().tolist()
                    self.FData(row = row, lim=lim, plot=plot)
                case 'all':
                    self.FData(row = [int(i) for i in self.index.values], lim=lim, plot=plot)
                case list():
                    match lim:
                        case 'all':                       
                            [self.FData(row = row, lim=key, plot=plot) for key in ['top', 'bottom']]
                        case _:
                            y = self.top_bottom_filled[lim].loc[row]
                            y = y.to_numpy().reshape(len(row), -1, 1)

                            curve = skfda.FDataGrid(y, grid_points=self.grid)
                            if plot:
                                curve.plot(axes = plt.gca(), color = "black", alpha = 0.5)
                            self._FData[lim] = curve

                    
            return self._FData

  



# limits_df  = sample.top_bottom_filled

# limits_df['top'].loc[:20].T.plot()
# plt.figure()
# limits_df['top'].loc[-20:].T.plot()

# # plot top and bottom shapes
# for i in range(40,42):
#     for var in ['top', 'bottom']:
#         col = limits_df[var].loc[i]
#         col.plot()



# fd_dict = sample.FData()
# fd = fd_dict['top']

# # plot mean of curves and curves
# target_curve = fd[target_idx]
# target_curve.plot(axes = plt.gca(), alpha = 0.1, color = "black")
# target_curve_mean = target_curve.mean()
# target_curve_mean.mean().plot(axes = plt.gca(),color = "blue")

# tcm = target_curve_mean.data_matrix.reshape(-1)
# top = find_extreme_grid(tcm, key = 'top')

# basis_top = BSpline(knots=top)
# basis_curve = target_curve.to_basis(basis_top)


# target_curve_mean.to_basis(basis_top).plot()
# for i in range(basis_curve.shape[0]):
#     basis_curve[i].plot(axes = plt.gca(), color = "red", linestyle = '--', alpha = 0.5)
#     plt.pause(0.1)
#     input()
#     plt.gca().get_lines()[-1].remove()


# plt.gca().get_lines()[-1].remove()

# from skfda.preprocessing.dim_reduction import FPCA
# from skfda.exploratory.visualization import FPCAPlot


# target_curve.plot()
# fpca_discretized = FPCA(n_components=2)
# fpca_discretized.fit(target_curve)
# fpca_discretized.components_.plot()
# basis = BSpline(knots=top)
# basis_fd = target_curve.to_basis(basis)
# basis_fd.plot()
# fpca = FPCA(n_components=2)
# fpca.fit(basis_fd)
# fpca.components_.plot()

# FPCAPlot(
#     basis_fd.mean(),
#     fpca.components_,
#     factor=30,
#     fig=plt.figure(figsize=(6, 2 * 4)),
#     n_rows=2,
# ).plot()


from skfda.misc.metrics import l2_distance, l2_norm
from skfda.preprocessing.dim_reduction import FPCA

data = pd.read_csv(r'data\Kernels\2023_02_07\Prusa_merged.csv')

sample = Sample(data)
labels = sample.labels.unique()


fd_dict = sample.FData()

fd_dict_top = fd_dict['top']

target_idx = sample.labels == labels[0]
target = fd_dict_top[target_idx]

np.random.seed(0)
train_ind = target_idx.idxmax() + np.random.choice(target.shape[0], 25, replace=False)
test_ind = sample.index.difference(train_ind)
train = fd_dict_top[train_ind]
train_y = sample.labels.loc[train_ind].values
test = fd_dict_top[test_ind]
test_y = sample.labels.loc[test_ind].values


target_curve_mean = train.mean()
tcm = target_curve_mean.data_matrix.reshape(-1)
top = find_extreme_grid(tcm, key = 'top')


basis = BSpline(knots=top)
train_set_basis = train.to_basis(basis)
test_set_basis = test.to_basis(basis)


test_set_labels = sample.labels.loc[test_ind].values


# fpca_clean = FPCA(n_components=train_set_basis.n_samples)


# fpca_clean.fit(train_set_basis)
# cumulative_variance = fpca_clean.explained_variance_ratio_.cumsum()
# plt.plot(cumulative_variance)
# plt.hlines(0.95, 0, fpca_clean.n_components, linestyle = '--', color = 'red')
# best_n_components = np.where(cumulative_variance > 0.95)[0][0]

# print(f"Best number of components: {best_n_components}")

# fpca_clean = FPCA(n_components=best_n_components)
fpca_clean = FPCA(n_components=3)
fpca_clean.fit(train_set_basis)

# fpca_clean.components_.plot()


train_scores = fpca_clean.transform(train_set_basis)
train_set_hat_basis = fpca_clean.inverse_transform(train_scores)

err_train_basis = l2_distance(
    train_set_basis,
    train_set_hat_basis
) / l2_norm(train_set_basis)

err_train_basis.min(), err_train_basis.max()

test_scores = fpca_clean.transform(test_set_basis)

test_set_hat_basis = fpca_clean.inverse_transform(test_scores)

err_test_basis = l2_distance(
    test_set_basis,
    test_set_hat_basis
) / l2_norm(test_set_basis)


err_thresh = err_train_basis.max()*1.25

plt.plot(err_test_basis, 'o')
plt.plot(err_train_basis, 'o')
plt.hlines(err_thresh, 0, len(err_test_basis), linestyle = '--', color = 'red')

err_test_basis/err_train_basis.max()

# plt.plot(range(len(err_test_basis)),err_test_basis, 'o')
# plt.plot(range(len(err_train_basis)),err_train_basis, 'o')

from sklearn.svm import OneClassSVM
from sklearn.model_selection import GridSearchCV

train = train_scores
test = test_scores


