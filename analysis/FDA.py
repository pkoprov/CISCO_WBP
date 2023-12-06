import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import skfda
from skfda.representation.basis import BSpline
from tqdm import tqdm
from skfda.misc.metrics import l2_distance, l2_norm
import warnings

# Suppress all UserWarnings
warnings.filterwarnings('ignore', category=UserWarning)


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


def find_extreme_grid(array: np.array, key='top'):
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
    grid = np.array(extreme)/1000
    return grid


def align_curves(fd, key='top'):
    from skfda.preprocessing.registration import landmark_elastic_registration_warping
    # get max value for each curve

    extreme_pts = np.argmax(fd.data_matrix, axis=1).reshape(
        -1) if key == 'top' else np.argmin(fd.data_matrix, axis=1).reshape(-1)
    # plt.plot(extreme_pts/1000, fd.data_matrix[range(fd.shape[0]), extreme_pts[0]], 'o')
    val, count = np.unique(extreme_pts//100, return_counts=True)
    mode = val[count.argmax()]*100
    lim = mode - 100, mode + 100
    landmarks = (fd.data_matrix.reshape(
        fd.shape[0], -1)[:, lim[0]:lim[1]].argmax(axis=1).reshape(-1, 1) + lim[0])/1000
    warping = landmark_elastic_registration_warping(fd, landmarks)
    fd_registered = fd.compose(warping)
    return fd_registered


class Sample(pd.DataFrame):

    def __init__(self, data):
        super().__init__(data)
        self.labels = self.iloc[:, 0]
        self.columns = [float(col) if is_convertible_to_float(
            col) else col for col in self.columns]
        self.numeric = self.iloc[:, 1:].astype(float)
        self.numeric.columns = self.numeric.columns.astype(float)
        self._top_bottom = {}  # Initialize the private attribute
        self._top_bottom_filled = {}  # Initialize the private attribute
        self.grid = self.numeric.columns
        self._FData = {}

    def top_bottom_idx(self, row='all', lim='all'):
        if row == 'all':
            row = range(self.shape[0])
        elif isinstance(row, int):
            row = [row]

        if lim == 'all':
            keys_to_process = ['top', 'bottom']
        else:
            keys_to_process = [lim]

        for key in keys_to_process:
            if key not in self._top_bottom or not all(i in self._top_bottom[key].index for i in row):
                self._calculate_top_bottom_idx(row, key)

        return self._top_bottom

    def _calculate_top_bottom_idx(self, row, lim):
        print(f"Calculating {lim} {row} indices")
        # Extract the relevant part of the DataFrame as a NumPy array
        df = self.numeric.loc[row]
        var = df.clip(lower=0) if lim == 'top' else df.clip(upper=0)
        df = var.to_numpy()
        final_n = round(self.numeric.columns[-1], 2)

        # Create bin edges and labels
        bin_edges = np.arange(0, final_n + 0.01, 0.01)
        bin_labels = np.round(bin_edges[:-1], 2)

        # Assign each column to a bin
        col_to_bin = np.digitize(
            self.numeric.columns.astype(float), bin_edges) - 1

        # Prepare the result array
        result = np.full((df.shape[0], bin_edges.shape[0] - 1), np.nan)

        # Apply the computation for each bin
        for i, _ in enumerate(bin_labels):
            mask = col_to_bin == i
            if not np.any(mask):
                continue

            masked_df = df[:, mask]

            if lim == 'top':
                idx_max = np.nanargmax(df[:, mask], axis=1)
            else:  # 'bottom'
                idx_min = np.nanargmin(df[:, mask], axis=1)

            # Convert relative indices to absolute column indices
            abs_idx = np.where(mask)[0][idx_max if lim == 'top' else idx_min]
            for row_idx in range(df.shape[0]):
                # Check if all values in the row slice are NaN
                if (masked_df[row_idx] == 0).all():
                    continue
                result[row_idx, i] = abs_idx[row_idx]

        # Convert the result back to DataFrame
        self._top_bottom[lim] = pd.DataFrame(
            result/1000, index=self.numeric.loc[row].index, columns=bin_labels)

    def top_bottom_filled(self, row='all', lim='all'):
        if lim not in self._top_bottom or not all(i in self._top_bottom[lim].index for i in row):
            self._calculate_top_bottom_filled(row, lim)
        return self._top_bottom_filled

    def _calculate_top_bottom_filled(self, row, lim):
        print(f"Calculating {lim} {row} filled")
        if row == 'all':
            row_indices = np.arange(self.shape[0])
        elif isinstance(row, int):
            row_indices = np.array([row])
        else:  # row is a list
            row_indices = np.array(row)

        if lim == 'all':
            keys_to_process = ['top', 'bottom']
        else:
            keys_to_process = [lim]

        for key in keys_to_process:
            if key not in self._top_bottom or not np.all(np.isin(row_indices, self._top_bottom[key].index)):
                self.top_bottom_idx(row, key)

            # Perform interpolation using numpy
            grid_points = self.grid.to_numpy()
            interpolated_results = []

            for row_idx in row_indices:
                if row_idx in self._top_bottom[key].index and row_idx in self.numeric.index:
                    # Get valid indices and corresponding values
                    valid_indices = self._top_bottom[key].loc[row_idx].dropna()
                    x_points = valid_indices.to_numpy()
                    y_points = self.numeric.loc[row_idx,
                                                valid_indices.values].to_numpy()
                    # Perform interpolation
                    interpolated = np.interp(grid_points, x_points, y_points)
                    interpolated_results.append(interpolated)

            self._top_bottom_filled[key] = pd.DataFrame(
                interpolated_results, index=self.numeric.index[row_indices], columns=self.grid)

    def FData(self, row='all', lim='all', plot=False):
        if row == 'all':
            row = range(self.shape[0])
        elif isinstance(row, int):
            row = [row]

        for key in ['top', 'bottom'] if lim == 'all' else [lim]:
            if key not in self._top_bottom_filled or not all(i in self._top_bottom_filled[key].index for i in row):
                self.top_bottom_filled(row, key)
            print(f"Converting {key} {row} to FData")
            y = self._top_bottom_filled[key].loc[row].to_numpy().reshape(
                len(row), -1, 1)
            self._FData[key] = skfda.FDataGrid(y, grid_points=self.grid)

            if plot:
                self._FData[key].plot(axes=plt.gca(), alpha=0.5)
                plt.legend(self._top_bottom_filled[key].index)
                plt.title(f"{key} shapes")

        return self._FData


def plot_FPCA_results(sample, labels, label, train_ind, test_ind, y_test, test_errors_fpca, train_errors_fpca):
    plt.plot(train_ind, np.log(train_errors_fpca), 'o',
             color='blue', fillstyle='none', label='train')
    plt.plot(test_ind[y_test == label], np.log(
        test_errors_fpca[y_test == label]), 'o', color='blue', label='test target')
    plt.plot(test_ind[y_test != label], np.log(
        test_errors_fpca[y_test != label]), 'o', color='red', label='test other')

    err_thresh = train_errors_fpca.max()
    plt.hlines(np.log(err_thresh), 0,
               sample.shape[0], linestyle='--', color='red', label='threshold')
    plt.title("L2 distance FPCA")
    plt.legend()
    plt.vlines([(sample.labels == label).idxmax() for label in labels],
               plt.gca().get_ylim()[0], plt.gca().get_ylim()[1], color='black',
               linestyle='--', label='label change')
    # add text to very label change
    for label in labels:
        plt.text((sample.labels == label).idxmax()+10,
                 plt.gca().get_ylim()[1]-0.05, label)

    plt.pause(0.5)


if __name__ == "__main__":
    data = pd.read_csv(r'data\Kernels\2023_02_07\Prusa_merged.csv')

    sample = Sample(data)
    labels = sample.labels.unique()
    fd_dict = sample.FData()

    for label in labels:
        np.random.seed(0)
        # Define train and test indices
        target_idx = sample.index[sample.labels == label]
        train_ind = target_idx[0] + \
            np.random.choice(target_idx.shape[0], 25, replace=False)
        test_ind = sample.index.difference(train_ind)
        y_test = sample.labels.loc[test_ind].values

        plt.figure()
        for n, key in enumerate(['top', 'bottom']):
            train = fd_dict[key][train_ind]
            test = fd_dict[key][test_ind]
            target_curve_mean = train.mean()
            tcm = target_curve_mean.data_matrix.reshape(-1)
            knots = find_extreme_grid(tcm, key=key)

            basis = BSpline(knots=knots)
            print(f"Fitting basis to {key} train data")
            train_basis = train.to_basis(basis)
            print(f"Fitting basis to {key} test data")
            test_basis = test.to_basis(basis)
            test_errors = l2_distance(
                test_basis, train_basis.mean())/l2_norm(train_basis.mean())
            train_errors = l2_distance(
                train_basis, train_basis.mean())/l2_norm(train_basis.mean())

            plt.subplot(2, 1, n+1)
            plot_FPCA_results(sample, labels, label, train_ind,
                              test_ind, y_test, test_errors, train_errors)
