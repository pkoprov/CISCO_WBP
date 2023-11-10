import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import skfda
from skfda.representation.basis import BSpline
from tqdm import tqdm
from skfda.misc.metrics import l2_distance, l2_norm


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


def align_curves(fd, key = 'top'):
    from skfda.preprocessing.registration import landmark_elastic_registration_warping
    # get max value for each curve

    extreme_pts = np.argmax(fd.data_matrix, axis=1).reshape(-1) if key == 'top' else np.argmin(fd.data_matrix, axis=1).reshape(-1)
    # plt.plot(extreme_pts/1000, fd.data_matrix[range(fd.shape[0]), extreme_pts[0]], 'o')
    val, count = np.unique(extreme_pts//100, return_counts=True)
    mode = val[count.argmax()]*100
    lim = mode - 100, mode + 100
    landmarks = (fd.data_matrix.reshape(fd.shape[0],-1)[:, lim[0]:lim[1]].argmax(axis=1).reshape(-1,1) + lim[0])/1000
    warping = landmark_elastic_registration_warping(fd, landmarks)
    fd_registered = fd.compose(warping)
    return fd_registered


class Sample(pd.DataFrame):

    def __init__(self, data):
        super().__init__(data)
        self.labels = self.iloc[:,0]
        self.columns = [float(col) if is_convertible_to_float(col) else col for col in self.columns]
        self.numeric = self.iloc[:,1:].astype(float)
        self.numeric.columns = self.numeric.columns.astype(float)
        self._top_bottom = {}  # Initialize the private attribute
        self._top_bottom_filled = {} # Initialize the private attribute
        self.grid = self.numeric.columns
        self._FData = {}
        

    def top_bottom_idx(self,row = 'all', lim = 'all'):
        ''' Property returning the indexes of top and bottom of a sample in 0.01 windows'''
        match row:
            case 'all':
                row = range(self.shape[0])
            case int():
                row = [row]
            case list():
                pass
        match lim:
            case 'all':
                if not self._top_bottom:
                    self._calculate_top_bottom_idx(row, lim)
                else:
                    for key in ['top', 'bottom']:
                        if key not in self._top_bottom:
                            self._calculate_top_bottom_idx(row, key)
                        elif not all(i in self._top_bottom[key].index for i in row):
                            self._calculate_top_bottom_idx(row, key)
            case _:
                if lim not in self._top_bottom or not all(i in self._top_bottom[lim].index for i in row):
                    self._calculate_top_bottom_idx(row, lim)
                
        return self._top_bottom

    def _calculate_top_bottom_idx(self, row = 'all', lim = 'all'):
        ''' Function to find the indexes of top and bottom of a sample in 0.01 windows'''         
            
        df = self.numeric.loc[row]
        
        match lim:
            case 'all':
                for key in ['top', 'bottom']:
                    self._calculate_top_bottom_idx(row, key)
            case _:
                print(f"Calculating {lim} 10 indices")
                var = df[df > 0] if lim == 'top' else df[df < 0]
                var.fillna(0, inplace=True)
                self._top_bottom[lim] = pd.DataFrame()
                final_n = round(var.columns[-1], 2)

                # Create a binning series using `cut` to define bins every 0.01 units
                bin_labels = np.arange(0, final_n, 0.01)
                # Define bin edges and labels
                bin_edges = np.arange(0, final_n + 0.01, 0.01)  # Ensure the last bin edge is included
                bin_labels = np.round(bin_edges[:-1], 2)  # Labels should be one less than bin edges

                # Create bins
                bins = pd.cut(var.columns, bins=bin_edges, labels=bin_labels, include_lowest=True)
                # Group by the bins and apply the idxmax or idxmin function
                grouped = var.groupby(bins, axis = 1, observed=True)
                self._top_bottom[lim] = grouped.agg(lambda x: x.idxmax(axis=1) if lim == "top" else x.idxmin(axis=1))

    def top_bottom_filled(self, row = 'all', lim = 'all'):
        ''' Property returning the DataFrames of top and bottom of a sample with filled values'''
        if not self._top_bottom:
            self._calculate_top_bottom_filled(row , lim)
        return self._top_bottom_filled

    def _calculate_top_bottom_filled(self, row = 'all', lim = 'all'):
        '''Funtion to interpolate the missing x values'''
        row = [row] if isinstance(row, int) else row
        match lim:
            case 'all':
                for key in ['top', 'bottom']:
                    self._calculate_top_bottom_filled(row, key)
            case _:
                if lim not in self._top_bottom or not all(i in self._top_bottom[lim].index for i in row):
                    self.top_bottom_idx(row, lim)
        
        for key in self._top_bottom:
            # linearly interpolate and add 0 as a starting point
            lim = self._top_bottom[key].apply(lambda row: np.interp(self.grid, [0]+row.tolist(), [0] + self.numeric.loc[row.name,row.values].tolist()), axis=1)
            self._top_bottom_filled[key] = pd.DataFrame(lim.values.tolist(), index=lim.index, columns=self.grid)
        return self._top_bottom_filled
    

    def FData(self, row = 'all', lim = 'all', plot = False):

            '''Function to convert the top and bottom shapes to a FData representation'''
            match row:
                case 'all':
                    row = range(self.shape[0])
                case int():
                    row = [row]
                case list():
                    pass
            if lim not in self._top_bottom_filled or not all(i in self._top_bottom_filled[lim].index for i in row):
                self.top_bottom_filled(row, lim)

            y = self._top_bottom_filled[lim].loc[row]
            y = y.to_numpy().reshape(len(row), -1, 1)

            self._FData[lim] = skfda.FDataGrid(y, grid_points=self.grid)
            if plot:
                self._FData[lim].plot(axes = plt.gca(), alpha = 0.5)
                plt.legend(self._top_bottom_filled[lim].index)
                plt.title(f"{lim} shapes")
                   
            return self._FData


def plot_FPCA_results(sample, labels, label, train_ind, test_ind, y_test, test_errors_fpca, train_errors_fpca):
    plt.plot(train_ind,np.log(train_errors_fpca), 'o', color = 'blue', fillstyle='none', label = 'train')
    plt.plot(test_ind[y_test==label], np.log(test_errors_fpca[y_test==label]), 'o', color = 'blue', label = 'test target')
    plt.plot(test_ind[y_test!=label], np.log(test_errors_fpca[y_test!=label]), 'o', color = 'red', label = 'test other')
    
    err_thresh = train_errors_fpca.max()
    plt.hlines(np.log(err_thresh), 0, sample.shape[0], linestyle = '--', color = 'red', label = 'threshold')
    plt.title("L2 distance FPCA")
    plt.legend()
    plt.vlines([(sample.labels == label).idxmax() for label in labels],
                plt.gca().get_ylim()[0], plt.gca().get_ylim()[1], color = 'black',
                 linestyle = '--', label='label change')
    # add text to very label change
    for label in labels:
        plt.text((sample.labels == label).idxmax()+10, plt.gca().get_ylim()[1]-0.05, label)

    plt.pause(0.5)


def calculate_errors(train_set_basis, test_set_basis, fpca_clean):
    train_scores = fpca_clean.transform(train_set_basis)
    test_scores = fpca_clean.transform(test_set_basis)
    train_set_hat_basis = fpca_clean.inverse_transform(train_scores)
    test_set_hat_basis = fpca_clean.inverse_transform(test_scores)
    print("Calculating l2 distance for test data")
    test_errors_fpca = l2_distance(test_set_basis, test_set_hat_basis) / l2_norm(test_set_basis)
    print("Calculating l2 distance for train data")
    train_errors_fpca = l2_distance(train_set_basis, train_set_hat_basis) / l2_norm(train_set_basis)
    return test_errors_fpca,train_errors_fpca

if __name__ == "__main__":
    data = pd.read_csv(r'data\Kernels\2023_02_07\Prusa_merged.csv')

    sample = Sample(data)
    labels = sample.labels.unique()
    fd_dict = sample.FData()

    for label in labels:
        np.random.seed(0)
        # Define train and test indices
        target_idx = sample.index[sample.labels == label]
        train_ind = target_idx[0]+np.random.choice(target_idx.shape[0], 25, replace=False)
        test_ind = sample.index.difference(train_ind)
        y_test = sample.labels.loc[test_ind].values
        
        plt.figure()
        for n, key in enumerate(['top', 'bottom']):
            train = fd_dict[key][train_ind]
            test = fd_dict[key][test_ind]
            target_curve_mean = train.mean()
            tcm = target_curve_mean.data_matrix.reshape(-1)
            knots = find_extreme_grid(tcm, key = key) 

            basis = BSpline(knots=knots)
            print(f"Fitting basis to {key} train data")
            train_basis = train.to_basis(basis)
            print(f"Fitting basis to {key} test data")
            test_basis = test.to_basis(basis)      
            test_errors = l2_distance(test_basis,train_basis.mean())/l2_norm(train_basis.mean())
            train_errors = l2_distance(train_basis,train_basis.mean())/l2_norm(train_basis.mean())
            
            plt.subplot(2,1,n+1)
            plot_FPCA_results(sample, labels, label, train_ind, test_ind, y_test, test_errors, train_errors)