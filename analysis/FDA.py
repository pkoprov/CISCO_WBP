import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import skfda
from skfda.representation.basis import BSpline
from tqdm import tqdm
from skfda.misc.metrics import l2_distance, l2_norm
from skfda.preprocessing.dim_reduction import FPCA
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix
import seaborn as sns



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


def align_curves(fd):
    from skfda.preprocessing.registration import landmark_elastic_registration_warping
    extreme_pts = fd.data_matrix.argmax(axis=1).reshape(-1)
    val, count = np.unique(extreme_pts, return_counts=True)
    mode = val[count.argmax()]
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


    fd_dict = sample.FData()

    fd_dict_top = fd_dict['top']


    for label in labels:
        target_idx = sample.labels == label
        target = fd_dict_top[target_idx]

        np.random.seed(0)
        train_ind = target_idx.idxmax() + np.random.choice(target.shape[0], 25, replace=False)
        test_ind = sample.index.difference(train_ind)
        train = fd_dict_top[train_ind]
        train_y = sample.labels.loc[train_ind].values
        test = fd_dict_top[test_ind]


        target_curve_mean = train.mean()

        tcm = target_curve_mean.data_matrix.reshape(-1)
        top = target_curve_mean.grid_points[0][np.argsort(tcm)[-100:]]
        bottom = target_curve_mean.grid_points[0][np.argsort(tcm)[:100]]
        knots = np.concatenate([bottom, top])
        knots.sort()
        
        basis = BSpline(knots=knots)
        train_basis = train.to_basis(basis)
        test_basis = test.to_basis(basis)

        plt.figure()
        test_errors = l2_distance(train_basis.mean(), train_basis) / l2_norm(train_basis.mean())
        train_errors = l2_distance(train_basis.mean(), test_basis) / l2_norm(train_basis.mean())
        plt.plot(test_errors, 'o')
        plt.plot(train_errors, 'o')
        plt.pause(0.1)


    test_set_labels = sample.labels.loc[test_ind].values
