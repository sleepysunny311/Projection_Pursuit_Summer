import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model import Lasso

# This file contains classes for different pursuit algorithms


class SignalAtomBagging:
    def __init__(
        self,
        N,
        signal_bag_percent=0.7,
        atom_bag_percent=0.7,
        replace_flag=True,
        random_seed=None,
    ):
        """ "
        This class is used to perform signal bagging

        Args:
        N (int): Number of bootstrap samples
        signal_bag_percent (float): Percentage of the original signal
        replace_flag (bool): Whether to sample with replacement
        random_seed (int): Random
        """
        self.s = None
        self.phi = None
        self.N = N
        self.replace_flag = replace_flag
        self.random_seed = random_seed
        self.signal_bag_percent = signal_bag_percent
        self.atom_bag_percent = atom_bag_percent
        self.s_bag = []
        self.phi_bag = []
        self.col_idx_bag = []
        self.row_idx_bag = []

    def fit(self, phi, s):
        """
        Args:
        s (numpy.ndarray): Input signal
        phi (numpy.ndarray): Dictionary
        """

        self.s = s
        self.phi = phi
        
        num_samples = int(self.signal_bag_percent * self.s.shape[0])
        num_atoms = int(self.atom_bag_percent * self.phi.shape[1])

        if self.random_seed is not None:
            np.random.seed(self.random_seed)

        if self.signal_bag_percent:
            for _ in range(self.N):
                row_indices = np.random.choice(
                    self.s.shape[0], num_samples, replace=self.replace_flag
                )
                col_indices = np.random.choice(
                    self.phi.shape[1], num_atoms, replace=False
                )
                s_tmp = self.s[row_indices]
                phi_tmp = self.phi[row_indices, :][:, col_indices]
                self.s_bag.append(s_tmp)
                self.phi_bag.append(phi_tmp)
                self.col_idx_bag.append(col_indices)
                self.row_idx_bag.append(row_indices)
        else:
            self.s_bag = [self.s] * self.N
            for _ in range(self.N):
                col_indices = np.random.choice(
                    self.phi.shape[1], num_atoms, replace=False
                )
                phi_tmp = self.phi[:, col_indices]
                self.phi_bag.append(phi_tmp)
                self.col_idx_bag.append(col_indices)

        return self.s_bag, self.phi_bag, self.col_idx_bag


class Bagging_Lasso(BaseEstimator):
    def __init__(
        self,
        N_bag= 10,
        alpha = 0.1,
        signal_bag_percent=0.7,
        atom_bag_percent=1,
        replace_flag=True,
        agg_func="weight",
        random_seed=None,
        ignore_warning=False,
        max_iter=10000, 
        tol=0.0001
    ):
        """
        Args:
        N (int): Number of submodels
        K (int): Number of iterations
        signal_bag_percent (float): Percentage of the original signal
        atom_bag_percent (float): Percentage of the original dictionary
        select_atom_percent (float): Percentage of the selected atoms
        replace_flag (bool): Whether to replace the samples
        agg_func (str): Aggregation function
        random_seed (int): Random seed
        """

        self.N_bag = N_bag
        self.signal_bag_percent = signal_bag_percent
        self.atom_bag_percent = atom_bag_percent
        self.alpha = alpha
        self.replace_flag = replace_flag
        self.agg_func = agg_func
        self.random_seed = random_seed
        self.ignore_warning = ignore_warning
        self.s = None
        self.phi = None
        self.max_iter = max_iter
        self.tol = tol
        self.tmpLassoModel = Lasso(alpha=self.alpha, max_iter=self.max_iter, tol=self.tol, random_state=self.random_seed)
        self.SignalBagging = None
        self.coefficients = None
        self.coefficients_lst = []
        self.a = None
        self.mse_lst = []
        
    def agg_weight_with_error(self, c_lst, mse_lst):
        """
        This function is used to aggregate the coefficients with the inverse of the mean squared error

        Args:
        c_lst (list): List of coefficients
        mse_lst (list): List of mean squared errors
        """
        # Calculate the weight
        mse_lst = np.array(mse_lst)
        weight = 1 / mse_lst
        weight = weight / np.sum(weight)

        # Calculate the weighted average
        tot = np.zeros_like(c_lst[0])
        for i in range(len(c_lst)):
            tot += c_lst[i] * weight[i]
        return tot

    def agg_weight_with_avg(self, c_lst):
        """
        This function is used to aggregate the coefficients with the inverse of the mean squared error

        Args:
        c_lst (list): List of coefficients
        """
        # Calculate the weighted average
        tot = np.zeros_like(c_lst[0])
        for i in range(len(c_lst)):
            tot += c_lst[i]
        return tot / len(c_lst)
    
    def reset(self):
        self.s = None
        self.a = None
        self.phi = None
        self.coefficients = None
        self.coefficients_lst = []
        self.mse_lst = []
        return
    
    def fit(self, phi, s):
        self.s = s
        self.phi = phi
        self.SignalBagging = SignalAtomBagging(
            self.N_bag,
            self.signal_bag_percent,
            self.atom_bag_percent,
            self.replace_flag,
            self.random_seed,
        )
        self.SignalBagging.fit(self.phi, self.s)

        s_bag = self.SignalBagging.s_bag
        phi_bag = self.SignalBagging.phi_bag
        col_idx_bag = self.SignalBagging.col_idx_bag
        row_idx_bag = self.SignalBagging.row_idx_bag

        for i in range(self.N_bag):
            sub_s = s_bag[i]
            sub_phi = phi_bag[i]
            col_sub_idx = col_idx_bag[i]
            row_sub_idx = row_idx_bag[i]
            
            self.tmpLassoModel = Lasso(
                alpha = self.alpha,
                max_iter = self.max_iter,
                tol = self.tol,
                random_state= np.random.randint(64),
            )
            
            self.tmpLassoModel.fit(sub_phi, sub_s)
            sub_coefficients = self.tmpLassoModel.coef_
            # calculate mse_lst using oob samples
            if self.signal_bag_percent < 1:
                row_oob_idx = np.setdiff1d(np.arange(self.s.shape[0]), row_sub_idx)
                phi_oob = phi[row_oob_idx, :][:, col_sub_idx]
                s_oob = s[row_oob_idx, :]
                oob_mse = np.mean((s_oob.ravel() - phi_oob @ sub_coefficients) ** 2)
                self.mse_lst.append(oob_mse)
            else:
                self.mse_lst.append(np.mean((sub_s.ravel() - sub_phi @ sub_coefficients) ** 2))
            real_sub_coefficients = np.zeros(phi.shape[1])
            real_sub_coefficients[col_sub_idx] = sub_coefficients
            self.coefficients_lst.append(real_sub_coefficients)
            # self.indices_lst.append(self.tmpPursuitModel.indices)

        if self.agg_func == "weight":
            self.coefficients = self.agg_weight_with_error(self.coefficients_lst, self.mse_lst)
        else:
            self.coefficients = self.agg_weight_with_avg(self.coefficients_lst)
        self.a = self.phi @ self.coefficients
        return self.a, self.coefficients
    
    def predict(self, phi_test):
        """
        Args:
        phi_test (numpy.ndarray): Test data

        Returns:
        numpy.ndarray: Predicted output
        """

        return (phi_test @ self.coefficients).reshape(-1, 1)

    def score(self, s_test, phi_test):
        s_pred = phi_test @ self.final_c
        pred_mse = np.mean((s_pred.ravel() - s_test.ravel()) ** 2)
        return pred_mse

    def input_coefficients(self, coefficients):
        self.coefficients = coefficients

    def update_seed(self, random_seed):
        self.random_seed = random_seed
        
    def get_params(self, deep: bool = True) -> dict:
        return super().get_params(deep)
    
    def set_params(self, **params):
        return super().set_params(**params)
        