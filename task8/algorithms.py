import numpy as np
from sklearn.base import BaseEstimator
import hashlib

# This file contains classes for different pursuit algorithms


class SignalBagging:
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


class AtomBaggingBase(BaseEstimator):
    # Submodel base
    def __init__(
        self,
        K,
        select_atom_percent=0,
        random_seed=0,
        ignore_warning=False,
    ):
        """
        Args:

        This class is used to perform atom bagging
        Each object of this class is a submodel

        K (int): Number of iterations
        atom_bag_percent (float): Percentage of the original dictionary
        select_atom_percent (float): Percentage of the selected atoms
        random_seed (int): Random seed
        """

        self.K = K
        self.select_atom_percent = np.max([0, np.min([1, select_atom_percent])])
        self.atom_weak_select_flag = select_atom_percent > 0

        self.indices = []
        self.s = None
        self.phi = None
        self.a = None
        self.coefficients = None
        self.r = None

        self.random_seed = random_seed
        self.ignore_warning = ignore_warning

    def reset(self):
        self.indices = []
        self.s = None
        self.phi = None
        self.a = None
        self.coefficients = None
        self.r = None

    def fit(self, phi, s):
        return None

    def predict(self, phi_test):
        """
        Args:
        phi_test (numpy.ndarray): Test data

        Returns:
        numpy.ndarray: Predicted output
        """
        return (phi_test @ self.coefficients).reshape(-1, 1)

    def score(self, phi_test, s_test):
        # return self.coefficients
        s_pred = (phi_test @ self.coefficients).reshape(-1, 1)
        pred_mse = np.mean((s_pred - s_test) ** 2)
        return pred_mse

    def input_coefficients(self, coefficients):
        self.coefficients = coefficients

    def update_seed(self, random_seed):
        self.random_seed = random_seed




class OMP_Augmented(AtomBaggingBase):
    def __init__(
        self,
        K=10,
        atom_bag_percent=1,
        select_atom_percent=0,
        random_seed=0,
        ignore_warning=False
    ):
        self.K = K
        self.random_seed = random_seed
        self.ignore_warning = ignore_warning

        self.select_atom_percent = np.max([0, np.min([1, select_atom_percent])])
        self.atom_weak_select_flag = select_atom_percent > 0
        self.atom_bag_percent = np.max([0, np.min([1, atom_bag_percent])])


        self.indices = []
        self.s = None
        self.phi = None
        self.a = None
        self.coefficients = None
        self.r = None
        self.coefficients_matrix_per_k = None


    def fit(self, phi, s):
        """
        Args:
        s (numpy.ndarray): Input signal
        phi (numpy.ndarray): Dictionary
        """
        self.reset()
        if s.ndim == 1:
            self.s = s.reshape(-1, 1)
        else:
            self.s = s
        self.phi = phi
        self.a = np.zeros_like(self.s)
        self.coefficients = np.zeros(phi.shape[1])
        self.r = self.s.copy()

        self.coefficients_matrix_per_k = np.zeros((phi.shape[1], self.K))
        if self.random_seed is not None:
            np.random.seed(self.random_seed)

        for i in range(self.K):
            inner_products = (phi.T @ self.r).flatten()
            # so that we will not select the same atom
            inner_products[self.indices] = 0
            if self.atom_weak_select_flag:
                top_ind = np.argsort(np.abs(inner_products))[::-1][
                    : int(phi.shape[1] * self.select_atom_percent)
                ]
                # randomly select one atom
                lambda_k = np.random.choice(top_ind)
            else:
                lambda_k = np.argmax(np.abs(inner_products))

            # Ordinary least squares
            X = phi[:, self.indices + [lambda_k]]

            try:
                betas = np.linalg.inv(X.T @ X) @ X.T @ self.s
            except:
                if not self.ignore_warning:
                    print("Singular matrix encountered in OMP")
                break

        self.coefficients = betas.flatten()

        # Update Projection
        self.a = self.predict(self.phi)

        # Update Residual
        self.r = self.s - self.a

        return self.a, self.coefficients

    def reset(self):
        super().reset()
        self.coefficients_matrix_per_k = None
        self.error_series = []

    def set_K(self, K):
        self.K = K


class BOMP(AtomBaggingBase):
    def __init__(
        self,
        N_bag = 10,
        K = 10,
        signal_bag_percent=0.7,
        atom_bag_percent=1,
        select_atom_percent=0,
        replace_flag=True,
        agg_func="weight",
        random_seed=None,
        ignore_warning=False,
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
        self.K = K
        self.signal_bag_percent = signal_bag_percent
        self.atom_bag_percent = atom_bag_percent
        self.select_atom_percent = select_atom_percent
        self.replace_flag = replace_flag
        self.agg_func = agg_func
        self.random_seed = random_seed
        self.ignore_warning = ignore_warning
        self.s = None
        self.phi = None
        self.tmpPursuitModel = OMP_Augmented(
            self.K,
            self.atom_bag_percent,
            self.select_atom_percent,
            self.replace_flag,
        )
        self.SignalBagging = None
        self.coefficients = None
        self.coefficients_matrix_per_bag = None
        self.a = None



    def agg_weight_with_error(self, coefficients_matrix, error_series):
        """
        This function is used to aggregate the coefficients with the inverse of the mean squared error

        Args:
        coefficients_matrix (numpy.ndarray): Matrix of coefficients
        error_series (numpy.ndarray): Error series
        """
        # Calculate the weight

        weight = 1 / error_series
        weight = weight / np.sum(weight)

        # Calculate the weighted average

        return coefficients_matrix @ weight


    def agg_weight_with_avg(self, coefficients_matrix):
        """
        This function is used to aggregate the coefficients with the average of the coefficients

        Args:
        coefficients_matrix (numpy.ndarray): Matrix of coefficients
        """

        # Calculate the average

        return np.mean(coefficients_matrix, axis=1)
    
    def agg_weight_with_count(self, coefficients_matrix, col_idx_bag):

        counted_array = np.array(
            np.unique(np.concatenate(col_idx_bag), return_counts=True)
        )
        temp_coefficients_matrix = self.coefficients_cubic.sum(axis=0)
        counted_array = counted_array[:,np.argsort(counted_array[0])]
        if (counted_array.shape[1] < self.phi.shape[1]):
            self.coefficients_matrix_per_bag[counted_array[0, :], :] = ((temp_coefficients_matrix[counted_array[0, :], :]).T/ counted_array[1, :]).T
        else:
            self.coefficients_matrix_per_bag = ((temp_coefficients_matrix).T/ counted_array[1, :]).T

        raise NotImplementedError
        pass # TODO

    def fit(self, phi, s):
        """
        Args:
        s (numpy.ndarray): Input signal
        phi (numpy.ndarray): Dictionary
        """

        self.reset()
        if self.random_seed is not None:
            np.random.seed(self.random_seed)

        self.s = s
        self.phi = phi
        self.SignalBagging = SignalBagging(
            self.N_bag,
            self.signal_bag_percent,
            self.replace_flag,
            np.random.randint(10000),
        )
        self.SignalBagging.fit(self.phi, self.s)
        s_bag = self.SignalBagging.s_bag
        phi_bag = self.SignalBagging.phi_bag
        col_idx_bag = self.SignalBagging.col_idx_bag
        self.coefficients_matrix_per_bag = np.zeros((phi.shape[1], self.N_bag))
        self.error_series = np.zeros(self.N_bag)
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
        if self.agg_func != "weight":
            for i in range(self.N_bag):
                sub_s = s_bag[i]
                sub_phi = phi_bag[i]
                row_sub_idx = col_idx_bag[i]
                self.tmpPursuitModel = OMP_Augmented(
                    self.K,
                    self.atom_bag_percent,
                    self.select_atom_percent,
                    np.random.randint(10000)
                )
                self.tmpPursuitModel.fit(sub_phi, sub_s)
                sub_coefficients = self.tmpPursuitModel.coefficients
                self.tmpPursuitModel.reset()
                self.coefficients_matrix_per_bag[row_sub_idx, i] = sub_coefficients

            for i in range(self.N_bag):
                sub_s = s_bag[i]
                sub_phi = phi_bag[i]
                row_sub_idx = col_idx_bag[i]
                self.tmpPursuitModel = OMP_Augmented(
                    self.K_lst,
                    self.select_atom_percent,
                    np.random.randint(10 * np.max(self.Bag_lst)),
                    self.ignore_warning,
                )
                self.tmpPursuitModel.fit(sub_phi, sub_s)
                sub_coefficients = self.tmpPursuitModel.coefficients
                self.tmpPursuitModel.reset()
                self.coefficients_matrix_per_bag[row_sub_idx, i] = sub_coefficients

                # calculate self.error_series using oob samples
                if self.signal_bag_percent < 1:
                    row_oob_idx = np.setdiff1d(np.arange(self.s.shape[0]), row_sub_idx)
                    phi_oob = phi[row_oob_idx, :]
                    s_oob = s[row_oob_idx, :]
                    oob_mse = np.mean((s_oob.ravel() - phi_oob @ sub_coefficients) ** 2)
                    self.error_series[i] = oob_mse
                else:
                    self.error_series[i] = np.mean((self.s.ravel() - self.phi @ sub_coefficients) ** 2)


        if self.agg_func == "weight":
            self.coefficients = self.agg_weight_with_error(self.coefficients_matrix_per_bag, self.error_series)
        elif self.agg_func == "count":
            self.coefficients = self.agg_weight_with_count(self.coefficients_matrix_per_bag, col_idx_bag)
        elif self.agg_func == "avg":
            self.coefficients = self.agg_weight_with_avg(self.coefficients_matrix_per_bag, col_idx_bag)
        else:
            self.coefficients = self.agg_weight_with_count(self.coefficients_matrix_per_bag, col_idx_bag)



        # Update Projection
        self.a = self.predict(self.phi)

        # Update Residual
        self.r = self.s - self.a


    def reset(self):
        """
        This function is used to reset the model
        """
        super().reset()
        self.coefficients_matrix_per_bag = None
        self.error_series = None
        self.coefficients = None
        self.a = None

    def set_N_bag(self, N_bag):
        """
        This function is used to set the number of bags
        """
        self.N_bag = N_bag

    
    def set_K(self, K):
        """
        This function is used to set the number of atoms
        """
        self.K = K

    def get_params(self, deep=True):
    # This assumes all parameters are primitives
        return {
            "N_bag": self.N_bag,
            "K": self.K,
            "signal_bag_percent": self.signal_bag_percent,
            "atom_bag_percent": self.atom_bag_percent,
            "select_atom_percent": self.select_atom_percent,
            "replace_flag": self.replace_flag,
            "agg_func": self.agg_func,
            "random_seed": self.random_seed,
            "ignore_warning": self.ignore_warning,
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self