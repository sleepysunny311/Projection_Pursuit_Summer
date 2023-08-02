import numpy as np
from sklearn.base import BaseEstimator

# This file contains classes for different pursuit algorithms


class SignalBagging:
    def __init__(
        self,
        N,
        signal_bag_percent=0.7,
        replace_flag=False,
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
        self.s_bag = []
        self.phi_bag = []
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
        if self.random_seed is not None:
            np.random.seed(self.random_seed)

        if self.signal_bag_percent < 1:
            for _ in range(self.N):
                row_indices = np.random.choice(
                    self.s.shape[0], num_samples, replace=self.replace_flag
                )
                s_tmp = self.s[row_indices]
                phi_tmp = self.phi[row_indices, :]
                self.s_bag.append(s_tmp)
                self.phi_bag.append(phi_tmp)
                self.row_idx_bag.append(row_indices)
        else:
            self.s_bag = [self.s] * self.N
            self.phi_bag = [self.phi] * self.N
            self.row_idx_bag = [np.arange(self.s.shape[0])] * self.N
            
        return self.s_bag, self.phi_bag


class AtomBaggingBase(BaseEstimator):
    # Submodel base
    def __init__(
        self,
        K,
        atom_bag_percent=1,
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
        self.atom_bag_percent = np.max([0, np.min([1, atom_bag_percent])])

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


class AtomBaggingMatchingPursuit(AtomBaggingBase):
    def __init__(self, K, atom_bag_percent=1, select_atom_percent=0, random_seed=None):
        """
        This class is used to perform atom bagging with matching pursuit

        Args:
        K (int): Number of iterations
        atom_bag_percent (float): Percentage of the original dictionary
        select_atom_percent (float): Percentage of the selected atoms
        random_seed (int): Random seed
        """

        super().__init__(K, atom_bag_percent, select_atom_percent, random_seed)

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

        if self.random_seed is not None:
            np.random.seed(self.random_seed)

        for i in range(self.K):
            inner_products = (phi.T @ self.r).flatten()
            dropping_indices = np.random.choice(
                phi.shape[1],
                int(phi.shape[1] * (1 - self.atom_bag_percent)),
                replace=False,
            )
            inner_products[dropping_indices] = 0
            if self.atom_weak_select_flag:
                top_ind = np.argsort(np.abs(inner_products))[::-1][
                    : int(phi.shape[1] * self.select_atom_percent)
                ]
                # randomly select one atom
                lambda_k = np.random.choice(top_ind)
            else:
                lambda_k = np.argmax(np.abs(inner_products))
            self.indices.append(lambda_k)
            phi_lambda_norm_sq = np.linalg.norm(phi[:, lambda_k]) ** 2
            self.coefficients[lambda_k] = (
                self.coefficients[lambda_k] + inner_products[lambda_k] / phi_lambda_norm_sq
            )
            self.a += (inner_products[lambda_k] / phi_lambda_norm_sq) * phi[:, lambda_k].reshape(-1, 1)
            self.r = self.s - self.a
        return self.a, self.coefficients


# class OMP_Augmented(AtomBaggingBase):
#     def __init__(
#         self,
#         K,
#         select_atom_percent=0,
#         random_seed=None,
#         ignore_warning=False,
#     ):
#         super().__init__(K, select_atom_percent, random_seed, ignore_warning)

#     def fit(self, phi, s):
#         self.reset()

#         """
#         Args:
#         s (numpy.ndarray): Input signal
#         phi (numpy.ndarray): Dictionary
#         """
#         if s.ndim == 1:
#             self.s = s.reshape(-1, 1)
#         else:
#             self.s = s
#         self.phi = phi
#         self.a = np.zeros_like(self.s)
#         self.coefficients = np.zeros(phi.shape[1])
#         self.r = self.s.copy()

#         if self.random_seed is not None:
#             np.random.seed(self.random_seed)

#         for i in range(self.K):
#             inner_products = (phi.T @ self.r).flatten()
#             inner_products[self.indices] = 0
#             if self.atom_weak_select_flag:
#                 top_ind = np.argsort(np.abs(inner_products))[::-1][
#                     : int(phi.shape[1] * self.select_atom_percent)
#                 ]
#                 # randomly select one atom
#                 lambda_k = np.random.choice(top_ind)
#             else:
#                 lambda_k = np.argmax(np.abs(inner_products))

#             # Ordinary least squares
#             X = phi[:, self.indices + [lambda_k]]

#             try:
#                 betas = np.linalg.inv(X.T @ X) @ X.T @ self.s
#             except:
#                 if not self.ignore_warning:
#                     print("Singular matrix encountered in OMP")
#                 break
#             # Update indices
#             self.indices.append(lambda_k)

#             # Update Coefficients
#             self.coefficients = np.zeros(phi.shape[1])
#             self.coefficients[self.indices] = betas.flatten()

#             # Update residual
#             self.r = self.s - X @ betas

#             # Update Projection
#             self.a = X @ betas
#         return self.a, self.coefficients


class BMP(AtomBaggingBase):
    def __init__(
        self,
        N_bag=10,
        K=10,
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
        self.tmpPursuitModel = AtomBaggingMatchingPursuit(
            self.K,
            self.atom_bag_percent,
            self.select_atom_percent,
            self.random_seed
        )
        self.SignalBagging = None
        self.coefficients_lst = []
        self.mse_lst = []
        # self.indices_lst = []
        self.coefficients = None
        self.a = None

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

    def fit(self, phi, s):
        """
        Args:
        s (numpy.ndarray): Input signal
        phi (numpy.ndarray): Dictionary
        """
        self.reset()
        self.coefficients_lst = []
        self.mse_lst = []

        self.s = s
        self.phi = phi
        self.SignalBagging = SignalBagging(
            self.N_bag,
            self.signal_bag_percent,
            self.replace_flag,
            self.random_seed
        )
        self.SignalBagging.fit(self.phi, self.s)

        s_bag = self.SignalBagging.s_bag
        phi_bag = self.SignalBagging.phi_bag
        row_idx_bag = self.SignalBagging.row_idx_bag

        for i in range(self.N_bag):
            sub_s = s_bag[i]
            sub_phi = phi_bag[i]
            row_sub_idx = row_idx_bag[i]
            
            self.tmpPursuitModel = AtomBaggingMatchingPursuit(
                self.K,
                self.atom_bag_percent,
                self.select_atom_percent,
                None
            )
            self.tmpPursuitModel.fit(sub_phi, sub_s)
            sub_coefficients = self.tmpPursuitModel.coefficients
            # calculate mse_lst using oob samples
            if self.signal_bag_percent < 1:
                row_oob_idx = np.setdiff1d(np.arange(self.s.shape[0]), row_sub_idx)
                phi_oob = phi[row_oob_idx, :]
                s_oob = s[row_oob_idx, :]
                oob_mse = np.mean((s_oob.ravel() - phi_oob @ sub_coefficients) ** 2)
                self.mse_lst.append(oob_mse)
            else:
                self.mse_lst.append(np.mean((sub_s.ravel() - sub_phi @ sub_coefficients) ** 2))
            self.coefficients_lst.append(sub_coefficients)
            # self.indices_lst.append(self.tmpPursuitModel.indices)

        if self.agg_func == "weight":
            self.coefficients = self.agg_weight_with_error(self.coefficients_lst, self.mse_lst)
        else:
            self.coefficients = self.agg_weight_with_avg(self.coefficients_lst)
        self.a = self.phi @ self.coefficients


class MP:
    def __init__(self, K):
        """
        This class is used to perform atom bagging with matching pursuit

        Args:
        K (int): Number of iterations
        atom_bag_percent (float): Percentage of the original dictionary
        select_atom_percent (float): Percentage of the selected atoms
        random_seed (int): Random seed
        """
        self.K = K
        self.indices = []
        self.s = None
        self.phi = None
        self.a = None
        self.coefficients = None
        self.r = None


    def reset(self):
        self.indices = []
        self.s = None
        self.phi = None
        self.a = None
        self.coefficients = None
        self.r = None
        
    def score(self, phi_test, s_test):
        # return self.coefficients
        s_pred = (phi_test @ self.coefficients).reshape(-1, 1)
        pred_mse = np.mean((s_pred - s_test) ** 2)
        return pred_mse
    
    def predict(self, phi_test):
        return (phi_test @ self.coefficients).reshape(-1, 1)
        
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

        if self.random_seed is not None:
            np.random.seed(self.random_seed)

        for i in range(self.K):
            inner_products = (phi.T @ self.r).flatten()
            lambda_k = np.argmax(np.abs(inner_products))
            self.indices.append(lambda_k)
            phi_lambda_norm_sq = np.linalg.norm(phi[:, lambda_k]) ** 2
            self.coefficients[lambda_k] = (
                self.coefficients[lambda_k] + inner_products[lambda_k] / phi_lambda_norm_sq
            )
            self.a += (inner_products[lambda_k] / phi_lambda_norm_sq) * phi[:, lambda_k].reshape(-1, 1)
            self.r = self.s - self.a
        return self.a, self.coefficients