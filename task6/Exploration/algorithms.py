import numpy as np
from sklearn.base import BaseEstimator

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

        return phi_test @ self.coefficients

    def score(self, s_test, phi_test):
        s_pred = phi_test @ self.final_c
        pred_mse = np.mean((s_pred - s_test) ** 2)
        return pred_mse

    def input_coefficients(self, coefficients):
        self.coefficients = coefficients

    def update_seed(self, random_seed):
        self.random_seed = random_seed


class AtomBaggingMatchingPursuit(AtomBaggingBase):
    def __init__(self, K, atom_bag_percent=1, select_atom_percent=0, random_seed=0):
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
            if self.atom_bag_flag:
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
            self.coefficients[lambda_k] = (
                self.coefficients[lambda_k] + inner_products[lambda_k]
            )
            self.a += inner_products[lambda_k] * phi[:, lambda_k].reshape(-1, 1)
            self.r = self.s - self.a
        return self.a, self.coefficients


class OMP_Augmented(AtomBaggingBase):
    def __init__(
        self, K_lst, select_atom_percent=0, random_seed=None, ignore_warning=False
    ):
        self.K_lst = K_lst
        self.random_seed = random_seed
        self.select_atom_percent = select_atom_percent
        if select_atom_percent == 0:
            self.atom_weak_select_flag = False

        self.indices = []
        self.coefficients = None
        self.ignore_warning = ignore_warning

        self.coefficients_matrix = None
        self.error_series = []

    def fit(self, phi, s):
        """
        Args:
        s (numpy.ndarray): Input signal
        phi (numpy.ndarray): Dictionary
        """
        self.reset()
        self.s = s
        self.phi = phi
        self.a = np.zeros_like(self.s)
        self.coefficients = np.zeros(phi.shape[1])
        self.r = self.s.copy()

        self.coefficients_matrix = np.zeros((phi.shape[1], len(self.K_lst)))

        if self.random_seed is not None:
            np.random.seed(self.random_seed)

        for k in range(np.max(self.K_lst)):
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

            # Update indices
            self.indices.append(lambda_k)

            # Update Coefficients
            self.coefficients = np.zeros(phi.shape[1])
            self.coefficients[self.indices] = betas.flatten()

            # Update Projection
            self.a = X @ betas

            # Update Residual
            self.r = self.s - self.a
            if (k+1) in self.K_lst:
                self.coefficients_matrix[:, self.K_lst.index(k+1)] = self.coefficients
                self.error_series.append(np.sum(self.r**2))

        minimal_k_index = np.argmin(self.error_series)

        # Update Coefficients

        self.coefficients = self.coefficients_matrix[:, minimal_k_index]

        # Update Projection
        self.a = phi @ self.coefficients

        # Update Residual
        self.r = self.s - self.a

        return self.a, self.coefficients

    def multi_score(self, phi_test, s_test):
        """
        Args:
        phi_test (numpy.ndarray): Test data
        s_test (numpy.ndarray): Test labels

        Returns:
        numpy.ndarray: Predicted output
        """

        test_score = []
        projection_matrix = phi_test @ self.coefficients_matrix
        residual_matrix = s_test.reshape(-1, 1) - projection_matrix
        test_score = np.mean(residual_matrix**2, axis=0)
        return test_score

    def reset(self):
        super().reset()
        self.coefficients_matrix = None
        self.error_series = []


class BOMP(AtomBaggingBase):
    def __init__(
        self,
        Bag_lst = list(range(1, 101)),
        K_lst=list(range(1, 21)),
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
        self.bag_lst = Bag_lst
        self.k_lst = K_lst
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
            K_lst, select_atom_percent, random_seed, ignore_warning
        )
        self.SignalBagging = None
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

        self.s = s
        self.phi = phi
        self.SignalBagging = SignalAtomBagging(
            np.max(self.bag_lst),
            self.signal_bag_percent,
            self.atom_bag_percent,
            self.replace_flag,
            self.random_seed,
        )
        self.SignalBagging.fit(self.phi, self.s)
        self.coefficients_matrix = None
        s_bag = self.SignalBagging.s_bag
        phi_bag = self.SignalBagging.phi_bag
        col_idx_bag = self.SignalBagging.col_idx_bag
        self.coefficients_cubic = np.zeros((np.max(self.bag_lst), phi.shape[1], len(self.k_lst)))
        self.bag_k_error_matrix = np.zeros((len(self.bag_lst), 3))


        if self.random_seed is not None:
            np.random.seed(self.random_seed)

        for i in range(np.max(self.bag_lst)):
            sub_s = s_bag[i]
            sub_phi = phi_bag[i]
            sub_idx = col_idx_bag[i]
            self.tmpPursuitModel = OMP_Augmented(
                self.k_lst,
                self.select_atom_percent,
                np.random.randint(10 * np.max(self.bag_lst)),
                self.ignore_warning,
            )
            self.tmpPursuitModel.fit(sub_phi, sub_s)
            real_sub_coefficients = np.zeros((phi.shape[1], len(self.k_lst)))
            real_sub_coefficients[sub_idx, :] = self.tmpPursuitModel.coefficients_matrix
            self.coefficients_cubic[i] = real_sub_coefficients
            self.tmpPursuitModel.reset()
            if (i+1) in self.bag_lst:
                counted_array = np.array(
                    np.unique(np.concatenate(col_idx_bag[: i + 1]), return_counts=True)
                )
                idx_count = counted_array[1, np.argsort(counted_array[0])]
                temp_coefficients_matrix = (
                    (self.coefficients_cubic.sum(axis=0).T) / idx_count
                ).T
                temp_projection_matrix = phi @ temp_coefficients_matrix
                temp_residual_matrix = s.reshape(-1, 1) - temp_projection_matrix
                temp_error_series = np.mean(temp_residual_matrix ** 2, axis=0)
                temp_optimal_idx = np.argmin(temp_error_series)
                self.bag_k_error_matrix[self.bag_lst.index(i+1), :] = np.array([i+1, self.k_lst[temp_optimal_idx], temp_error_series[temp_optimal_idx]])

        self.optimal_idx = np.argmin(self.bag_k_error_matrix[:, 2])

        self.optimal_k = int(self.bag_k_error_matrix[self.optimal_idx, 1])

        self.optimal_bag = int(self.bag_k_error_matrix[self.optimal_idx, 0])


        counted_array = np.array(
            np.unique(np.concatenate(col_idx_bag[: self.optimal_bag]), return_counts=True)
        )

        idx_count = counted_array[1, np.argsort(counted_array[0])]
        self.coefficients_matrix = (
            (self.coefficients_cubic[: self.optimal_bag].sum(axis=0).T) / idx_count
        ).T
        self.coefficients = self.coefficients_matrix[:, self.optimal_idx]

        # Update Projection
        self.a = phi @ self.coefficients

        # Update Residual
        self.r = self.s - self.a

        return self.a, self.coefficients

    def reset(self):
        """
        This function is used to reset the model
        """
        super().reset()
        self.coefficients_matrix = None
        self.coefficients_cubic = None
        self.error_series = []
        self.coefficients = None
        self.a = None
