import numpy as np
from sklearn.base import BaseEstimator

# This file contains classes for different pursuit algorithms


class SignalBagging:
    def __init__(
        self,
        N,
        signal_bag_percent=0.7,
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
        self.s_bag = []
        self.phi_bag = []

    def fit(self, phi, s):
        """
        Args:
        s (numpy.ndarray): Input signal
        phi (numpy.ndarray): Dictionary
        """

        self.s = s
        self.phi = phi

        if self.random_seed is not None:
            np.random.seed(self.random_seed)
        
        if self.signal_bag_percent:
            num_samples = int(self.signal_bag_percent * self.s.shape[0])
            for i in range(self.N):
                indices = np.random.choice(
                    self.s.shape[0], num_samples, replace=self.replace_flag
                )
                s_tmp = self.s[indices]
                phi_tmp = self.phi[indices, :]
                self.s_bag.append(s_tmp)
                self.phi_bag.append(phi_tmp)
        else:
            self.s_bag = [self.s] * self.N
            self.phi_bag = [self.phi] * self.N

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
        self.atom_bag_percent = np.max([0, np.min([1, atom_bag_percent])])
        self.select_atom_percent = np.max([0, np.min([1, select_atom_percent])])
        self.atom_bag_flag = atom_bag_percent < 1
        self.atom_weak_select_flag = select_atom_percent > 0

        self.indices = []
        self.s = None
        self.phi = None
        self.a = None
        self.coefficients = None
        self.r = None

        self.random_seed = random_seed
        self.ignore_warning = ignore_warning

    def fit(self, phi, s):
        pass

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


class OMP(AtomBaggingBase):
    def __init__(
        self, K, select_atom_percent=0, random_seed=None, ignore_warning=False
    ):
        self.K = K
        self.random_seed = random_seed
        self.select_atom_percent = select_atom_percent
        if select_atom_percent == 0:
            self.atom_weak_select_flag = False

        self.indices = []
        self.coefficients = None
        self.ignore_warning = ignore_warning

    def fit(self, phi, s):
        """
        Args:
        s (numpy.ndarray): Input signal
        phi (numpy.ndarray): Dictionary
        """

        self.s = s
        self.phi = phi
        self.a = np.zeros_like(self.s)
        self.coefficients = np.zeros(phi.shape[1])
        self.r = self.s.copy()

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

            # Update indices
            self.indices.append(lambda_k)

            # Update Coefficients
            self.coefficients = np.zeros(phi.shape[1])
            self.coefficients[self.indices] = betas.flatten()

            # Update residual
            self.r = self.s - X @ betas

            # Update Projection
            self.a = X @ betas
        return self.a, self.coefficients


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
    



class AtomBaggingOrthogonalMatchingPursuit(AtomBaggingBase):
    def __init__(
        self,
        K,
        atom_bag_percent=1,
        select_atom_percent=0,
        random_seed=None,
        ignore_warning=False,
    ):
        super().__init__(
            K, atom_bag_percent, select_atom_percent, random_seed, ignore_warning
        )


    def fit(self, phi, s):
        self.reset()

        """
        Args:
        s (numpy.ndarray): Input signal
        phi (numpy.ndarray): Dictionary
        """
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
            inner_products[self.indices] = 0
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

            # Update residual
            self.r = self.s - X @ betas

            # Update Projection
            self.a = X @ betas
        return self.a, self.coefficients



class BaggingPursuit(AtomBaggingBase):
    def __init__(
        self,
        N,
        K,
        method="MP",
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

        self.N = N
        self.K = K
        self.method = method
        self.signal_bag_percent = signal_bag_percent
        self.atom_bag_percent = atom_bag_percent
        self.select_atom_percent = select_atom_percent
        self.replace_flag = replace_flag
        self.agg_func = agg_func
        self.random_seed = random_seed
        self.ignore_warning = ignore_warning

        self.s = None
        self.phi = None

        if self.method == "MP":
            self.tmpPursuitModel = AtomBaggingMatchingPursuit(
                K, atom_bag_percent, select_atom_percent, random_seed
            )
        elif self.method == "OMP":
            self.tmpPursuitModel = AtomBaggingOrthogonalMatchingPursuit(
                K, atom_bag_percent, select_atom_percent, random_seed, ignore_warning
            )
        else:
            raise ValueError("Method not supported Yet")

        self.SignalBagging = None

        self.c_lst = []
        self.mse_lst = []
        self.indices_lst = []
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
        self.SignalBagging = SignalBagging(
            self.N, self.signal_bag_percent, self.replace_flag, self.random_seed
        )
        self.SignalBagging.fit(self.phi, self.s)

        s_bag = self.SignalBagging.s_bag
        phi_bag = self.SignalBagging.phi_bag

        for i in range(self.N):
            sub_s = s_bag[i]
            sub_phi = phi_bag[i]
            self.tmpPursuitModel.fit(sub_phi, sub_s)
            c = self.tmpPursuitModel.coefficients
            self.c_lst.append(c)
            self.mse_lst.append(np.mean((sub_s - sub_phi @ c) ** 2))
            self.indices_lst.append(self.tmpPursuitModel.indices)

        if self.agg_func == "weight":
            self.coefficients = self.agg_weight_with_error(self.c_lst, self.mse_lst)
        else:
            self.coefficients = self.agg_weight_with_avg(self.c_lst)
        self.a = self.phi @ self.coefficients
        # return self.final_a, self.final_c


class BOMP(AtomBaggingBase):
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
        self.tmpPursuitModel = AtomBaggingOrthogonalMatchingPursuit(
            K, atom_bag_percent, select_atom_percent, random_seed, ignore_warning
        )
        self.SignalBagging = None
        self.c_lst = []
        self.mse_lst = []
        self.indices_lst = []
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
        self.SignalBagging = SignalBagging(
            self.N_bag,
            self.signal_bag_percent,
            self.replace_flag,
            self.random_seed,
        )
        self.SignalBagging.fit(self.phi, self.s)

        s_bag = self.SignalBagging.s_bag
        phi_bag = self.SignalBagging.phi_bag

        for i in range(self.N_bag):
            sub_s = s_bag[i]
            sub_phi = phi_bag[i]
            self.tmpPursuitModel = AtomBaggingOrthogonalMatchingPursuit(
            self.K, self.atom_bag_percent, self.select_atom_percent, np.random.randint(64), self.ignore_warning)
            self.tmpPursuitModel.fit(sub_phi, sub_s)
            c = self.tmpPursuitModel.coefficients
            self.c_lst.append(c)
            self.mse_lst.append(np.mean((sub_s - sub_phi @ c) ** 2))
            self.indices_lst.append(self.tmpPursuitModel.indices)

        if self.agg_func == "weight":
            self.coefficients = self.agg_weight_with_error(self.c_lst, self.mse_lst)
        else:
            self.coefficients = self.agg_weight_with_avg(self.c_lst)
        self.a = self.phi @ self.coefficients


class OMP:
    def __init__(self, K, select_atom_percent = 0, random_seed=None, ignore_warning=False):
        self.K = K
        self.random_seed = random_seed
        self.select_atom_percent = select_atom_percent
        if select_atom_percent == 0:
            self.atom_weak_select_flag = False
        
        self.indices = []
        self.coefficients = []
        self.ignore_warning = ignore_warning
    
    def fit(self, s, phi):

        """
        Args:
        s (numpy.ndarray): Input signal
        phi (numpy.ndarray): Dictionary
        """

        self.s = s
        self.phi = phi
        self.a = np.zeros_like(self.s)
        self.c = np.zeros(phi.shape[1])
        self.r = self.s.copy()

        if self.random_seed is not None:
            np.random.seed(self.random_seed)

        for i in range(self.K):
            inner_products = (phi.T @ self.r).flatten()
            #so that we will not select the same atom
            inner_products[self.indices] = 0
            if self.atom_weak_select_flag:
                top_ind = np.argsort(np.abs(inner_products))[::-1][:int(phi.shape[1] * self.select_atom_percent)]
                # randomly select one atom
                lambda_k = np.random.choice(top_ind)
            else:
                lambda_k = np.argmax(np.abs(inner_products))
            

            #Ordinary least squares
            X = phi[:, self.indices+[lambda_k]]

            ### TODO: 1. Dump Chosen Index
            try:
                betas = np.linalg.inv(X.T @ X) @ X.T @ self.s
            except:
                if not self.ignore_warning:
                    print('Singular matrix encountered in OMP')
                break

            #Update indices
            self.indices.append(lambda_k)

            #Update Coefficients
            self.c = np.zeros(phi.shape[1])
            self.c[self.indices] = betas.flatten()

            #Update residual
            self.r = self.s - X @ betas

            #Update Projection
            self.a = X @ betas
        return self.a, self.c
    
    def score(self, s_test, phi_test):
        s_pred = phi_test @ self.c
        pred_mse = np.mean((s_pred - s_test)**2)
        return pred_mse
    