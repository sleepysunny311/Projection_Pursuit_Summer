import numpy as np
from sklearn.linear_model import Lasso

# Weak Orthogonal Matching Pursuit
# TODO: Implement the Weak Orthogonal Matching Pursuit algorithm into class
def weak_orthogonal_matching_pursuit(s, phi, alpha):
    """
    Perform the Weak Orthogonal Matching Pursuit algorithm

    Args:
    s (numpy.ndarray): Input signal
    phi (numpy.ndarray): Dictionary
    alpha (float): Threshold for stopping the algorithm

    Returns:
    a (numpy.ndarray): Sparse representation of s
    indices (list): Indices of the selected atoms
    coefficients (list): Coefficients of the selected atoms
    """

    # Initialize a and r
    a = np.zeros_like(s)
    r = s.copy()
    indices = []
    coefficients = []
    flag = True

    while flag:
        # Compute inner products
        inner_products = phi.T @ r

        # Create a copy of inner_products to find out the largest inner product
        # Though paper says OMP will not choose the same index twice, it does.
        inner_products_copy = inner_products.copy()
        inner_products_copy[indices] = np.min(np.abs(inner_products_copy))

        # Find the index with maximum absolute correlation
        lambda_k = np.argmax(np.abs(inner_products_copy), axis=0)

        # Save the index
        indices.append(lambda_k[0])

        # Ordinary Least Squares
        X = phi[:, indices]
        betas = np.linalg.inv(X.T @ X) @ X.T @ s

        # Save the coefficient
        coefficients = betas

        # Update a
        a = X @ betas

        # Update r
        r = s - a

        # Check if the largest inner product is greater than alpha times the largest inner product
        largest_inner_product = np.abs(inner_products[lambda_k][0][0])
        if largest_inner_product >= alpha * np.max(np.abs(inner_products)):
            flag = False
    return a, indices, coefficients

# LASSO

def sparse_LASSO(y, phi, ALPHA):
    """
    Perform the sparse LASSO algorithm

    Args:
    y (numpy.ndarray): Input signal
    phi (numpy.ndarray): Dictionary
    alpha (float): Regularization parameter

    Returns:
    a (numpy.ndarray): Sparse representation of y
    """

    lasso_signal = Lasso(alpha=ALPHA, fit_intercept=False, max_iter=10000, tol=0.0001)
    lasso_signal.fit(phi, y)
    lasso_coefficients = lasso_signal.coef_

    lasso_residual = y - phi @ lasso_signal.coef_
    lasso_indices = np.nonzero(lasso_coefficients)[0]
    lasso_coefficients = lasso_coefficients[lasso_indices]

    return lasso_residual, lasso_indices, lasso_coefficients 


class BaseMatchingPursuit:
    def __init__(self, s, phi, K, alpha=1, beta=0):
        """
        Initialize the BaseMatchingPursuit class.

        Args:
            s (numpy.ndarray): Input signal.
            phi (numpy.ndarray): Dictionary.
            K (int): Number of iterations (sparsity).
            alpha (float): Hyperparameter controlling dropping of atoms.
            beta (float): Hyperparameter controlling random selection of atoms.
        """
        self.s = s
        self.phi = phi
        self.K = K
        self.a = np.zeros_like(s)
        self.r = s.copy()
        self.indices = []
        self.coefficients = []
        self.alpha = np.min([1, alpha])
        self.dropping_flag = (alpha < 1)
        self.beta = beta
        self.random_choose_flag = (beta > 0)

    def set_signal(self, new_s):
        """
        Set a new input signal.

        Args:
            new_s (numpy.ndarray): New input signal.
        """
        self.s = new_s
        self.r = new_s.copy()
        self.a = np.zeros_like(new_s)
        self.coefficients = []

    def set_dictionary(self, new_phi):
        """
        Set a new dictionary.

        Args:
            new_phi (numpy.ndarray): New dictionary.
        """
        self.phi = new_phi
        self.a = np.zeros_like(self.s)
        self.r = self.s.copy()
        self.coefficients = []

    def set_alpha(self, new_alpha):
        """
        Set a new alpha value and update the dropping flag accordingly.

        Args:
            new_alpha (float): New alpha value.
        """
        self.alpha = np.min([1, new_alpha])
        self.dropping_flag = (new_alpha < 1)

    def run(self):
        raise NotImplementedError("Subclass must implement this method")

    def __str__(self):
        """
        Return a string representation of the BaseMatchingPursuit class.

        Returns:
            str: String representation of the BaseMatchingPursuit class.
        """
        return f"MatchingPursuit:\n\tInput signal: {self.s}\n\tDictionary: {self.phi}\n\tNumber of iterations: {self.K}\n\tAlpha: {self.alpha}\n\tBeta: {self.beta}\n\tRandom choose flag: {self.random_choose_flag}"

    def get_a(self):
        """
        Get the sparse representation of the input signal.

        Returns:
            numpy.ndarray: Sparse representation of the input signal.
        """
        return self.a

    def get_indices(self):
        """
        Get the list of indices of selected atoms.

        Returns:
            list: List of indices of selected atoms.
        """
        return self.indices

    def get_coefficients(self):
        """
        Get the list of coefficients of selected atoms.

        Returns:
            list: List of coefficients of selected atoms.
        """
        return self.coefficients

# Matching Pursuit
class MatchingPursuit(BaseMatchingPursuit):
    def run(self):
        """
        Perform the Matching Pursuit algorithm.

        Returns:
            a (numpy.ndarray): Sparse representation of s.
            indices (list): List of indices of selected atoms.
            coefficients (list): List of coefficients of selected atoms.
        """
        for _ in range(self.K):
            # Compute inner products
            inner_products = (self.phi.T @ self.r).flatten()

            # Apply alpha dropping
            dropping_indice = np.random.choice(np.arange(self.phi.shape[1]), size=int(self.alpha * self.phi.shape[1]), replace=False)
            if self.dropping_flag:
                inner_products[dropping_indice] = 0

            # Apply beta random choosing
            if self.random_choose_flag:
                num_atoms = min(int(self.beta * self.phi.shape[1]), self.phi.shape[1])
                lambda_k = np.random.choice(np.argsort(np.abs(inner_products))[-num_atoms:])
            else:
                lambda_k = np.argmax(np.abs(inner_products))

            # Save the index
            self.indices.append(lambda_k)

            # Save the coefficient
            self.coefficients.append(inner_products[self.indices[-1]])

            # Update a
            self.a = self.a + (self.coefficients[-1] * self.phi[:, self.indices[-1]]).reshape(-1, 1)

            # Update r
            self.r = self.s - self.a

        return self.a, self.indices, self.coefficients



# Orthogonal Matching Pursuit
class OrthogonalMatchingPursuit(BaseMatchingPursuit):
    def run(self):
        """
        Perform the Orthogonal Matching Pursuit algorithm.

        Returns:
            a (numpy.ndarray): Sparse representation of s.
            indices (list): Indices of the selected atoms.
            coefficients (list): Coefficients of the selected atoms.
        """
        for _ in range(self.K):
            # Compute inner products
            inner_products = (self.phi.T @ self.r).flatten()

            # Apply alpha dropping
            dropping_indice = np.random.choice(np.arange(self.phi.shape[1]), size=int(self.alpha * self.phi.shape[1]), replace=False)
            if self.dropping_flag:
                inner_products[dropping_indice] = 0

            # Apply beta random choosing
            if self.random_choose_flag:
                num_atoms = min(int(self.beta * self.phi.shape[1]), self.phi.shape[1])
                lambda_k = np.random.choice(np.argsort(np.abs(inner_products))[-num_atoms:])
            else:
                lambda_k = np.argmax(np.abs(inner_products))



            # Ordinary Least Squares
            X = self.phi[:, self.indices+[lambda_k]]

            try:
                betas = np.linalg.inv(X.T @ X) @ X.T @ self.s
            except np.linalg.LinAlgError:
                print("Current params:", self.K)
                print("Singular matrix, stopping the algorithm")
                break

            # Save the index
            self.indices.append(lambda_k)

            # Save the coefficient
            self.coefficients = betas

            # Update a
            self.a = X @ betas

            # Update r
            self.r = self.s - self.a

        return self.a, self.indices, self.coefficients

