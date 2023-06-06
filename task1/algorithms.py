import numpy as np
from sklearn.linear_model import Lasso

# Matching Pursuit
def matching_pursuit(s, phi, K):
    """
    Perform the Matching Pursuit algorithm

    Args:
    s (numpy.ndarray): Input signal
    phi (numpy.ndarray): Dictionary
    K (int): Number of iterations (sparsity)

    Returns:
    a (numpy.ndarray): Sparse representation of s
    indices (list): List of indices of selected atoms
    coefficients (list): List of coefficients of selected atoms
    """
    # Initialize a and r
    a = np.zeros_like(s)
    r = s.copy()
    indices = []
    coefficients = []
    # Perform Matching Pursuit
    for _ in range(K):
        # Compute inner products
        inner_products = phi.T @ r

        # Find the index with maximum absolute correlation
        lambda_k = np.argmax(np.abs(inner_products), axis=0)

        # Save the index
        indices.append(lambda_k[0])

        # Save the coefficient
        coefficients.append(inner_products[lambda_k])

        # Update a
        a += coefficients[-1] * phi[:, lambda_k]

        # Update r
        r = s - a

    return a, indices, coefficients


# Orthogonal Matching Pursuit

def orthogonal_matching_pursuit(s, phi, K):
    """
    Perform the Orthogonal Matching Pursuit algorithm

    Args:
    s (numpy.ndarray): Input signal
    phi (numpy.ndarray): Dictionary
    K (int): Number of iterations (sparsity)

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

    for _ in range(K):
        # Compute inner products
        inner_products = phi.T @ r
        
        # Though paper says OMP will not choose the same index twice, it does.
        inner_products[indices] = np.min(np.abs(inner_products))

        # Find the index with maximum absolute correlation
        lambda_k = np.argmax(np.abs(inner_products), axis=0)
        # print(np.max(np.abs(inner_products)))
        
        # Save the index
        indices.append(lambda_k[0])
        # print(indices)

        # Ordinary Least Squares
        X = phi[:, indices]
        
        try:
            betas = np.linalg.inv(X.T @ X) @ X.T @ s
        except np.linalg.LinAlgError:
            print("Current params:", s, phi, K)
            print("Singular matrix, stopping the algorithm")
            break

        # Save the coefficient
        coefficients = betas

        # Update a
        a = X @ betas

        # Update r
        r = s - a

    return a, indices, coefficients


# Weak Orthogonal Matching Pursuit

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


def bootstrap_sample(y, phi, percentage):
    '''
    Perform the bootstrap sample algorithm
    Args:
    y (numpy.ndarray): Input signal
    phi (numpy.ndarray): Dictionary
    percentage (float): Percentage of the original signal
    '''
    # Get the number of samples
    num_samples = int(percentage * y.shape[0])
    
    # Get the indices of the samples
    indices = np.random.choice(y.shape[0], num_samples, replace=True)
    
    # Get the samples
    y_samples = y[indices]
    phi_samples = phi[indices]
    
    return y_samples, phi_samples