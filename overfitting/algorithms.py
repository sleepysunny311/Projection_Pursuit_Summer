import numpy as np

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
    
    def predict(self, phi_test):
        s_pred = phi_test @ self.c
        return s_pred

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