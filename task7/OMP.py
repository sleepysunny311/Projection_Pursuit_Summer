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
    
    
