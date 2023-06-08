import numpy as np
from sklearn.linear_model import Lasso


# class BaseMatchingPursuit:
#     def __init__(self, s, phi, K, keeping_ratio=1, beta=0):


class SignalBagging:
    def __init__(self, N, signal_bag_percent=0.7, replace_flag=True, random_seed=0):
        self.s = None
        self.phi = None
        self.N = N
        self.replace_flag = replace_flag
        self.random_seed = random_seed
        self.signal_bag_percent = signal_bag_percent
        self.s_bag = []
        self.phi_bag = []
        self.shuffle_flag = False

    def fit(self, s, phi):
        self.s = s
        self.phi = phi

        num_samples = int(self.signal_bag_percent * len(self.s))
        np.random.seed(self.random_seed)
        for i in range(self.N):
            indices = np.random.choice(self.s.shape[0], num_samples, replace=self.replace_flag)
            s_tmp = self.s[indices]
            phi_tmp = self.phi[indices, :]
            self.s_bag.append(s_tmp)
            self.phi_bag.append(phi_tmp)
        
        self.shuffle_flag = True
        return self.s_bag, self.phi_bag
    
    def change_seed(self, random_seed):
        self.random_seed = random_seed
        self.fit(self.s, self.phi)



class AtomBaggingBase:
    # Submodel base
    def __init__(self, K, atom_bag_percent=1, select_atom_percent=0, random_seed=0):
        self.K = K
        self.atom_bag_percent = np.max([0, np.min([1, atom_bag_percent])])
        self.select_atom_percent = np.max([0, np.min([1, select_atom_percent])])
        self.atom_bag_flag = (atom_bag_percent < 1)
        self.atom_weak_select_flag = (select_atom_percent > 0)
        
        self.indices = []
        self.coefficients = []
        self.s = None
        self.phi = None
        self.a = None
        self.c = None
        self.r = None
        
        self.random_seed = random_seed


class AtomBaggingMatchingPursuit(AtomBaggingBase):
    def __init__(self, K, atom_bag_percent=1, select_atom_percent=0, random_seed=0):
        super().__init__(K, atom_bag_percent, select_atom_percent, random_seed)
    
    def fit(self, s, phi):
        self.s = s
        self.phi = phi
        self.a = np.zeros_like(self.s)
        self.c = np.zeros_like(phi.shape[1])
        self.r = self.s.copy()
        
        np.random.seed(self.random_seed)
        
        for i in range(self.K):
            inner_products = (phi.T @ self.r).flatten()
            if self.atom_bag_flag:
                dropping_indices = np.random.choice(phi.shape[1], int(phi.shape[1] * (1 - self.atom_bag_percent)), replace=False)
                inner_products[dropping_indices] = 0
            if self.atom_weak_select_flag:
                top_ind = np.argsort(np,abs(inner_products))[::-1][:int(phi.shape[1] * self.select_atom_percent)]
                # randomly select one atom
                lambda_k = np.random.choice(top_ind)
            else:
                lambda_k = np.argmax(np.abs(inner_products))
            self.indices.append(lambda_k)
            self.coefficients.append(inner_products[lambda_k])
            self.c[lambda_k] += inner_products[lambda_k]
            self.a += inner_products[lambda_k] * phi[:, lambda_k].reshape(-1, 1)
            self.r = self.s - self.a
        return self.a, self.c

    
class BaggingAlgorithmBase:
    def __init__(self, N, K, signal_bag_flag=True, signal_bag_percent = 0.7, atom_bag_percent=1, select_atom_percent=0, replace_flag=True, agg_func='weight', random_seed=0):

        
        self.N = N
        self.K = K
        self.signal_bag_flag = signal_bag_flag
        if signal_bag_flag:
            self.signal_bag_percent = signal_bag_percent
        else:
            self.signal_bag_percent = None
        self.atom_bag_percent = atom_bag_percent
        self.select_atom_percent = select_atom_percent
        self.replace_flag = replace_flag
        self.agg_func = agg_func
        self.random_seed = random_seed
        
        self.s = None
        self.phi = None
        self.tmpPursuitModel = None
        self.SignalBagging = None
        
        self.c_lst = []
        self.mse_lst = []
        self.indices_lst = []
        self.coefficients_lst = []
        self.final_c = None
        self.final_a = None
        
    def bag_agg_weight(c_lst, mse_lst):
        # Calculate the weight
        mse_lst = np.array(mse_lst)
        weight = 1 / mse_lst
        weight = weight / np.sum(weight)
        
        # Calculate the weighted average
        tot = np.zeros_like(c_lst[0])
        for i in range(len(c_lst)):
            tot += c_lst[i] * weight[i]
        return tot

    def fit(self, s, phi):
        self.s = s
        self.phi = phi
        self.SignalBagging = SignalBagging(self.N, self.signal_bag_percent, self.replace_flag, self.random_seed)
        self.SignalBagging.fit(self.s, self.phi)
        self.tmpPursuitModel = AtomBaggingMatchingPursuit(self.K, self.atom_bag_percent, self.select_atom_percent, self.random_seed)
        
        s_bag = self.SignalBagging.s_bag
        phi_bag = self.SignalBagging.phi_bag
        
        for i in range(self.N):
            sub_s = s_bag[i]
            sub_phi = phi_bag[i]
            self.tmpPursuitModel.fit(sub_s, sub_phi)
            c = self.tmpPursuitModel.c
            self.c_lst.append(c)
            self.mse_lst.append(np.mean((sub_s - sub_phi @ c)**2))
            self.indices_lst.append(self.tmpPursuitModel.indices)
            self.coefficients_lst.append(self.tmpPursuitModel.coefficients)
        
        self.final_c = np.mean(self.c_lst, axis=0)
        if self.agg_func == 'weight':
            self.final_c = self.bag_agg_weight(self.c_lst, self.mse_lst)
        elif self.agg_func == 'simple':
            self.final_c = np.mean(self.c_lst, axis=0)
        else:
            raise ValueError('agg_func must be weight or simple')
        return self.final_c, self.final_a
            



# # Matching Pursuit
# class MatchingPursuit(BaseMatchingPursuit):
#     def run(self):
#         """
#         Perform the Matching Pursuit algorithm.

#         Returns:
#             a (numpy.ndarray): Sparse representation of s.
#             indices (list): List of indices of selected atoms.
#             coefficients (list): List of coefficients of selected atoms.
#         """
#         for _ in range(self.K):
#             # Compute inner products
#             inner_products = (self.phi.T @ self.r).flatten()

#             # Apply dropping
#             if self.dropping_flag:
#                 dropping_indice = np.random.choice(np.arange(self.phi.shape[1]), size=int((1-self.keep_ratio) * self.phi.shape[1]), replace=False)
#                 inner_products[dropping_indice] = 0

#             # Apply beta random choosing
#             if self.random_choose_flag:
#                 num_atoms = min(int(self.beta * self.phi.shape[1]), self.phi.shape[1])
#                 lambda_k = np.random.choice(np.argsort(np.abs(inner_products))[-num_atoms:])
#             else:
#                 lambda_k = np.argmax(np.abs(inner_products))

#             # Save the index
#             self.indices.append(lambda_k)

#             # Save the coefficient
#             self.coefficients.append(inner_products[self.indices[-1]])

#             # Update a
#             self.a = self.a + (self.coefficients[-1] * self.phi[:, self.indices[-1]]).reshape(-1, 1)

#             # Update r
#             self.r = self.s - self.a

#         return self.a, self.indices, self.coefficients



# # Orthogonal Matching Pursuit
# class OrthogonalMatchingPursuit(BaseMatchingPursuit):
#     def run(self):
#         """
#         Perform the Orthogonal Matching Pursuit algorithm.

#         Returns:
#             a (numpy.ndarray): Sparse representation of s.
#             indices (list): Indices of the selected atoms.
#             coefficients (list): Coefficients of the selected atoms.
#         """
#         for _ in range(self.K):
#             # Compute inner products
#             inner_products = (self.phi.T @ self.r).flatten()

#             # Apply dropping

#             if self.dropping_flag:
#                 dropping_indice = np.random.choice(np.arange(self.phi.shape[1]), size=int((1-self.keep_ratio) * self.phi.shape[1]), replace=False)
#                 inner_products[dropping_indice] = 0

#             # Apply beta random choosing
#             if self.random_choose_flag:
#                 num_atoms = min(int(self.beta * self.phi.shape[1]), self.phi.shape[1])
#                 lambda_k = np.random.choice(np.argsort(np.abs(inner_products))[-num_atoms:])
#             else:
#                 lambda_k = np.argmax(np.abs(inner_products))



#             # Ordinary Least Squares
#             X = self.phi[:, self.indices+[lambda_k]]

#             try:
#                 betas = np.linalg.inv(X.T @ X) @ X.T @ self.s
#             except np.linalg.LinAlgError:
#                 print("Current params:", self.K)
#                 print("Singular matrix, stopping the algorithm")
#                 break

#             # Save the index
#             self.indices.append(lambda_k)

#             # Save the coefficient
#             self.coefficients = betas

#             # Update a
#             self.a = X @ betas

#             # Update r
#             self.r = self.s - self.a

#         return self.a, self.indices, self.coefficients

