import numpy as np

def bootstrap_sample(y, phi, sig_bagging_percent = 1):
    '''
    Perform the bootstrap sample algorithm
    Args:
    y (numpy.ndarray): Input signal
    phi (numpy.ndarray): Dictionary
    percentage (float): Percentage of the original signal
    '''
    # Get the number of samples
    num_samples = int(sig_bagging_percent * y.shape[0])
    
    # Get the indices of the samples
    indices = np.random.choice(y.shape[0], num_samples, replace=True)
    
    # Get the samples
    y_samples = y[indices]
    phi_samples = phi[indices, :]
    return y_samples, phi_samples

def feature_bagging_marching_pursuit(s, phi, K, atom_bag_percent=1, select_atom_percent=0):
    indices = []
    coefficients = []
    a = np.zeros_like(s)
    c = np.zeros(phi.shape[1])
    r = s.copy()
    
    select_atom_percent = np.max([0, np.min([1, select_atom_percent])])
    atom_bag_flag = (atom_bag_percent < 1)
    atom_weak_select_flag = (select_atom_percent > 0)
    
    for i in range(K):
        # Compute inner products
        inner_products = (phi.T @ r).flatten()
        # Apply atom bagging
        if atom_bag_flag:
            dropping_indice = np.random.choice(np.arange(phi.shape[1]), size=int((1-atom_bag_percent) * phi.shape[1]), replace=False)
            inner_products[dropping_indice] = 0
        # Apply atom random choosing
        if atom_weak_select_flag:
            # get top select_atom_percent indices
            top_indices = np.argsort(np.abs(inner_products))[::-1][:int(select_atom_percent * phi.shape[1])]
            # randomly choose one from top indices
            lambda_k = np.random.choice(top_indices)
        else:
            lambda_k = np.argmax(np.abs(inner_products))
            
        # Save the index
        indices.append(lambda_k)
        # Save the coefficient
        coefficients.append(inner_products[lambda_k])
        c[lambda_k] += inner_products[lambda_k]
        # Update a
        a += coefficients[-1] * phi[:, lambda_k].reshape(-1, 1)
        # Update r
        r = s - a
    
    return a, c, indices, coefficients
    
    
def bag_agg_simple(c_lst, s, phi):
    # simple average
    c = np.mean(c_lst, axis=0)
    return c

def bag_agg_weight(c_lst, mse_lst):
    # calculate weight
    mse_lst = np.array(mse_lst)
    weight = 1 / mse_lst
    weight = weight / np.sum(weight)
    
    # Calculate the weighted average
    tot = np.zeros_like(c_lst[0])
    for i in range(len(c_lst)):
        tot += c_lst[i] * weight[i]
    return tot
    


def bagging_marching_pursuit(s, phi, K, N, signal_bag_percent = 0.7, atom_bag_percent=1, select_atom_percent=0):
    '''
    Perform the bagging marching pursuit algorithm
    Args:
    s (numpy.ndarray): Input signal
    phi (numpy.ndarray): Dictionary
    K (int): Number of iterations
    sig_bagging_percent (float): Percentage of the original signal
    '''
    c_lst = []
    mse_lst = []
    indices_lst = []
    coefficients_lst = []
    for i in range(N):
        sub_s, sub_phi = bootstrap_sample(s, phi, signal_bag_percent)
        a, c, indice, coefficients = feature_bagging_marching_pursuit(sub_s, sub_phi, K, atom_bag_percent, select_atom_percent)
        sub_mse = np.mean((sub_s - sub_phi @ c)**2)
        c_lst.append(c)
        mse_lst.append(sub_mse)
        indices_lst.append(indice)
        coefficients_lst.append(coefficients)
    
    final_c = bag_agg_weight(c_lst, mse_lst)
    final_a = phi @ final_c
    
    return final_c, final_a, c_lst, mse_lst, indices_lst, coefficients_lst