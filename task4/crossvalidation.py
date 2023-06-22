import numpy as np

class CrossValidator:
    def __init__(self, algorithm, K_lst, cv = 5, shuffle_split = False, seed = 0):

        """
        Initialize the CrossValidator class

        Args:
            signal: the true signal
            dictionary: the dictionary
            cv_num: the number of folds
            K_lst: the list of K to try
            algorithm: the algorithm to calculate the error
        """
        
        self.cv_num = cv
        self.algorithm = algorithm
        self.K_lst = K_lst
        
        self.cv_res = None
        self.signal = None
        self.dictionary = None
        self.lowest_error = None
        self.lowest_error_K = None
        self.K_cv_error = None
        self.shuffle_split = shuffle_split
        self.seed = seed
                
    def cv_split(self, shuffle_split = False, seed = 0):

        """
        Split the signal and dictionary into cv_num folds

        Returns:
            cv_res: a list of tuples, each tuple is a fold of train signal, train dictionary, test signal, test dictionary
        """
        self.cv_res = []
        signal = self.signal.ravel()
        dictionary = self.dictionary
        if shuffle_split:
            np.random.seed(seed)
            shuffle_idx = np.random.permutation(len(signal))
            signal = signal[shuffle_idx]
            dictionary = dictionary[shuffle_idx, :]
        cv_signal = np.split(signal, self.cv_num)
        cv_dictionary = np.split(dictionary, self.cv_num)
        for i in range(self.cv_num):
            train_signal = np.concatenate(cv_signal[:i] + cv_signal[i + 1:], axis = 0)
            train_dictionary = np.concatenate(cv_dictionary[:i] + cv_dictionary[i + 1:], axis=0)
            test_signal = cv_signal[i]
            test_dictionary = cv_dictionary[i]
            self.cv_res.append((train_signal, train_dictionary, test_signal, test_dictionary))
        return self.cv_res

        
    def cal_cv_error(self, algorithm):

        """
        Calculate the cross validation error of the algorithm

        Args:
            algorithm: the algorithm to calculate the error
        Returns:
            error: the cross validation error
        """

        error_lst = []
        for i in range(self.cv_num):
            train_signal, train_dictionary, test_signal, test_dictionary = self.cv_res[i]
            algorithm.fit(train_signal, train_dictionary)
            error_lst.append(algorithm.score(test_signal, test_dictionary))
        return np.mean(error_lst)
    
    def fit(self, signal, dictionary):
        """
        Calculate the best K for OMP algorithm using cross validation

        Args:
            signal: the true signal
            dictionary: the dictionary
            cv_num: the number of folds
            K_lst: the list of K to try
        Returns:
            lowest_error: the lowest error
            lowest_error_K: the K that gives the lowest error
            K_cv_error: the list of cross validation error for each K
        """
        self.signal = signal
        self.dictionary = dictionary
        self.cv_split(self.shuffle_split, self.seed)
        self.K_cv_error = []
        algorithm = self.algorithm
        K_lst = self.K_lst
        cv_num = self.cv_num
        for K in self.K_lst:
            current_K_algorithm = algorithm(K, ignore_warning=True)
            self.K_cv_error.append(self.cal_cv_error(current_K_algorithm))
        
        self.lowest_error = np.min(self.K_cv_error)
        self.lowest_error_K = self.K_lst[int(np.argmin(self.K_cv_error))]
        return self.lowest_error, self.lowest_error_K, self.K_cv_error


    def get_cv_res(self):
        if self.cv_res is None:
            raise ValueError("The cv_res is empty, please run fit() first")
        else:
            return self.cv_res

    # def update_signal(self, new_signal):

    #     """
    #     Update the signal and reset the updated flag

    #     Args:
    #         new_signal: the new signal
    #     """

    #     if new_signal.ravel().shape != self.signal.shape:
    #         raise ValueError("The shape of the new signal is not the same as the old signal")
    #     self.signal = new_signal.ravel()
    #     if len(self.cv_res) == 0:
    #         raise ValueError("The cv_res is empty, please run cv_split() first")
    #     cv_signal = np.split(new_signal, self.cv_num)
    #     for i in range(self.cv_num):
    #         train_signal, train_dictionary, test_signal, test_dictionary = self.cv_res[i]
    #         train_signal = np.concatenate(cv_signal[:i] + cv_signal[i + 1:], axis = 0)
    #         test_signal = cv_signal[i]
    #         self.cv_res[i] = (train_signal, train_dictionary, test_signal, test_dictionary)