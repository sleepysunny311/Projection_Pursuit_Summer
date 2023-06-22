import numpy as np

def generate_gaussian_noises_dict(N, d, seed=0):
    np.random.seed(seed)
    gaussian_noises = np.random.normal(size=(d, N))
    norms = np.linalg.norm(gaussian_noises, axis=0, keepdims=True)
    # Create unit-norm vectors
    unit_vectors = gaussian_noises / norms
    return unit_vectors

def generate_sparse_response(gaussian_matrix, m, seed=0):
    np.random.seed(seed)
    indices = np.random.choice(gaussian_matrix.shape[1], size=m, replace=False)
    selected_vectors = gaussian_matrix[:, indices]
    coefficients = np.random.normal(size=(m, 1))  # random coefficients for each selected vector
    y = selected_vectors @ coefficients
    return y, indices, coefficients

def generate_perturbed_response(y, noise_level, seed=0):
    np.random.seed(seed)
    norm_y = np.linalg.norm(y)
    noise = np.random.normal(size=y.shape, scale=norm_y * noise_level)
    y_perturbed = y + noise
    return y_perturbed

def generate_perturbed_responses(y, noise_levels, seed=0):
    return [generate_perturbed_response(y, noise_level, seed) for noise_level in noise_levels]



class DataGeneratorBase:
    # Sub Data Generator Base
    def __init__(self,dictionary_length, dictionary_dimensions, indice_number, noise_level, random_seed, data_path = ""):
        self.dictionary_length = dictionary_length
        self.dictionary_dimensions = dictionary_dimensions
        self.indice_number = indice_number
        self.noise_level = noise_level
        self.random_seed = random_seed

        self.dictionary = None
        self.signal = None
        self.indices = None
        self.coefficients = None
        self.perturbed_signal = None
        self.coherence_list = None
        self.coherence = None
        
    def generate_dictionary(self):
        return None
    
    def generate_simulated_signal(self):
        return None
    
    def input_noise(self):
        return None
    
    def shuffle(self):
        np.random.seed(self.random_seed)
        self.random_seed = np.random.randint(100000000)
        self.generate_dictionary()
        self.generate_simulated_signal()
        self.input_noise()
        return self.signal, self.dictionary, self.indices, self.coefficients,self.perturbed_signal
    
    def get_current_shuffle(self):
        return self.signal, self.dictionary, self.indices, self.coefficients,self.perturbed_signal

    def retrive_data(self,path = ""):
        # TODO: retrive data from given data path so you don't have to calcaute the dictionary every goddamn time
        pass
    
    def measure_coherence(self):

        """
        Measure coherence of the dictionary
        """
        temp_coherence =  (self.dictionary).T @ (self.dictionary)
        self.coherence_list = np.abs(temp_coherence[np.triu_indices(temp_coherence.shape[0], 1)])
        self.coherence = np.max(self.coherence_list)
        return self.coherence
    
    def update_noise_level(self,new_noise_level):
        """
        This function is used to change the noise_level and change the noise only

        Returns:
            signal: the signal
            dictionary: the dictionary
            indices: the indices
            coefficients: the coefficients
            perturbed_signal: the perturbed signal
        """

        self.noise_level = new_noise_level
        self.input_noise()
        return self.perturbed_signal



class GaussianDataGenerator(DataGeneratorBase):
    def __init__(self, dictionary_length, dictionary_dimensions, indice_number, noise_level, random_seed):
        super().__init__(dictionary_length, dictionary_dimensions, indice_number, noise_level, random_seed)
    def generate_dictionary(self):
        np.random.seed(self.random_seed)
        gaussian_noises = np.random.normal(size = (self.dictionary_dimensions,self.dictionary_length))
        norms = np.linalg.norm(gaussian_noises,axis=0, keepdims=True)
        self.dictionary = gaussian_noises / norms
    def generate_simulated_signal(self):
        np.random.seed(self.random_seed)
        self.indices = np.random.choice(self.dictionary.shape[1], size=self.indice_number, replace=False)
        self.coefficients = np.random.normal(size=(self.indice_number,1))
        self.signal = (self.dictionary[:,self.indices]) @ (self.coefficients)
    def input_noise(self):
        np.random.seed(self.random_seed)
        # norm_y = np.linalg.norm(self.signal)
        norm_beta = np.linalg.norm(self.coefficients)
        #y' = y/norm beta' = beta/norm
        # self.signal = self.signal/norm_y
        #self.coefficients = self.coefficients/norm_y
        noise = np.random.normal(size=self.signal.shape, scale= self.noise_level*norm_beta)
        self.perturbed_signal = self.signal + noise