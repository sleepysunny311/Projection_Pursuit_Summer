import numpy as np
import pandas as pd
import pickle as pkl
from concurrent.futures import ThreadPoolExecutor
import os
from algorithms import orthogonal_matching_pursuit

# TODO: Hyperparameters search for omp

##* Data Generation
##* N (columns of dictionary) = 10k
##* d (dimension of signal) = 300, 600, 900, 1200, 1500, 2000
##* X_{ij} \sim N(0, \frac{1}{n}) (The same as N(0,1) them normalize)
##* m (sparse level) = 20, 40, 80


##! Task 1: Given signal, dictionary, sparsity level and noise level, use testset (10% of the whole signal) to find the best K(depth) for omp


##! Task 2: Relationship between best K and noise level (std)
##* Noise level = 0.01, 0.05, 0.1, 0.2, 0.5



##! Task 3: Relationship between Test Error and noise level (std)