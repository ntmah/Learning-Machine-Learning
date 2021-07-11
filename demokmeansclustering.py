import numpy as np 
import matplotlib.pyplot as plt 
from scipy.spatial.distance import cdist 

np.random.seed(11)

means = [[2, 2], [8, 3], [3, 6]]
cov = [[1, 0], [0, 1]]
N = 500
X0 = np.random.multivariate_normal(means[0], cov, N)
X1 = np.random.multivariate_normal(means[1], cov, N)
X2 = np.random.multivariate_normal(means[2], cov, N)
print(f'X0:', X2)

X = np.concatenate((X0, X1, X2), axis = 0)
print(f'X', X)