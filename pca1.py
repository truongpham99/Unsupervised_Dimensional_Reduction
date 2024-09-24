import numpy as np
import sys

if len(sys.argv) != 3 :
    print('usage : ', sys.argv[0], 'input_file', 'output_file')
    sys.exit()

input_file = sys.argv[1]
output_file = sys.argv[2]

X = np.genfromtxt(input_file, delimiter=',')

# Eigen decomposition
covariance_matrix = np.matmul(X.T,X)

eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

# Sort eigenvalues and eigenvectors in descending order
idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

# Project data onto principal components
eigenvectors = eigenvectors[:, :2]

transformed_X = np.dot(X, eigenvectors)
    

np.savetxt(output_file, transformed_X, delimiter=',', fmt='%f')

