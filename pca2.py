import numpy as np
import sys

if len(sys.argv) != 3 :
    print('usage : ', sys.argv[0], 'input_file', 'output_file')
    sys.exit()

input_file = sys.argv[1]
output_file = sys.argv[2]

X = np.genfromtxt(input_file, delimiter=',')

# Centering the data
mean_value = np.mean(X, axis=0)

centered_matrix = X - mean_value

# Eigen decomposition
covariance_matrix = np.matmul(centered_matrix.T,centered_matrix)

eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

# Sort eigenvalues and eigenvectors in descending order
idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

# Project data onto principal components
eigenvectors = eigenvectors[:, :2]

transformed_X = np.dot(centered_matrix, eigenvectors)
    

np.savetxt(output_file, transformed_X, delimiter=',', fmt='%f')

