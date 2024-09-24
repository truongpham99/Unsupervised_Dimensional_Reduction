import numpy as np
import sys

if len(sys.argv) != 4 :
    print('usage : ', sys.argv[0], 'input_file', 'output_file')
    sys.exit()

input_file = sys.argv[1]
output_file = sys.argv[2]
alpha = float(sys.argv[3])

X = np.genfromtxt(input_file, delimiter=',')

# Calculate the distance matrix
squared_diff = np.sum((X[:, np.newaxis] - X[np.newaxis, :]) ** 2, axis=-1)

D = np.sqrt(squared_diff)

D_alpha = D**alpha

# Apply double centering (matrix centering) to get Gram matrix
n = D.shape[0]
identity_matrix = np.eye(n)
ones_matrix = np.ones((n, n)) / n
centering_matrix = identity_matrix - ones_matrix

# Apply double centering
B = -0.5 * centering_matrix @ D_alpha @ centering_matrix

# Eigen decomposition of the centered Gram matrix
eigenvalues, eigenvectors = np.linalg.eigh(B)

# Sort eigenvalues and eigenvectors in descending order
idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

# Select top n_components dimensions and compute the coordinates
L = np.diag(np.sqrt(eigenvalues[:2]))  
V = eigenvectors[:, :2]  
transformed_X = V @ L  

np.savetxt(output_file, transformed_X, delimiter=',', fmt='%f')
