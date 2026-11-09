import numpy as np

aa = np.array([[2,3], [5,6]])
determinant = np.linalg.det(aa)
# print('Determinant: ', determinant)

inverse = np.linalg.inv(aa)
# print('Inverse of A: \n', inverse)

eigenValues, eigenVectors = np.linalg.eig(aa)
print('EigenValues: \n', eigenValues)
print('EigenVectors: \n', eigenVectors)