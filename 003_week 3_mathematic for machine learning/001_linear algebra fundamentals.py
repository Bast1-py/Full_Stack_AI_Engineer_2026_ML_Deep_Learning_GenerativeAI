import numpy as np

# example 1: matrix operations
A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
B = np.array([[10, 11, 12], [13, 14, 15], [16, 17, 18]])
# print('Addition: \n', A+B)
# print('Substraction: \n', B-A)

C = 2 * A
# print('Scalar Multiplication \n', C)

result = np.dot(A,C)
# print('Matrix Multiplication \n', result)

# example 2: special matrics
I = np.eye(7)
# print('Identify Matrix \n', I)

Z = np.zeros((2,3))
# print('Zero Matrix \n', Z)

D = np.diag([1,2,3,4,5,6,7])
# print('Diagonal Matrix \n', D)


# exercise 1: create vectors and matrices using numpy
# aa = np.array([[1,3,5], [2,4,6]])
# bb = np.array([[3,5,6], [8,3,10]])
# print('Addition \n', aa+bb)
# print('Subtraction \n', bb-aa)
# print('Scalar Mult: \n', 3 * aa)


# exercise 2: implement matrix-vector multiplication
# mm = np.array([[1,2,3,5], [4,5,6,8], [7,8,9,11]])
# vv = np.array([1,0,-1,2])
# result = np.dot(mm,vv)
# print('Matrix-vector Multiplication: \n', result)


# exercise 3: explore special matrices

