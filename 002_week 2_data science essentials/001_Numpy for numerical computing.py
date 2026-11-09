import numpy as np

arr = np.array(
    [1,2,3,4,5]
)
zeroes = np.zeros((3,3))
# print(zeroes)
# print(arr)

ones = np.ones((2,4))
# print(ones)

range_array = np.arange(1,100,3)
# print(range_array)

linspace_array = np.linspace(0,10,3)
# print(linspace_array)

arr = np.array([1,2,3,4,5,6,7,8,9])
reshaped = arr.reshape((3,3))
# print(reshaped)

expanded = arr[:, np.newaxis]
# print(expanded)

a = np.array([1,2,3,4,5,6,7,8,9])
b = np.array([0.23,0.35,0.88,1.12,1.14,1.56,3.45,4.01,4.15])
# print(a+b)
# print(a*b)
# print(a/b)

# print(np.sqrt(arr))
# print(np.sum(arr))
# print(np.mean(arr))
# print(np.max(arr))


# exercise 1: generate array for basic mathematical operations
a = np.arange(1,6)
b = np.arange(6,11)

# print(a)
# print(b)
#
# print('Add: ', a+b)
# print('Sub: ', a-b)
# print('Mult: ', a*b)
# print('Div: ', a/b)

# exercise 2: create a 4x4 matrix and perform operations
matrix = np.array(
    [[1,2,3,4],
     [4,5,6,7],
     [9,10,11,12]]
)
print('Original Matrix: \n', matrix)

# transpose
transpose = matrix.T
print('Transposed Matrix: \n', transpose)

another_matrix = np.array([[9,8,7,0.77], [6,5,4,0.6], [3,2,1,0.5]])
print('Addition: \n', matrix + another_matrix)
print('Multiplication : \n', matrix * another_matrix)