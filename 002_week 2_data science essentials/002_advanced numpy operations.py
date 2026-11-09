import numpy as np


np.random.seed(42)
random_array = np.random.rand(13,13)
# print('Random Array: \n', random_array)


random_integers = np.random.randint(0,10, size=(4,3))
# print('Random Integers: ', random_integers)



# exercise 1: broadcasting operations
matrix = np.array(
    [[1,2,3], [4,5,6], [7,8,9]]
)
vector = np.array([1,0,-1])
result_add = matrix + vector
# print('Add: ', result_add)

result_mul = matrix * 2
# print('Multiplication: \n', result_mul)



# exercise 2: generate and filter a random dataset
dataset = np.random.randint(1,56, size=(8,8))
print('Original: \n', dataset)
dataset[dataset > 25] = 0
print('Modified Dataset: \n', dataset)
print('Sum: ', np.sum(dataset))
print('Mean: ', np.mean(dataset))
print('Standard Deviation: ', np.std(dataset))