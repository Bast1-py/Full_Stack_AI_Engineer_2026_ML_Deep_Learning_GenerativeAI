# import numpy as np
#
#
# queries = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
# keys = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
# values = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
#
# scores = np.dot(queries, keys.T)
#
#
# # apply softmax to normalize scores
# def softmax(x):
#     exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
#     return exp_x / exp_x.sum(axis=-1, keepdims=True)
#
# attention_weights = softmax(scores)
#
#
# # compute the weighted sum of values
# context = np.dot(attention_weights, values)
# print('Attention Weights:\n', attention_weights)
# print('\n\nContext Vector:\n', context)


import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt

queries = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
keys = torch.tensor([[1.0, 0.0, 1.0], [1.0, 1.0, 0.0], [0.0, 1.0, 1.0]])
values = torch.tensor([[10.0, 0.0], [0.0, 12.0], [5.0, 5.0]])


# compute attention scores
scores = torch.matmul(queries, keys.T)

# apply softmax to normalize scores
attention_weights = F.softmax(scores, dim=-1)

# compute the weighted sum of values
context = torch.matmul(attention_weights, values)

print('Attention Weights:\n', attention_weights)
print('\n\nContext Vector:\n', context)

# visualize attention weights
plt.matshow(attention_weights)
plt.colorbar()
plt.title('Attention Weights')
plt.show()