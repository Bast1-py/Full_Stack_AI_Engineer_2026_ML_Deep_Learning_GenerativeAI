from tensorflow.keras.datasets import mnist, cifar10
import tensorflow as tf
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt


# load MNIST
(X_train_mnist, y_train_mnist), (X_test_mnist, y_test_mnist) = mnist.load_data()
print(f"MNIST Datasets: Train - {X_train_mnist.shape}, Test - {X_test_mnist.shape}")

print('\n')
(X_train_cifar, y_train_cifar), (X_test_cifar, y_test_cifar) = cifar10.load_data()
print(f'CIFAR-10 Dataset: Train - {X_train_cifar.shape}, Test - {X_test_cifar.shape}')


# define a basic dense layer
layer = tf.keras.layers.Dense(units=10, activation='relu')
print(f'\nTensorFlow Layer: {layer}')

# define a basic dense layer
layer = nn.Linear(in_features=10, out_features=5)
print(f'\nPyTorch Layer: {layer}')


# visaulize MNIST sample
plt.imshow(X_train_mnist[0], cmap='gray')
plt.title(f'MNIST Label: {y_train_mnist[0]}')
plt.show()


# visaulize CIFAR-10 sample
plt.imshow(X_train_cifar[0])
plt.title(f'CIFAR Label: {y_train_cifar[0]}')
plt.show()