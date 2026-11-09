from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

print(f'Training Data Shape: {X_train.shape}, Label Shapes: {y_train.shape}')
print(f'Test Data Shape: {X_test.shape}, Label Shapes: {y_test.shape}')


# Build the CNN Model
model = Sequential(
    [
        Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)),
        MaxPooling2D((2,2)),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D((2,2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ]
)


model.summary()

model.compile(
    optimizer='adam',
    loss = 'categorical_crossentropy',
    metrics = ['accuracy']
)

# training the model
history = model.fit(
    X_train, y_train,
    epochs = 18,
    batch_size = 64,
    validation_split = 0.2,
)


# evaluate on the test datasets
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'\nTest Accuracy: {test_accuracy*100:.4f}%')

#plot accuracy
plt.plot(
    history.history['accuracy'], label = 'Training Accuracy'
)
plt.plot(
    history.history['val_accuracy'], label = 'Validation Accuracy'
)
plt.title('Model Accuracy Score')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# plot loss
plt.plot(
    history.history['loss'], label = 'Training Loss'
)
plt.plot(
    history.history['val_loss'], label = 'Validation Loss'
)
plt.title('Model Loss Score')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()