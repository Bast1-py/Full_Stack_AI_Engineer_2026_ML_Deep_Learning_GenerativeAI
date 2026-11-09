import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0


# one-hot encoding target labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

print(f'Training Data Shape: {X_train.shape}, {y_train.shape}')
print(f'Testing Data Shape: {X_test.shape}, {y_test.shape}')

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])


# compile the model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# display the model summary
histroy = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=11,
    batch_size=64,
    verbose=1
)


# evaluate the baseline model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Baseline Model Test Accuracy {accuracy*100}%')




# define an improved model
improved_model = Sequential([
    Conv2D(64, (5,5), activation='relu', input_shape=(32,32,3)),
    MaxPooling2D((2,2)),
    Conv2D(128, (5,5), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])


# compile the improved model with a learning rate scheduler
optimizer = tf.keras.optimizers.Adam(learning_rate=0.002)
improved_model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)



# train the improved model
improved_histroy = improved_model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=21,
    batch_size=64,
    verbose=1
)


# evaluate the improved model
improved_loss, improved_accuracy = improved_model.evaluate(
    X_test, y_test,
    verbose=0
)
print(f'Improved Model Test Accuracy {improved_accuracy*100}%')


# visualize plot training and validation accuracy
plt.plot(improved_histroy.history['accuracy'], label='Training Accuracy')
plt.plot(improved_histroy.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# visualize plot training and validation Loss
plt.plot(improved_histroy.history['loss'], label='Training Loss')
plt.plot(improved_histroy.history['val_loss'], label='Validation Loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()