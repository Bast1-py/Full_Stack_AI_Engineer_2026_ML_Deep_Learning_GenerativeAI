import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU, SimpleRNN, Embedding, LSTM, Dropout, TimeDistributed

vocab_size = 14600
max_len = 180

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocab_size)
X_train = pad_sequences(X_train, maxlen=max_len, padding='post')
X_test = pad_sequences(X_test, maxlen=max_len, padding='post')

print(f'Training Data Shape: {X_train.shape}')
print(f'Testing Data Shape: {X_test.shape}')
print('\n\n')


gru_model = Sequential(
    [
        Embedding(input_dim=vocab_size, output_dim=128),
        GRU(128, activation='tanh', return_sequences=False),
        Dense(1, activation='sigmoid')
    ]
)
gru_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
gru_model.summary()
gru_history = gru_model.fit(X_train, y_train, validation_split=0.2, epochs=15, batch_size=32)
gru_loss, gru_accuracy = gru_model.evaluate(X_test, y_test)
print(f'GRU Testing Loss: {gru_loss:.4f}, Testing Accuracy: {gru_accuracy:.4f}')