from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Dropout
from datasets import load_dataset
from huggingface_hub import HfFileSystem

fs = HfFileSystem()


import numpy as np




vocab_size = 12200
max_len = 150
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocab_size)

# decode reviews to text preprocessing
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}
decode_reviews = [' '.join([reverse_word_index.get(i - 3, "?") for i in review]) for review in X_train[:8]]


# pad sequences
X_train = pad_sequences(X_train, maxlen=max_len, padding='post')
X_test = pad_sequences(X_test, maxlen=max_len, padding='post')
print(f'Training Data Shape: {X_train.shape}, {y_train.shape}')
print(f'Testing Data Shape: {X_test.shape}, {y_test.shape}')


# load GloVe embeddings
# https://huggingface.co/kcz358/glove/tree/main/glove.6B?utm_source=chatgpt.com
glove_file = "models/kcz358/glove/glove.6B/glove.6B.100d.txt"
embedding_index = {}

with fs.open(glove_file, 'rt', encoding='utf-8') as file:
    for line in file:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embedding_index[word] = coefs

print(f'Loaded {len(embedding_index)} word vectors')


# prepare embedding matrix
embeddings_dim = 100
embedding_matrix = np.zeros((vocab_size, embeddings_dim))

for word, i in word_index.items():
    if i < vocab_size:
        embedding_vector = embedding_index.get(word)

        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector


# define LSTM model with GloVe embeddings
model = Sequential(
    [
        Embedding(input_dim=vocab_size, output_dim=embeddings_dim, weights=[embedding_matrix], input_length=max_len, trainable=False),
        LSTM(128, activation='tanh', return_sequences=False),
        Dense(1, activation='sigmoid')
    ]
)


# compile the model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
model.summary()

# training the model
history = model.fit(
    X_train, y_train, validation_split=0.2,
    epochs=26, batch_size=32, verbose=1
)

loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f'LSTM Model with GloVe Testing Accuracy: {accuracy*100:.5f}%')
