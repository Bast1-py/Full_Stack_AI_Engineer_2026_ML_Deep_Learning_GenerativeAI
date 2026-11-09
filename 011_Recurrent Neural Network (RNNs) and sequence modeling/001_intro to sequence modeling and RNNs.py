from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, Dropout, TimeDistributed, Embedding
from torch.utils.data import Dataset, DataLoader, TensorDataset

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim



# vocab_size = 25000
# max_len = 220
#
# (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocab_size)
# X_train = pad_sequences(X_train, maxlen=max_len, padding='post')
# X_test = pad_sequences(X_test, maxlen=max_len, padding='post')
# print(f'Training DataShape: {X_train.shape}')
# print(f'Testing DataShape: {X_test.shape}')
#
# model = Sequential(
#     [
#         Embedding(
#             input_dim=vocab_size,
#             output_dim=128,
#         ),
#         SimpleRNN(128, activation='tanh', return_sequences=False),
#         Dense(1, activation='sigmoid'),
#     ]
# )
#
# model.compile(
#     optimizer='adam',
#     loss='binary_crossentropy',
#     metrics=['accuracy']
# )
#
# model.summary()
# history = model.fit(X_train, y_train, epochs=13, validation_split=0.2)
# loss, accuracy = model.evaluate(X_test, y_test)
# print(f'Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}')


vocab_size = 16500
max_len = 160
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocab_size)
X_train = pad_sequences(X_train, maxlen=max_len, padding='post')
X_test = pad_sequences(X_test, maxlen=max_len, padding='post')
train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)

class RNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(RNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        output, hidden = self.rnn(embedded)
        return torch.sigmoid(self.fc(hidden.squeeze(0)))


model = RNNModel(vocab_size=16500, embedding_dim=128, hidden_dim=128, output_dim=1)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.002)


def train_rnn(model, train_loader, criterion, optimizer, epochs=11):
    model.train()

    for epoch in range(epochs):
        epoch_loss = 0

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            predictions = model(X_batch).squeeze(1)
            loss = criterion(predictions, y_batch.float())
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f'Epochs: {epoch+1}, Loss: {epoch_loss/len(train_loader):.4f}')


train_rnn(model, train_loader, criterion, optimizer)

def evaluate_rnn(model, X_test, y_test):
    model.eval()

    with torch.no_grad():
        predictions = model(torch.tensor(X_test)).squeeze(1)
        loss = criterion(predictions, torch.tensor(y_test).float())
        accuracy = ((predictions > 0.5) == torch.tensor(y_test).bool()).float().mean().item()

    print(f'Test Loss: {loss.item():.4f}, Test Accuracy: {accuracy:.4f}')


evaluate_rnn(model, X_test, y_test)