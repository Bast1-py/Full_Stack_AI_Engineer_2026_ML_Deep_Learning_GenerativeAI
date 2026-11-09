from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import  Embedding, Dense, SimpleRNN, GRU, LSTM, Dropout

import matplotlib.pyplot as plt
import seaborn as sns


vocab_size = 22500
max_len = 150
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocab_size)


# pad sequences
X_train = pad_sequences(X_train, maxlen=max_len, padding='post')
X_test = pad_sequences(X_test, maxlen=max_len, padding='post')
print(f'Training Data Shape: {X_train.shape}, {y_train.shape}')
print(f'Testing Data Shape: {X_test.shape}, {y_test.shape}')

# define the RNN Model
rnn_model = Sequential(
    [
        Embedding(input_dim=vocab_size, output_dim=128),
        SimpleRNN(128, activation='tanh'),
        Dense(1, activation='sigmoid')
    ]
)

rnn_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
rnn_model.summary()




# define LSTM Model
lstm_model = Sequential(
    [
        Embedding(input_dim=vocab_size, output_dim=128),
        LSTM(128, activation='tanh'),
        Dense(1, activation='sigmoid')
    ]
)

lstm_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
lstm_model.summary()



# define GRU Model
gru_model = Sequential(
    [
        Embedding(input_dim=vocab_size, output_dim=128),
        GRU(128, activation='tanh'),
        Dense(1, activation='sigmoid')
    ]
)

gru_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
gru_model.summary()


# training model
history_rnn = rnn_model.fit(X_train, y_train, validation_split=0.2, epochs=9, batch_size=64, verbose=1)
history_lstm = lstm_model.fit(X_train, y_train, validation_split=0.2, epochs=9, batch_size=64, verbose=1)
history_gru = gru_model.fit(X_train, y_train, validation_split=0.2, epochs=9, batch_size=64, verbose=1)


# evaluate model
loss_rnn, accuracy_rnn = rnn_model.evaluate(X_test, y_test, verbose=0)
loss_lstm, accuracy_lstm = lstm_model.evaluate(X_test, y_test, verbose=0)
loss_gru, accuracy_gru = gru_model.evaluate(X_test, y_test, verbose=0)


print(f'RNN Test Accuracy: {accuracy_rnn*100:.3f}%')
print(f'LSTM Test Accuracy: {accuracy_lstm*100:.3f}%')
print(f'GRU Test Accuracy: {accuracy_gru*100:.3f}%')


# plot training accuracy
sns.set_theme(
    style='whitegrid',
    context='notebook',
    font_scale=1.1,
)
# store model histories
histories = {
    'SimpleRNN': history_rnn,
    'LSTM': history_lstm,
    'GRU': history_gru
}
colors = {
    'SimpleRNN': '#3498D8',
    'LSTM': '#E67E22',
    'GRU': '#2ECC71'
}

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('Deep Learning Model Performance Comparison', fontsize=20,
             fontweight='bold', y=1.02)

# metrics konfigurasi
metrics = [
    ('accuracy', 'Training Accuracy', 0, 0),
    ('val_accuracy', 'Validation Accuracy', 0, 1),
    ('loss', 'Training Loss', 1, 0),
    ('val_loss', 'Validation Loss', 1, 1)
]

# plot all metrics
for metric, title, row, col in metrics:
    ax = axes[row, col]

    for model_name, history in histories.items():
        values = history.history[metric]
        epochs = range(1, len(values) + 1)
        ax.plot(epochs, values, label=model_name, color=colors[model_name],
                linewidth=2.5, marker='o', markersize=5, alpha=0.9)

    if 'loss' in metric:
        best_idx = values.index(min(values))
    else:
        best_idx = values.index(max(values))


    ax.scatter(
        best_idx + 1, values[best_idx],
        color=colors[model_name],
        s=90, edgecolors='white',
        linewidth=1.5, zorder=5
    )

    ax.set_title(
        title, fontsize=14,
        fontweight='bold',
        pad=12
    )

    ax.set_xlabel('Epochs', fontsize=11)
    ax.set_ylabel(
        'Accuracy' if 'accuracy' in metric else 'Loss', fontsize=11,
    )
    ax.set_xticks(range(1, 14))
    ax.legend(
        loc='best', frameon=True, fontsize=10
    )
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# adjust layout
plt.tight_layout()
plt.show()


# ==========================================
# TEST ACCURACY COMPARISON
# ==========================================

results = {
    "SimpleRNN": accuracy_rnn * 100,
    "LSTM": accuracy_lstm * 100,
    "GRU": accuracy_gru * 100
}

plt.figure(figsize=(10, 6))

bars = plt.bar(
    results.keys(),
    results.values(),
    color=["#3498DB", "#E67E22", "#2ECC71"],
    width=0.55,
    edgecolor="white"
)

# Display accuracy values
for bar in bars:

    height = bar.get_height()

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height + 0.3,
        f"{height:.2f}%",
        ha="center",
        fontsize=12,
        fontweight="bold"
    )

plt.title(
    "Test Accuracy Comparison",
    fontsize=17,
    fontweight="bold",
    pad=15
)

plt.xlabel("Deep Learning Models", fontsize=12)
plt.ylabel("Test Accuracy (%)", fontsize=12)

plt.ylim(
    max(0, min(results.values()) - 5),
    105
)

plt.grid(
    axis="y",
    linestyle="--",
    alpha=0.4
)

sns.despine()

plt.tight_layout()
plt.show()



# plt.plot(history_rnn.history['accuracy'], label='RNN Training Accuracy')
# plt.plot(history_lstm.history['accuracy'], label='LSTM Training Accuracy')
# plt.plot(history_gru.history['accuracy'], label='GRU Training Accuracy')
#
# plt.title('Training Accuracy Comparison')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.show()