from tensorflow.keras.layers import Dense, Input, LayerNormalization, Add, MultiHeadAttention, Dropout, Flatten, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model

from transformers import BertTokenizer, BertModel, TFBertModel



# # define a simplified transformer encoder block
# def transformer_encoder(input_dim, num_heads, ff_dim):
#     inputs = Input(shape=(None, input_dim))
#
#     # multi-head self attention
#     attention_output = MultiHeadAttention(
#         num_heads=num_heads,
#         key_dim=input_dim,
#     )(inputs, inputs)
#     attention_output = Add()([inputs, attention_output])
#     attention_output = LayerNormalization()(attention_output)
#
#     # feed-forward network
#     ff_output = Dense(ff_dim, activation='relu')(attention_output)
#     ff_output = Dense(input_dim)(ff_output)
#     outputs = Add()([attention_output, ff_output])
#     outputs = LayerNormalization()(outputs)
#     return Model(inputs=inputs, outputs=outputs)
#
#
# # create and visualization a sample transformer encoder model
# encoder_block = transformer_encoder(
#     input_dim=64,
#     num_heads=4,
#     ff_dim=128
# )
#
# plot_model(
#     encoder_block,
#     show_shapes=True,
#     to_file='transformer_encoder_block.png',
# )



# load a pretrained BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# tokenizer a sample input text
text = 'Recurrent neural networks (RNNs) are a big method in traditional machine learning, and they were developed to handle sequential data, like sentences in text or time series. RNNs process data sequentially, maintaining an internal memory of previous inputs to influence future outputs. However, they struggle with long-range dependencies due to the vanishing gradient problem, where the influence of initial inputs diminishes in long sequences. Long short-term memory networks (LSTMs) address this limitation. LSTMs, an advanced type of RNN, have a more complex structure that includes gates to regulate the flow of information. These gates help LSTMs retain important information over long sequences, making them more effective for tasks like language modeling and text generation.'
input = tokenizer(text, return_tensors='pt')


# pass the input through the model
outputs = model(**input)
print('Hidden states shape:\n\n', outputs.last_hidden_state.shape)



# load a pretrained BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = TFBertModel.from_pretrained('bert-base-uncased')

# tokenizer a sample input text
text = 'Recurrent neural networks (RNNs) are a big method in traditional machine learning, and they were developed to handle sequential data, like sentences in text or time series. RNNs process data sequentially, maintaining an internal memory of previous inputs to influence future outputs. However, they struggle with long-range dependencies due to the vanishing gradient problem, where the influence of initial inputs diminishes in long sequences. Long short-term memory networks (LSTMs) address this limitation. LSTMs, an advanced type of RNN, have a more complex structure that includes gates to regulate the flow of information. These gates help LSTMs retain important information over long sequences, making them more effective for tasks like language modeling and text generation.'
inputs = tokenizer(text, return_tensors='tf')


# pass the input through the model
output = model(**inputs)
print('Hidden states shape:\n\n', output.last_hidden_state.shape)