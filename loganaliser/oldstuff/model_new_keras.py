from keras.layers import Embedding, LSTM, Input, RepeatVector
from keras import Model
import wordembeddings.createvectors as cv

embeddings, dict_len = cv.create_word_vectors()

HIDDEN_UNITS = 512
SEQUENCE_LEN = 100
EMBED_SIZE = embeddings[0][0].shape
NUM_EPOCHS = 1000
# Embedding input_dim:      vocabulary or the total number of unique words in a corpus
#           output_dim:     number of the dimensions for each word vector
#           input_length:   length of the input sentence
embedding_layer = Embedding(input_dim=dict_len,
                            output_dim=EMBED_SIZE,
                            input_length=1)
input = Input(shape=(SEQUENCE_LEN,), name="input")
embedding_layer = embedding_layer(input)
encoder = LSTM(HIDDEN_UNITS, name="encoder")(embedding_layer)
decoder = RepeatVector(SEQUENCE_LEN)(encoder)
decoder = LSTM(EMBED_SIZE, return_sequences=True)(decoder)
autoencoder = Model(input, decoder)
autoencoder.compile(optimizer="adam", loss="mse")
autoencoder.fit(train, epochs=NUM_EPOCHS, callbacks=[checkpoint])
encoder = Model(autoencoder.input, autoencoder.get_layer("encoder").output)
