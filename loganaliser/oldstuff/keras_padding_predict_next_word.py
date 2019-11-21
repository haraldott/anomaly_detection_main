from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import pickle
import argparse
from keras.callbacks import ModelCheckpoint
from loganaliser.main import split

parser = argparse.ArgumentParser()
parser.add_argument('-loadglove', type=str,
                    default='../../data/openstack/utah/embeddings/glove_18k_anomalies_no_norm_and_padding.model')
parser.add_argument('-loadvectors', type=str,
                    default='../../data/openstack/utah/embeddings/glove_18k_anomalies_no_norm_and_padding.pickle')

args = parser.parse_args()

embeddings = pickle.load(open(args.loadvectors, 'rb'))
glove = pickle.load(open(args.loadglove, 'rb'))

padded_inputs = tf.keras.preprocessing.sequence.pad_sequences(embeddings, padding='post', dtype=dtype)

n_chars = len(padded_inputs)
data_x = []
data_y = []
for i in range(0, n_chars - 1):
    data_x.append(padded_inputs[i])
    data_y.append(padded_inputs[i+1])
n_patterns = len(data_x)

dict_size = len(glove.dictionary)
input_size = embeddings[0][0].shape[0]
dtype = embeddings[0][0][0].dtype


input_x = np.reshape(data_x, (n_patterns, 1, len(embeddings[0])))
output_y = np.asanyarray(data_y)

model = tf.keras.Sequential([
    layers.Embedding(input_dim=dict_size + 1, output_dim=16, mask_zero=True),
    layers.LSTM(units=256, dropout=0.5),
    layers.LSTM(units=256, dropout=0.5)
])

model.compile(loss='categorical_crossentropy', optimizer='adam')

filepath = "weights-improvement-{epoch:02d}-{.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

model.fit(input_x, output_y, epochs=100, batch_size=32, callbacks=callbacks_list)

# embedding = layers.Embedding(input_dim=dict_size+1, output_dim=16, mask_zero=True)
# masked_output = embedding(padded_inputs)
