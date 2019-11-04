from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import wordembeddings.createvectors as cv

embeddings, glove = cv.create_word_vectors()
dict_size = len(glove.dictionary)
input_size = embeddings[0][0].shape[0]
dtype = embeddings[0][0][0].dtype

padded_inputs = tf.keras.preprocessing.sequence.pad_sequences(embeddings, padding='post', dtype=dtype)

embedding = layers.Embedding(input_dim=dict_size+1, output_dim=16, mask_zero=True)
masked_output = embedding(padded_inputs)
