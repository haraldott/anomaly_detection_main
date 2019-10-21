import numpy as np
import torch
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import Sequential

# Hyperparameters
input_size = 1
num_epochs = 300
batch_size = 1
num_classes = 2148
hidden_size = 128
window_size = 10
seq_length = 5
log = 'Adam_batch_size=' + str(batch_size) + ';epoch=' + str(num_epochs)
model_dir = 'model'

dataset = torch.load('first_1000_stacked_embeddings.pt')
dataset_numpy = []

for tensor in dataset:
    dataset_numpy.append(tensor.numpy())

n_chars = len(dataset_numpy)
data_x = []
data_y = []
for i in range(0, n_chars - seq_length):
    data_x.append(dataset_numpy[i: i + seq_length])
    data_y.append(dataset_numpy[i + seq_length])
n_patterns = len(data_x)

input_x = np.reshape(data_x, (n_patterns * len(dataset[0]), seq_length, 1))
# hier müssen die Klassen data_y zugewiesen werden...
# input_y = data_y ....

model = Sequential()
model.add(LSTM(256, input_shape=(input_x.shape[1], input_x.shape[2]), dropout=0.2, return_sequences=True))
model.add(LSTM(256, input_shape=(input_x.shape[1], input_x.shape[2]), dropout=0.2))
model.add(Dense(len(dataset[0])))
model.compile(loss='categorical_crossentropy', optimizer='adam')
filepath = "weights-improvement-{epoch:02d}-{.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

model.fit(input_x, data_y, epochs=20, batch_size=12, callbacks=callbacks_list)