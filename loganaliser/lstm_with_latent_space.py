import argparse
import math
import pickle
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from loganaliser.variational_autoencoder import AutoEncoder


def split(x, n_splits):
    n_samples = len(x)
    k_fold_size = n_samples // n_splits
    indices = np.arange(n_samples)

    margin = 0
    for i in range(n_splits):
        start = i * k_fold_size
        stop = start + k_fold_size
        mid = int(0.8 * (stop - start)) + start
        yield indices[start: mid], indices[mid + margin: stop]


class LSTM(nn.Module):
    def __init__(self, embedding_dim, nb_lstm_units=256, batch_size=5, dropout=0.2):
        super(LSTM, self).__init__()
        self.nb_lstm_units = nb_lstm_units
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size

        self.lstm = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.nb_lstm_units,
            num_layers=2,
            dropout=dropout
        )

        self.linear = nn.Linear(self.nb_lstm_units, self.embedding_dim)

        # output layer which projects back to tag space
        # self.hidden_to_tag = nn.Linear(self.nb_lstm_units, self.nb_tags)

    def forward(self, input):
        h1, _ = self.lstm(input)
        pred = self.linear(h1)
        return pred

    def loss(self, y_pred, y):
        # y = y.view(-1)
        # y_pred = y_pred.view(-1)

        mask_y = y * (y != np.zeros(embeddings_dim))
        mask_y_pred = y_pred * (y_pred != np.zeros(embeddings_dim))
        distance(mask_y, mask_y_pred)

        return distance


# Parse args input
parser = argparse.ArgumentParser()
parser.add_argument('-loadglove', type=str, default='../data/openstack/utah/embeddings/glove.model')
parser.add_argument('-loadvectors', type=str, default='../data/openstack/utah/embeddings/vectors.pickle')
parser.add_argument('-loadautoencodermodel', type=str, default='sim_autoencoder.pth')
parser.add_argument('-seq-length')
args = parser.parse_args()

# load vectors and glove obj
padded_embeddings = pickle.load(open(args.loadvectors, 'rb'))
glove = pickle.load(open(args.loadglove, 'rb'))

# Hyperparamters
seq_length = 5
num_epochs = 100
learning_rate = 1e-5
batch_size = 8
folds = 5

dict_size = len(glove.dictionary)  # number of different words
embeddings_dim = padded_embeddings[0][0].shape[0]  # dimension of each of the word embeddings vectors
sentence_lens = [len(sentence) for sentence in
                 padded_embeddings]  # how many words a log line consists of, without padding
longest_sent = max(sentence_lens)  # length of the longest sentence

# load the AutoEncoder model
autoencoder_model = AutoEncoder()
autoencoder_model.double()
autoencoder_model.load_state_dict(torch.load(args.loadautoencodermodel))
autoencoder_model.eval()

# use the loaded AutoEncoder model, to receive the latent space representation of the padded embeddings
latent_space_representation_of_padded_embeddings = []
for sentence in padded_embeddings:
    sentence = torch.from_numpy(sentence)
    sentence = sentence.reshape(-1)
    encoded_sentence, _ = autoencoder_model.encode(sentence)
    latent_space_representation_of_padded_embeddings.append(encoded_sentence.detach().numpy())

number_of_sentences = len(latent_space_representation_of_padded_embeddings)
feature_length = latent_space_representation_of_padded_embeddings[0].size

data_x = []
data_y = []
for i in range(0, number_of_sentences - seq_length):
    data_x.append(latent_space_representation_of_padded_embeddings[i: i + seq_length])
    data_y.append(latent_space_representation_of_padded_embeddings[i + 1: i + 1 + seq_length])
n_patterns = len(data_x)

data_x = torch.Tensor(data_x)
data_y = torch.Tensor(data_y)

data_x_k_fold = split(data_x, 5)

# samples, timesteps, features
# dataloader_x = DataLoader(data_x, batch_size=64)
# dataloader_y = DataLoader(data_y, batch_size=64)
input_x = np.reshape(data_x, (n_patterns, seq_length, feature_length))

model = LSTM(embedding_dim=feature_length)
optimizer = torch.optim.Adam(model.parameters(), weight_decay=learning_rate)
#  check if softmax is applied on loss, if not do it yourself
#  überprüfe was mse genau macht, abspeichern
#  zb jede 10. epoche die distanz plotten
#  quadrat mean squared error mal probieren
distance = nn.MSELoss()


def evaluate(idx):
    model.eval()
    dataloader_x = DataLoader(data_x[idx])
    dataloader_y = DataLoader(data_y[idx])
    total_loss = 0
    with torch.no_grad():
        for x, target in zip(dataloader_x, dataloader_y):
            prediction = model(x)
            loss = distance(prediction.view(-1), target.view(-1))
            total_loss += loss.item()
    return total_loss / len(indices)  # TODO: check this


def train(idx):
    model.train()
    dataloader_x = DataLoader(data_x[idx])
    dataloader_y = DataLoader(data_y[idx])
    for x, target in zip(dataloader_x, dataloader_y):
        optimizer.zero_grad()
        prediction = model(x)
        loss = distance(prediction.view(-1), target.view(-1))
        loss.backward()
        optimizer.step()


indices_generator = split(data_x, 5)
best_val_loss = None
val_loss = 0  # TODO: make sure that this is correct

try:
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        for i in range(0, folds):
            indices = next(indices_generator)
            train_incides = indices[0]
            eval_indices = indices[1]

            train(train_incides)
            val_los = + evaluate(eval_indices)

        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {} | '
              'valid ppl {}'.format(epoch, (time.time() - epoch_start_time),
                                    val_loss, math.exp(val_loss)))
        print('-' * 89)
        if not best_val_loss or val_loss < best_val_loss:
            torch.save(model.state_dict(), './sim_autoencoder.pth')
        else:
            # anneal learning rate
            print("annealeating")
            learning_rate /= 2.0
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')
