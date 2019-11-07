import argparse
import pickle

import numpy as np
import torch
import torch.nn as nn

from loganaliser.variational_autoencoder import AutoEncoder, pad_embeddings


class LSTM(nn.Module):
    def __init__(self, nb_lstm_units=256, embedding_dim=3, batch_size=5):
        super(LSTM, self).__init__()
        self.nb_lstm_units = nb_lstm_units
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size

        self.lstm = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.nb_lstm_units,
            num_layers=2,
            batch_first=True,
        )

        # output layer which projects back to tag space
        # self.hidden_to_tag = nn.Linear(self.nb_lstm_units, self.nb_tags)

    def forward(self, input, sentence_lens):
        # TODO: check if enforce_sorted=True is correct
        input = torch.nn.utils.rnn.pack_padded_sequence(input, sentence_lens, batch_first=True)
        # TODO: check if this is correct (comments of article)
        pred, _ = self.lstm(input)
        pred, _ = torch.nn.utils.rnn.pad_packed_sequence(pred, batch_first=True)

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
args = parser.parse_args()
glove_load_path = args.loadglove
vectors_load_path = args.loadvectors
autoencoder_model_path = args.loadautoencodermodel

# load vectors and glove obj
embeddings = pickle.load(open(vectors_load_path, 'rb'))
glove = pickle.load(open(glove_load_path, 'rb'))

# Hyperparamters
seq_length = 5
num_epochs = 100
learning_rate = 1e-5

dict_size = len(glove.dictionary)  # number of different words
embeddings_dim = embeddings[0][0].shape[0]  # dimension of each of the word embeddings vectors
sentence_lens = [len(sentence) for sentence in embeddings]  # how many words a log line consists of, without padding
longest_sent = max(sentence_lens)  # length of the longest sentence
padded_embeddings = pad_embeddings(embeddings, sentence_lens, embeddings_dim)

# load the AutoEncoder model
autoencoder_model = AutoEncoder()
autoencoder_model = autoencoder_model.double()
autoencoder_model.load_state_dict(torch.load(autoencoder_model_path))
autoencoder_model.eval()

# use the loaded AutoEncoder model, to receive the latent space representation of the padded embeddings
latent_space_representation_of_padded_embeddings = \
    [autoencoder_model.encode(sentence)[0] for sentence in padded_embeddings]

number_of_sentences = len(latent_space_representation_of_padded_embeddings)
data_x = []
data_y = []
for i in range(0, number_of_sentences - seq_length):
    data_x.append(np.asanyarray(latent_space_representation_of_padded_embeddings[i: i + seq_length]))
    data_y.append(np.asanyarray(latent_space_representation_of_padded_embeddings[i + seq_length]))
n_patterns = len(data_x)

# samples, timesteps, features
# data_x = np.asanyarray(data_x)
input_x = np.reshape(data_x, (n_patterns, seq_length, embeddings_dim))
output_y = data_y

model = LSTM()
optimizer = torch.optim.Adam(model.parameters(), weight_decay=learning_rate)
distance = nn.MSELoss()



for epoch in range(num_epochs):
    for j, sentence in enumerate(padded_embeddings):
        # vec = torch.from_numpy(vec)
        # forward
        output = model(torch.as_tensor(sentence), sentence_lens[j])
        loss = distance(output, padded_embeddings[j + 1])
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, num_epochs, loss.data()))
