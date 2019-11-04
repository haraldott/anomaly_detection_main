import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
import wordembeddings.createvectors as cv
import numpy as np


class LSTM(nn.Module):
    def __init__(self, nb_lstm_units=256, embedding_dim=3, batch_size=3):
        super(LSTM, self).__init__()
        self.nb_lstm_units = nb_lstm_units
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size

        # build actual NN
        self.__build_model()

    def __build_model(self):

        # design LSTM
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
        #y = y.view(-1)
        #y_pred = y_pred.view(-1)

        mask_y = y * (y != np.zeros(input_size))
        mask_y_pred = y_pred * (y_pred != np.zeros(input_size))
        distance(mask_y, mask_y_pred)

        return distance


num_epochs = 3

embeddings, glove = cv.create_word_vectors()
dict_len = len(glove.dictionary)
input_size = embeddings[0][0].shape[0]

sentence_lens = [len(sentence) for sentence in embeddings]
pad_vector = np.zeros(input_size)
longest_sent = max(sentence_lens)
batch_size = len(embeddings)
padded_embeddings = np.ones((batch_size, longest_sent, input_size)) * pad_vector

for i, x_len in enumerate(sentence_lens):
    sequence = embeddings[i]
    padded_embeddings[i, 0:x_len] = sequence[:x_len]

number_of_sentences = len(embeddings)

# samples, timesteps, features
#input_x = np.reshape(data_x, (n_patterns, seq_length, input_size))
#output_y = data_y

model = LSTM()
optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-5)
distance = nn.MSELoss()

for epoch in range(num_epochs):
    for j, sentence in enumerate(padded_embeddings):
        # vec = torch.from_numpy(vec)
        # ===================forward=====================
        output = model(torch.as_tensor(sentence), sentence_lens[j])
        loss = distance(output, padded_embeddings[j+1])
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # ===================log========================
    print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, num_epochs, loss.data()))
