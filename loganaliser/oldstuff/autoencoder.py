import os

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.utils import save_image

import wordembeddings.createvectors as cv

if not os.path.exists('./vae_img'):
    os.mkdir('./vae_img')

num_epochs = 100
batch_size = 128
learning_rate = 1e-3

embeddings, glove = cv.create_word_vectors()
dict_size = len(glove.dictionary)
input_dimension = embeddings[0][0].shape[0]

sentence_lens = [len(sentence) for sentence in embeddings]
pad_vector = np.zeros(input_dimension)
longest_sent = max(sentence_lens)
total_batch_size = len(embeddings)
padded_embeddings = np.ones((total_batch_size, longest_sent, input_dimension)) * pad_vector

for i, x_len in enumerate(sentence_lens):
    sequence = embeddings[i]
    padded_embeddings[i, 0:x_len] = sequence[:x_len]

dataloader = DataLoader(padded_embeddings, batch_size=batch_size)

# TODO: überprüfe ob dropout bei autoencoder
# TODO: eventuell autoencoder mit gru (lstm) probieren, decoder layer kann so bleiben (ggf. auch decoder anpassen,
#  falls mit linear nicht so gut funktioniert, es geht nur um die Zeit)
class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(longest_sent * input_dimension, 512),
            nn.ReLU(True),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Linear(128, 64))
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, longest_sent * input_dimension), nn.Tanh())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

model = autoencoder()
model = model.double() # TODO: take care that we use double *everywhere*
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=learning_rate, weight_decay=1e-4)

for epoch in range(num_epochs):
    loss = 0
    for sentence in dataloader:
        sentence = sentence.view(sentence.size(0), -1)

        output = model(sentence)
        loss = criterion(output, sentence)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # ===================log========================
    print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, num_epochs, loss.item()))

torch.save(model.state_dict(), './sim_autoencoder.pth')
