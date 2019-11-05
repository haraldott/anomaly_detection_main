import os

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

import wordembeddings.createvectors as cv

if not os.path.exists('./vae_img'):
    os.mkdir('./vae_img')

num_epochs = 100
batch_size = 128
learning_rate = 1e-3

embeddings, glove = cv.create_word_vectors()
dict_size = len(glove.dictionary)
input_size = embeddings[0][0].shape[0]

sentence_lens = [len(sentence) for sentence in embeddings]
pad_vector = np.zeros(input_size)
longest_sent = max(sentence_lens)
total_batch_size = len(embeddings)
padded_embeddings = np.ones((total_batch_size, longest_sent, input_size)) * pad_vector

for i, x_len in enumerate(sentence_lens):
    sequence = embeddings[i]
    padded_embeddings[i, 0:x_len] = sequence[:x_len]

dataloader = DataLoader(padded_embeddings, batch_size=batch_size)


class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.fc1 = nn.Linear(longest_sent * input_size, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, longest_sent * input_size)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return F.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar

model = autoencoder()
model = model.double()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)


reconstruction_function = nn.MSELoss(size_average=False)

def loss_function(recon_x, x, mu, logvar):
    """
    recon_x: generating images
    x: origin images
    mu: latent mean
    logvar: latent log variance
    """
    BCE = reconstruction_function(recon_x, x)  # mse loss
    # loss = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)
    # KL divergence
    return BCE + KLD

for epoch in range(num_epochs):
    loss = 0
    model.train()
    train_loss = 0
    for batch_idx, sentence in enumerate(dataloader):
        sentence = sentence.view(sentence.size(0), -1)

        optimizer.zero_grad()
        recon_batch, mu, logvar = model(sentence)
        loss = loss_function(recon_batch, sentence, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % 100 == 0:
            print("loss: {}".format(loss.item()))
    # ===================log========================
    print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, num_epochs, loss.item()))

torch.save(model.state_dict(), './sim_autoencoder.pth')
