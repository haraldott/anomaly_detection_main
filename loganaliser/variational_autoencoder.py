import argparse
import pickle

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader


def pad_embeddings(emb, sentlens, emb_dim):
    total_batch_size = len(emb)
    pad_vector = np.zeros(emb_dim)
    padded_emb = np.ones((total_batch_size, longest_sent, emb_dim)) * pad_vector
    for i, x_len in enumerate(sentlens):
        sequence = embeddings[i]
        padded_emb[i, 0:x_len] = sequence[:x_len]

    return padded_emb


# Parse args input
parser = argparse.ArgumentParser()
parser.add_argument('-loadglove', type=str, default='../data/openstack/utah/embeddings/glove.model')
parser.add_argument('-loadvectors', type=str, default='../data/openstack/utah/embeddings/vectors.pickle')
args = parser.parse_args()
glove_load_path = args.loadglove
vectors_load_path = args.loadvectors

# load vectors and glove obj
embeddings = pickle.load(open(vectors_load_path, 'rb'))
glove = pickle.load(open(glove_load_path, 'rb'))

# Hyperparameters
num_epochs = 100
batch_size = 128
learning_rate = 1e-5

dict_size = len(glove.dictionary)  # number of different words
embeddings_dim = embeddings[0][0].shape[0]  # dimension of each of the word embeddings vectors
sentence_lens = [len(sentence) for sentence in embeddings]  # how many words a log line consists of, without padding
longest_sent = max(sentence_lens)
padded_embeddings = pad_embeddings(embeddings, sentence_lens, embeddings_dim)

dataloader = DataLoader(padded_embeddings, batch_size=batch_size)


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.fc1 = nn.Linear(longest_sent * embeddings_dim, 400)
        self.fc21 = nn.Linear(400, 64)
        self.fc22 = nn.Linear(400, 64)
        self.fc3 = nn.Linear(64, 400)
        self.fc4 = nn.Linear(400, longest_sent * embeddings_dim)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar


model = AutoEncoder()
model = model.double()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# model.load_state_dict(torch.load('./sim_autoencoder.pth'))
# model.eval()

reconstruction_function = nn.MSELoss()


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


# for epoch in range(num_epochs):
#     loss = 0
#     model.train()
#     train_loss = 0
#     for sentence in dataloader:
#         sentence = sentence.view(sentence.size(0), -1)
#
#         optimizer.zero_grad()
#         recon_batch, mu, logvar = model(sentence)
#         loss = loss_function(recon_batch, sentence, mu, logvar)
#         loss.backward()
#         train_loss += loss.item()
#         optimizer.step()
#     print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, num_epochs, loss.item()))
#
# torch.save(model.state_dict(), './sim_autoencoder.pth')
