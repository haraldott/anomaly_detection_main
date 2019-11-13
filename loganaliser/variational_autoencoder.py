import argparse
import pickle

import torch
import torch.nn.functional as F
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, random_split
import math, time

# Parse args input
parser = argparse.ArgumentParser()
parser.add_argument('-loadglove', type=str, default='../data/openstack/utah/embeddings/glove.model')
parser.add_argument('-loadvectors', type=str, default='../data/openstack/utah/embeddings/vectors.pickle')
args = parser.parse_args()
glove_load_path = args.loadglove
vectors_load_path = args.loadvectors

# load vectors and glove obj
padded_embeddings = pickle.load(open(vectors_load_path, 'rb'))
glove = pickle.load(open(glove_load_path, 'rb'))

# Hyperparameters
num_epochs = 100
batch_size = 128
learning_rate = 1e-5

dict_size = len(glove.dictionary)  # number of different words
embeddings_dim = padded_embeddings[0][0].shape[0]  # dimension of each of the word embeddings vectors
longest_sent = len(padded_embeddings[0])

val_set_len = math.floor(len(padded_embeddings) / 10)
test_set_len = math.floor(len(padded_embeddings) / 20)
train_set_len = len(padded_embeddings) - test_set_len - val_set_len
train, test, val = random_split(padded_embeddings, [train_set_len, test_set_len, val_set_len])

train_dataloader = DataLoader(train, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test, batch_size=batch_size, shuffle=False)
val_dataloader = DataLoader(val, batch_size=batch_size, shuffle=False)


# TODO: überprüfe ob dropout bei autoencoder
# TODO: eventuell autoencoder mit gru (lstm) probieren, decoder layer kann so bleiben (ggf. auch decoder anpassen,
#  falls mit linear nicht so gut funktioniert, es geht nur um die Zeit)
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
        h1 = F.dropout(h1, 0.1)
        h21 = self.fc21(h1)
        h21 = F.dropout(h21, 0.1)
        h22 = self.fc22(h1)
        h22 = F.dropout(h22, 0.1)
        return h21, h22

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
        h3 = F.dropout(h3, 0.1)
        h4 = self.fc4(h3)
        h4 = F.dropout(h4, 0.1)
        return torch.sigmoid(h4)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar


model = AutoEncoder()
model = model.double()  # TODO: take care that we use double *everywhere*, glove uses float currently
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


def train():
    model.train()
    total_loss = 0
    for sentence in train_dataloader:
        sentence = sentence.view(sentence.size(0), -1)

        optimizer.zero_grad()
        recon_batch, mu, logvar = model(sentence)
        loss = loss_function(recon_batch, sentence, mu, logvar)
        loss.backward()
        total_loss += loss.item()
        optimizer.step()


def evaluate(test_dl):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for sentence in test_dl:
            sentence = sentence.view(sentence.size(0), -1)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(sentence)
            loss = loss_function(recon_batch, sentence, mu, logvar)
            total_loss += loss.item()
    return total_loss / test_dl.dataset.dataset.shape[0]


best_val_loss = None

try:
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        model.train()
        train()
        val_loss = evaluate(val_dataloader)
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

# load best saved model and evaluate
test_model = AutoEncoder()
test_model = test_model.double()
test_model = test_model.load_state_dict(torch.load('./sim_autoencoder.pth'))
test_model.eval()

test_loss = evaluate(test_dataloader)
print('=' * 89)
print('| End of training | test loss {} | test ppl {}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)