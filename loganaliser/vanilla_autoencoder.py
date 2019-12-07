import math
import pickle
import time

import adabound
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, random_split
import os


# TODO: überprüfe ob dropout bei autoencoder
# TODO: eventuell autoencoder mit gru (lstm) probieren, decoder layer kann so bleiben (ggf. auch decoder anpassen,
#  falls mit linear nicht so gut funktioniert, es geht nur um die Zeit)
class AutoEncoder(nn.Module):
    def __init__(self, longest_sent, embeddings_dim):
        super(AutoEncoder, self).__init__()
        self.fc1 = nn.Linear(longest_sent * embeddings_dim, 400)
        self.fc2 = nn.Linear(400, 128)
        self.fc3 = nn.Linear(128, 400)
        self.fc4 = nn.Linear(400, longest_sent * embeddings_dim)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        h1 = F.dropout(h1, 0.1)
        h2 = self.fc2(h1)
        h2 = F.dropout(h2, 0.1)
        return h2

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        h3 = F.dropout(h3, 0.1)
        h4 = self.fc4(h3)
        h4 = F.dropout(h4, 0.1)
        return torch.sigmoid(h4)

    def forward(self, x):
        latent_space = self.encode(x)
        return self.decode(latent_space)


class VanillaAutoEncoder:
    def __init__(self,
                 load_vectors='../data/openstack/utah/padded_embeddings_pickle/openstack_52k_normal_embeddings.pickle',
                 model_save_path='saved_models/18k_anomalies_autoencoder.pth',
                 learning_rate=1e-6,
                 batch_size=64,
                 num_epochs=100):
        os.chdir(os.path.dirname(__file__))
        self.load_vectors = load_vectors
        self.model_save_path = model_save_path
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs

        # load vectors and glove obj
        padded_embeddings = pickle.load(open(self.load_vectors, 'rb'))

        self.embeddings_dim = padded_embeddings[0][0].shape[0]  # dimension of each of the word embeddings vectors
        self.longest_sent = len(padded_embeddings[0])

        val_set_len = math.floor(len(padded_embeddings) / 10)
        test_set_len = math.floor(len(padded_embeddings) / 20)
        train_set_len = len(padded_embeddings) - test_set_len - val_set_len
        train, test, val = random_split(padded_embeddings, [train_set_len, test_set_len, val_set_len])

        self.train_dataloader = DataLoader(train, batch_size=self.batch_size, shuffle=True)
        self.test_dataloader = DataLoader(test, batch_size=self.batch_size, shuffle=False)
        self.val_dataloader = DataLoader(val, batch_size=self.batch_size, shuffle=False)

        self.model = AutoEncoder(self.longest_sent, self.embeddings_dim)
        if torch.cuda.is_available():
            self.model.cuda()
        self.model.double()  # TODO: take care that we use double *everywhere*, glove uses float currently
        self.criterion = nn.MSELoss()
        self.optimizer = adabound.AdaBound(self.model.parameters(), lr=self.learning_rate)

        # model.load_state_dict(torch.load('./sim_autoencoder.pth'))
        # model.eval()
        self.start()

    def train(self):
        self.model.train()
        for sentence in self.train_dataloader:
            sentence = sentence.view(sentence.size(0), -1)
            if torch.cuda.is_available():
                sentence = sentence.cuda()
            self.optimizer.zero_grad()
            output = self.model(sentence)
            loss = self.criterion(output, sentence)
            loss.backward()
            self.optimizer.step()

    def evaluate(self, test_dl):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for sentence in test_dl:
                sentence = sentence.view(sentence.size(0), -1)
                if torch.cuda.is_available():
                    sentence = sentence.cuda()
                output = self.model(sentence)
                loss = self.criterion(output, sentence)
                total_loss += loss.item()
        return total_loss / test_dl.dataset.dataset.shape[0]

    def start(self):
        best_val_loss = None
        anneal_count = 0
        try:
            if anneal_count < 6:
                for epoch in range(self.num_epochs):
                    epoch_start_time = time.time()
                    self.train()
                    val_loss = self.evaluate(self.val_dataloader)
                    print('-' * 89)
                    print('AE: | end of epoch {:3d} | time: {:5.2f}s | valid loss {} | '
                          'valid ppl {}'.format(epoch, (time.time() - epoch_start_time),
                                                val_loss, math.exp(val_loss)))
                    print('-' * 89)
                    if not best_val_loss or val_loss < best_val_loss:
                        torch.save(self.model.state_dict(), self.model_save_path)
                        best_val_loss = val_loss
                    else:
                        # anneal learning rate
                        print("anneal")
                        # TODO: actually do the annealing
                        anneal_count += 1
                        self.learning_rate /= 2.0
            else:
                print('Learning rate has been annealed {} times. Ending training'.format(anneal_count))
        except KeyboardInterrupt:
            print('-' * 89)
            print('Exiting from training early')

        # load best saved model and evaluate
        test_model = AutoEncoder(self.longest_sent, self.embeddings_dim)
        test_model.double()
        test_model.load_state_dict(torch.load(self.model_save_path))
        test_model.eval()

        test_loss = self.evaluate(self.test_dataloader)
        print('=' * 89)
        print('| End of training | test loss {} | test ppl {}'.format(
            test_loss, math.exp(test_loss)))
        print('=' * 89)


if __name__ == '__main__':
    vae = VanillaAutoEncoder()
    vae.start()
