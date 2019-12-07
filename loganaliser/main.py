import argparse
import math
import pickle
import time
import adabound

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from loganaliser.vanilla_autoencoder import AutoEncoder
import matplotlib.pyplot as plt
import loganaliser.model as lstm_model
from scipy import stats
from torch import optim
import os


class AnomalyDetection:
    def __init__(self,
                 loadvectors='../data/openstack/utah/padded_embeddings_pickle/openstack_52k_normal_embeddings.pickle',
                 loadautoencodermodel='saved_models/18k_anomalies_autoencoder.pth',
                 savemodelpath='saved_models/lstm.pth',
                 n_layers=4,
                 n_hidden_units=200,
                 seq_length=7,
                 num_epochs=100,
                 learning_rate=1e-5,
                 batch_size=20,
                 folds=4,
                 clip=0.25
                 ):
        os.chdir(os.path.dirname(__file__))
        self.loadvectors = loadvectors
        self.loadautoencodermodel = loadautoencodermodel
        self.savemodelpath = savemodelpath
        self.n_layers = n_layers
        self.n_hidden_units = n_hidden_units
        self.seq_length = seq_length
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.folds = folds
        self.clip = clip

        self.data_x, self.data_y, self.feature_length = self.prepare_data()
        self.model = lstm_model.LSTM(self.feature_length, self.n_hidden_units, self.n_layers)
        self.model = self.model.double()  # TODO: check this double stuff
        self.optimizer = adabound.AdaBound(self.model.parameters(), lr=self.learning_rate)
        # optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        #  überprüfe was mse genau macht, abspeichern
        #  zb jede 10. epoche die distanz plotten
        #  quadrat mean squared error mal probieren
        self.distance = nn.MSELoss()

    def prepare_data(self):
        # load vectors and glove obj
        padded_embeddings = pickle.load(open(self.loadvectors, 'rb'))

        sentence_lens = [len(sentence) for sentence in
                         padded_embeddings]  # how many words a log line consists of, without padding
        longest_sent = max(sentence_lens)
        embeddings_dim = padded_embeddings[0][0].shape[0]  # dimension of each of the word embeddings vectors

        # load the AutoEncoder model
        autoencoder_model = AutoEncoder(longest_sent, embeddings_dim)
        autoencoder_model.double()
        autoencoder_model.load_state_dict(torch.load(self.loadautoencodermodel))
        autoencoder_model.eval()

        # use the loaded AutoEncoder model, to receive the latent space representation of the padded embeddings
        latent_space_representation_of_padded_embeddings = []
        for sentence in padded_embeddings:
            sentence = torch.from_numpy(sentence)
            sentence = sentence.reshape(-1)
            encoded_sentence = autoencoder_model.encode(sentence)
            latent_space_representation_of_padded_embeddings.append(encoded_sentence.detach().numpy())

        number_of_sentences = len(latent_space_representation_of_padded_embeddings)
        feature_length = latent_space_representation_of_padded_embeddings[0].size

        data_x = []
        data_y = []
        # for i in range(0, number_of_sentences - 1):
        #     data_x.append(latent_space_representation_of_padded_embeddings[i])
        #     data_y.append(latent_space_representation_of_padded_embeddings[i + 1])
        for i in range(0, number_of_sentences - self.seq_length):
            data_x.append(latent_space_representation_of_padded_embeddings[i: i + self.seq_length])
            data_y.append(latent_space_representation_of_padded_embeddings[i + 1: i + 1 + self.seq_length])
        n_patterns = len(data_x)

        data_x = torch.Tensor(data_x)
        data_y = torch.Tensor(data_y)

        # samples, timesteps, features
        # dataloader_x = DataLoader(data_x, batch_size=64)
        # dataloader_y = DataLoader(data_y, batch_size=64)
        data_x = np.reshape(data_x, (n_patterns, self.seq_length, feature_length))
        return data_x, data_y, feature_length

    def split(self, x, n_splits):
        n_samples = len(x)
        k_fold_size = n_samples // n_splits
        indices = np.arange(n_samples)

        margin = 0
        for i in range(n_splits):
            start = i * k_fold_size
            stop = start + k_fold_size
            mid = int(0.8 * (stop - start)) + start
            yield indices[start: mid], indices[mid + margin: stop]

    def repackage_hidden(self, h):
        """Wraps hidden states in new Tensors, to detach them from their history."""

        if isinstance(h, torch.Tensor):
            return h.detach()
        else:
            return tuple(self.repackage_hidden(v) for v in h)

    def evaluate(self, idx):
        self.model.eval()
        dataloader_x = DataLoader(self.data_x[idx], batch_size=self.batch_size)
        dataloader_y = DataLoader(self.data_y[idx], batch_size=self.batch_size)
        loss_distribution = []
        total_loss = 0
        hidden = self.model.init_hidden(self.seq_length)  # TODO: check stuff with batch size
        with torch.no_grad():
            for data, target in zip(dataloader_x, dataloader_y):
                prediction, hidden = self.model(data, hidden)
                hidden = self.repackage_hidden(hidden)
                loss = self.distance(prediction.view(-1), target.view(-1))
                loss_distribution.append(loss.item())
                total_loss += loss.item()
        return total_loss / len(idx), loss_distribution

    def train(self, idx):
        self.model.train()
        dataloader_x = DataLoader(self.data_x[idx], batch_size=self.batch_size)
        dataloader_y = DataLoader(self.data_y[idx], batch_size=self.batch_size)
        hidden = self.model.init_hidden(self.seq_length)  # TODO: check stuff with batch size
        for data, target in zip(dataloader_x, dataloader_y):
            self.optimizer.zero_grad()
            hidden = self.repackage_hidden(hidden)
            prediction, hidden = self.model(data, hidden)

            loss = self.distance(prediction.view(-1), target.view(-1))
            loss.backward()
            self.optimizer.step()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
            for p in self.model.parameters():
                p.data.add_(-self.learning_rate, p.grad.data)

    def start_training(self):
        best_val_loss = None
        try:
            loss_values = []
            for epoch in range(self.num_epochs):
                val_loss = 0
                indices_generator = self.split(self.data_x, self.folds)
                epoch_start_time = time.time()
                for i in range(0, self.folds):
                    indices = next(indices_generator)
                    train_incides = indices[0]
                    eval_indices = indices[1]

                    self.train(train_incides)
                    this_loss, _ = self.evaluate(eval_indices)
                    val_loss += this_loss
                print('-' * 89)
                print('LSTM: | end of epoch {:3d} | time: {:5.2f}s | valid loss {} | '
                      'valid ppl {}'.format(epoch, (time.time() - epoch_start_time),
                                            val_loss, math.exp(val_loss)))
                print('-' * 89)
                if not best_val_loss or val_loss < best_val_loss:
                    torch.save(self.model.state_dict(), self.savemodelpath)
                    best_val_loss = val_loss
                else:
                    # anneal learning rate
                    self.learning_rate /= 2.0
                    print("anneal lr to: {}".format(self.learning_rate))
                loss_values.append(val_loss / self.folds)
            plt.plot(loss_values)
        except KeyboardInterrupt:
            print('-' * 89)
            print('Exiting from training early')

    def loss_values(self, normal: bool = True):
        model = lstm_model.LSTM(self.feature_length, self.n_hidden_units, self.n_layers)
        model.load_state_dict(torch.load(self.savemodelpath))
        model.eval()
        if normal:
            loss_values = []
            indices = [x for x in self.split(self.data_x, self.folds)]
            for idx in indices:
                eval_indices = idx[1]
                _, loss_distribution = self.evaluate(eval_indices)
                loss_values.append(loss_distribution)
        else:
            n_samples = len(self.data_x)
            indices_containing_anomalies = np.arange(n_samples)
            _, loss_values = self.evaluate(indices_containing_anomalies)

        loss_values = np.array(loss_values)
        loss_values = loss_values.flatten()
        return loss_values


if __name__ == '__main__':
    ad = AnomalyDetection()
    ad.start_training()
