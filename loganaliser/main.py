import math
import pickle
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim

import loganaliser.model as lstm_model
from loganaliser.vanilla_autoencoder import AutoEncoder


class AnomalyDetection:
    def __init__(self,
                 model,
                 loadvectors='../data/openstack/utah/padded_embeddings_pickle/openstack_52k_normal.pickle',
                 loadautoencodermodel='saved_models/18k_anomalies_autoencoder.pth',
                 savemodelpath='saved_models/lstm.pth',
                 n_layers=3,
                 n_hidden_units=250,
                 seq_length=7,
                 num_epochs=0,
                 learning_rate=1e-5,
                 batch_size=20,
                 folds=5,
                 clip=0.25,
                 train_mode=False
                 ):
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
        self.model = model
        self.train_mode = train_mode

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if model == 'glove':
            self.data_x, self.data_y, self.feature_length = self.prepare_data_latent_space()
        elif model == 'bert' or model == 'embeddings_layer':
            self.data_x, self.data_y, self.feature_length = self.prepare_data_raw()

        self.model = lstm_model.LSTM(n_input=self.feature_length,
                                     n_hidden_units=self.n_hidden_units,
                                     n_layers=self.n_layers,
                                     train_mode=self.train_mode).to(self.device)
        # self.model = self.model.double()  # TODO: check this double stuff
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min')
        # optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        #  überprüfe was mse genau macht, abspeichern
        #  zb jede 10. epoche die distanz plotten
        #  quadrat mean squared error mal probieren
        self.distance = nn.MSELoss()

    def prepare_data_raw(self):
        embeddings = pickle.load(open(self.loadvectors, 'rb'))
        number_of_sentences = len(embeddings)
        feature_length = embeddings[0].size(0)

        data_x = []
        data_y = []
        for i in range(0, number_of_sentences - self.seq_length - 1):
            data_x.append(embeddings[i: i + self.seq_length])
            data_y.append(embeddings[i + 1 + self.seq_length])

        data_x = torch.stack(data_x).to(self.device)
        data_y = torch.stack(data_y).to(self.device)

        return data_x, data_y, feature_length

    def prepare_data_latent_space(self):
        # load vectors and glove obj
        padded_embeddings = pickle.load(open(self.loadvectors, 'rb'))

        sentence_lens = [len(sentence) for sentence in
                         padded_embeddings]  # how many words a log line consists of, without padding
        longest_sent = max(sentence_lens)
        embeddings_dim = padded_embeddings[0][0].shape[0]  # dimension of each of the word embeddings vectors

        # load the AutoEncoder model
        autoencoder_model = AutoEncoder(longest_sent, embeddings_dim, train_mode=False)
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
            data_y.append(latent_space_representation_of_padded_embeddings[i + self.seq_length])

        data_x = torch.Tensor(data_x).to(self.device)
        data_y = torch.Tensor(data_y).to(self.device)

        # samples, timesteps, features
        # dataloader_x = DataLoader(data_x, batch_size=64)
        # dataloader_y = DataLoader(data_y, batch_size=64)
        # data_x = np.reshape(data_x, (n_patterns, self.seq_length, feature_length))

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
        dataloader_x = DataLoader(self.data_x[idx], batch_size=self.batch_size, drop_last=True)
        dataloader_y = DataLoader(self.data_y[idx], batch_size=self.batch_size, drop_last=True)
        loss_distribution = []
        total_loss = 0
        hidden = self.model.init_hidden(self.batch_size, self.device)  # TODO: check stuff with batch size
        with torch.no_grad():
            for data, target in zip(dataloader_x, dataloader_y):
                prediction, hidden = self.model(data, hidden)
                hidden = self.repackage_hidden(hidden)
                loss = self.distance(prediction.reshape(-1), target.reshape(-1))
                loss_distribution.append(loss.item())
                total_loss += loss.item()
        return total_loss / len(idx), loss_distribution

    def predict(self, idx):
        self.model.eval()
        # TODO: since we want to predict *every* loss of every line, we don't use batches, so here we use batch_size
        #   1 is this ok?
        hidden = self.model.init_hidden(1, self.device)
        loss_distribution = []
        with torch.no_grad():
            for data, target in zip(self.data_x[idx], self.data_y[idx]):
                data = data.view(1, self.seq_length, self.feature_length)
                prediction, hidden = self.model(data, hidden)
                hidden = self.repackage_hidden(hidden)
                loss = self.distance(prediction.reshape(-1), target.reshape(-1))  # TODO check if reshape is necessary
                loss_distribution.append(loss.item())
        return loss_distribution

    def train(self, idx):
        self.model.train()
        dataloader_x = DataLoader(self.data_x[idx], batch_size=self.batch_size, drop_last=True)
        dataloader_y = DataLoader(self.data_y[idx], batch_size=self.batch_size, drop_last=True)
        hidden = self.model.init_hidden(self.batch_size, self.device)  # TODO: check stuff with batch size
        for data, target in zip(dataloader_x, dataloader_y):
            self.optimizer.zero_grad()
            hidden = self.repackage_hidden(hidden)
            prediction, hidden = self.model(data, hidden)
            loss = self.distance(prediction.reshape(-1), target.reshape(-1))
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
            for p in self.model.parameters():
                p.data.add_(-self.learning_rate, p.grad.data)

    def start_training(self):
        best_val_loss = None
        try:
            loss_values = []
            indices_generator = self.split(self.data_x, self.folds)
            for epoch in range(1, self.num_epochs + 1):
                val_loss = 0
                indices_generator = self.split(self.data_x, self.folds)
                epoch_start_time = time.time()
                for i in range(0, self.folds - 1):
                    indices = next(indices_generator)
                    train_incides = indices[0]
                    eval_indices = indices[1]

                    self.train(train_incides)
                    this_loss, _ = self.evaluate(eval_indices)
                    self.optimizer.step()
                    self.scheduler.step(this_loss)
                    val_loss += this_loss
                print('-' * 89)
                print('LSTM: | end of epoch {:3d} | time: {:5.2f}s | valid loss {} | '
                      'valid ppl {}'.format(epoch, (time.time() - epoch_start_time),
                                            val_loss, math.exp(val_loss)))
                print('-' * 89)
                if not best_val_loss or val_loss < best_val_loss:
                    torch.save(self.model.state_dict(), self.savemodelpath)
                    best_val_loss = val_loss
                loss_values.append(val_loss / self.folds)
        except KeyboardInterrupt:
            print('-' * 89)
            print('Exiting from training early')
        return indices_generator

    def loss_values(self, indices_generator=None, normal: bool = True):
        model = lstm_model.LSTM(self.feature_length, self.n_hidden_units, self.n_layers, train_mode=False)
        model.load_state_dict(torch.load(self.savemodelpath))
        model.eval()
        if normal:
            assert indices_generator is not None, "indices_generator should not be None when in normal mode"
            loss_values = []
            indices = next(indices_generator)
            indices = np.concatenate([indices[0], indices[1]])
            loss_distribution = self.predict(indices)
            loss_values.append(loss_distribution)
        else:
            n_samples = len(self.data_x)
            indices_containing_anomalies = np.arange(n_samples)
            drop_last = len(indices_containing_anomalies) % self.batch_size - 1
            indices_containing_anomalies = indices_containing_anomalies[:-drop_last]
            loss_values = self.predict(indices_containing_anomalies)

        loss_values = np.array(loss_values)
        loss_values = loss_values.flatten()
        return loss_values


if __name__ == '__main__':
    ad = AnomalyDetection()
    ad.start_training()
