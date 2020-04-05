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
                 n_classes,
                 target_labels,
                 debug_embeddings,
                 results_dir=None,
                 embeddings_model='glove',
                 loadvectors='../data/openstack/utah/padded_embeddings_pickle/openstack_52k_normal.pickle',
                 loadautoencodermodel='saved_models/openstack_52k_normal_vae.pth',
                 savemodelpath='saved_models/lstm.pth',
                 n_layers=1,
                 n_hidden_units=128,
                 seq_length=7,
                 num_epochs=1000,
                 learning_rate=1e-4,
                 batch_size=128,
                 folds=5,
                 clip=1.0,
                 train_mode=False,
                 instance_information_file=None,
                 anomalies_run=False,
                 transfer_learning=False
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
        self.train_mode = train_mode
        self.instance_information_file = instance_information_file
        self.anomalies_run = anomalies_run
        self.results_dir = results_dir

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.target_labels = target_labels
        self.n_classes = n_classes
        self.debug_embeddings = debug_embeddings

        # select word embeddings
        if instance_information_file is None:
            if embeddings_model == 'glove':
                self.data_x, self.data_y, self.feature_length = self.prepare_data_latent_space()
            elif embeddings_model == 'bert' or embeddings_model == 'embeddings_layer':
                self.data_x, self.data_y, self.feature_length = self.prepare_data_raw()
        else:
            self.data_x, self.data_y, self.feature_length = self.prepare_data_per_request()

        self.model = lstm_model.LSTM(n_input=self.feature_length,
                                     n_hidden_units=self.n_hidden_units,
                                     n_layers=self.n_layers,
                                     train_mode=self.train_mode,
                                     n_classes=self.n_classes).to(self.device)
        if transfer_learning:
            self.model.load_state_dict(torch.load(self.savemodelpath))
        # self.model = self.model.double()  # TODO: check this double stuff
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        # self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=0.1, betas=(0.9, 0.999))
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', verbose=True)
        # optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        #  überprüfe was mse genau macht, abspeichern
        #  zb jede 10. epoche die distanz plotten
        #  quadrat mean squared error mal probieren
        self.distance = nn.NLLLoss()

        test_set_len = math.floor(self.data_x.size(0) / 10)
        train_set_len = self.data_x.size(0) - test_set_len

        self.train_indices = range(0, train_set_len)
        self.test_indices = range(train_set_len, train_set_len + test_set_len)

    def prepare_data_per_request(self):
        instance_information = pickle.load(open(self.instance_information_file, 'rb'))
        embeddings = pickle.load(open(self.loadvectors, 'rb'))
        feature_length = embeddings[0].size(0)

        ring = True
        data_x = []
        data_y = []
        target_indices = []
        for l in instance_information:
            begin, end = l[0], l[1]
            if ring:
                if end - begin > self.seq_length:
                    index_array = [i for i in range(begin, end + 1)]
                    roll_indices = [np.roll(index_array, -i)[0:self.seq_length + 1] for i in range(0, len(index_array))]
                    for indices in roll_indices:
                        data_x_temp = []
                        [data_x_temp.append(embeddings[i]) for i in indices[:-1]]
                        data_x.append(torch.stack(data_x_temp))
                        data_y.append(self.target_labels[indices[-1]])
                        target_indices.append(indices[-1])
            else:
                for i in range(0, end - begin - self.seq_length - + 1):
                    data_x.append(embeddings[begin + i:begin + i + self.seq_length])
                    data_y.append(self.target_labels[begin + i + self.seq_length + 1])
                    target_indices.append(begin + i + self.seq_length + 1)
        if self.anomalies_run:
            anomaly_indices_file = open(self.results_dir + 'anomaly_label_indices', 'w+')
            for val in target_indices:
                anomaly_indices_file.write(str(val) + "\n")
            anomaly_indices_file.close()

        data_x = torch.stack(data_x).to(self.device)
        data_y = torch.tensor(data_y).to(self.device)

        return data_x, data_y, feature_length

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

        return data_x, data_y, feature_length

    def split(self, x, n_splits):
        n_samples = len(x)
        k_fold_size = n_samples // n_splits
        indices = np.arange(n_samples)

        margin = 0
        indices_aggregated = []
        for i in range(n_splits):
            start = i * k_fold_size
            stop = start + k_fold_size
            mid = int(0.8 * (stop - start)) + start
            indices_aggregated.append(tuple((indices[start: mid], indices[mid + margin: stop])))
        return indices_aggregated

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
                #pred_label = prediction.cpu().data.max(1)[1].numpy()
                hidden = self.repackage_hidden(hidden)
                loss = self.distance(prediction, target)
                loss_distribution.append(loss.item())
                total_loss += loss.item()
        return total_loss / len(idx), loss_distribution

    def predict(self, idx):
        self.model.eval()
        # TODO: since we want to predict *every* loss of every line, we don't use batches, so here we use batch_size
        #   1 is this ok?
        hidden = self.model.init_hidden(1, self.device)
        predicted_labels = []
        with torch.no_grad():
            for data, target in zip(self.data_x[idx], self.data_y[idx]):
                data = data.view(1, self.seq_length, self.feature_length)
                prediction, hidden = self.model(data, hidden)
                pred_label = prediction.cpu().data.max(1)[1].numpy()
                predicted_labels.append(pred_label)
                hidden = self.repackage_hidden(hidden)
        return predicted_labels

    def train(self, idx):
        self.model.train()
        dataloader_x = DataLoader(self.data_x[idx], batch_size=self.batch_size, drop_last=True)
        dataloader_y = DataLoader(self.data_y[idx], batch_size=self.batch_size, drop_last=True)
        hidden = self.model.init_hidden(self.batch_size, self.device)  # TODO: check stuff with batch size
        for data, target in zip(dataloader_x, dataloader_y):
            self.optimizer.zero_grad()
            hidden = self.repackage_hidden(hidden)
            prediction, hidden = self.model(data, hidden)
            #pred_label = prediction.cpu().data.max(1)[1].numpy()
            loss = self.distance(prediction, target)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
            for p in self.model.parameters():
                p.data.add_(-self.learning_rate, p.grad.data)

    def start_training(self):
        best_val_loss = None
        log_output = open(self.results_dir + 'training_output.txt', 'w')
        loss_over_time = open(self.results_dir + 'loss_over_time.txt', 'w')
        try:
            loss_values = []
            train_and_eval_indices = self.split(self.train_indices, self.folds)
            for epoch in range(1, self.num_epochs + 1):
                loss_this_epoch = []
                val_loss = 0
                epoch_start_time = time.time()
                for i in range(0, self.folds):
                    train_incides = train_and_eval_indices[i][0]
                    eval_indices = train_and_eval_indices[i][1]

                    self.train(train_incides)
                    this_loss, _ = self.evaluate(eval_indices)
                    loss_this_epoch.append(this_loss)
                    self.optimizer.step()
                    self.scheduler.step(this_loss)
                    val_loss += this_loss
                output = '-' * 89 + "\n" + 'LSTM: | end of epoch {:3d} | time: {:5.2f}s | valid loss {} | ' 'valid ppl {} \n'\
                       .format(epoch, (time.time() - epoch_start_time), val_loss, math.exp(val_loss))\
                       + '-' * 89
                print(output)
                log_output.write(output + "\n")
                loss_over_time.write(str(val_loss) + "\n")
                if not best_val_loss or val_loss < best_val_loss:
                    torch.save(self.model.state_dict(), self.savemodelpath)
                    best_val_loss = val_loss
                loss_values.append(val_loss / self.folds)
        except KeyboardInterrupt:
            print('-' * 89)
            print('Exiting from training early')

    def calc_labels(self):
        self.model = lstm_model.LSTM(self.feature_length, self.n_hidden_units, self.n_layers, train_mode=False,
                                     n_classes=self.n_classes).to(self.device)
        self.model.load_state_dict(torch.load(self.savemodelpath))
        self.model.eval()
        n_samples = len(self.data_x)
        indices = np.arange(n_samples)
        predicted_labels = self.predict(indices)

        return predicted_labels


if __name__ == '__main__':
    ad = AnomalyDetection()
    ad.start_training()
