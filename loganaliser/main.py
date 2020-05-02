import math
import pickle
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import torch
from sklearn.metrics import ConfusionMatrixDisplay
from torch import optim
from torch.utils.data import DataLoader

import loganaliser.model as lstm_model
from loganaliser.vanilla_autoencoder import AutoEncoder


class AnomalyDetection:
    def __init__(self,
                 num_epochs,
                 n_layers,
                 n_hidden_units,
                 seq_length,
                 batch_size,
                 clip,
                 lines_that_have_anomalies,
                 train_vectors,
                 train_instance_information_file,
                 test_vectors,
                 test_instance_information_file,
                 n_features,
                 n_input,
                 results_dir,
                 embeddings_model,
                 learning_rate=1e-4,
                 savemodelpath='saved_models/lstm.pth',
                 folds=5,
                 train_mode=False,
                 transfer_learning=False,
                 loadautoencodermodel='saved_models/openstack_52k_normal_vae.pth'
                 ):
        self.train_vectors = pickle.load(open(train_vectors, 'rb'))
        self.train_instance_information_file = train_instance_information_file
        self.test_vectors = pickle.load(open(test_vectors, 'rb'))
        self.test_instance_information_file = test_instance_information_file
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
        self.results_dir = results_dir
        self.lines_that_have_anomalies = lines_that_have_anomalies
        self.feature_length = n_features
        self.log_frequency_interval = 10
        self.n_input = n_input

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # select word embeddings
        if self.train_instance_information_file is None:
            if embeddings_model == 'glove':
                self.train_data_x, self.train_data_y,  = self.prepare_data_latent_space(self.train_vectors)
                self.test_data_x, self.test_data_y = self.prepare_data_latent_space(self.test_vectors)
            elif embeddings_model == 'bert' or embeddings_model == 'embeddings_layer':
                self.train_data_x, self.train_data_y = self.prepare_data_raw(self.train_vectors)
                self.test_data_x, self.test_data_y = self.prepare_data_raw(self.test_vectors)
        else:
            self.train_data_x, self.train_data_y = self.prepare_data_per_request(self.train_vectors,
                                                                                 self.train_instance_information_file)
            self.test_data_x, self.test_data_y = self.prepare_data_per_request(self.test_vectors,
                                                                               self.test_instance_information_file)

        self.model = lstm_model.LSTM(n_input=self.n_input,
                                     n_hidden_units=self.n_hidden_units,
                                     n_layers=self.n_layers,
                                     train_mode=self.train_mode,
                                     n_output=self.feature_length).to(self.device)
        if transfer_learning:
            self.model.load_state_dict(torch.load(self.savemodelpath))
        # self.model = self.model.double()  # TODO: check this double stuff
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=0.1)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min')

        test_set_len = math.floor(self.train_data_x.size(0) / 10)
        train_set_len = self.train_data_x.size(0) - test_set_len

        self.train_indices = range(0, train_set_len)
        self.test_indices = range(train_set_len, train_set_len + test_set_len)


    def return_target(self, embeddings):
        """
        Append target val depending on which mode is selected: regression or classification
        :return:
        """
        raise NotImplementedError


    def prepare_data_per_request(self, embeddings, instance_information_file):
        instance_information = pickle.load(open(instance_information_file, 'rb'))

        ring = True
        data_x = []
        data_y = []
        self.target_indices = []
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
                        data_y.append(self.return_target(embeddings)[indices[-1]])
                        self.target_indices.append(indices[-1])
            else:
                for i in range(0, end - begin - self.seq_length - + 1):
                    data_x.append(embeddings[begin + i:begin + i + self.seq_length])
                    data_y.append(self.return_target(embeddings)[begin + i + self.seq_length + 1])
                    self.target_indices.append(begin + i + self.seq_length + 1)

        data_x = torch.stack(data_x).to(self.device)
        data_y = torch.stack(data_y).to(self.device)

        return data_x, data_y

    def prepare_data_raw(self, vectors):
        embeddings = pickle.load(open(vectors, 'rb'))
        number_of_sentences = len(embeddings)

        data_x = []
        data_y = []
        for i in range(0, number_of_sentences - self.seq_length - 1):
            data_x.append(embeddings[i: i + self.seq_length])
            data_y.append(embeddings[i + 1 + self.seq_length])

        data_x = torch.stack(data_x).to(self.device)
        data_y = torch.stack(data_y).to(self.device)

        return data_x, data_y

    def prepare_data_latent_space(self, padded_embeddings):

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

        data_x = torch.tensor(data_x).to(self.device)
        data_y = torch.tensor(data_y).to(self.device)

        # samples, timesteps, features
        # dataloader_x = DataLoader(data_x, batch_size=64)
        # dataloader_y = DataLoader(data_y, batch_size=64)
        # data_x = np.reshape(data_x, (n_patterns, self.seq_length, feature_length))

        return data_x, data_y

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

    def evaluate(self, idx, dist):
        self.model.eval()
        dataloader_x = DataLoader(self.train_data_x[idx], batch_size=self.batch_size, drop_last=True)
        dataloader_y = DataLoader(self.train_data_y[idx], batch_size=self.batch_size, drop_last=True)
        loss_distribution = []
        total_loss = 0
        hidden = self.model.init_hidden(self.batch_size, self.device)  # TODO: check stuff with batch size
        with torch.no_grad():
            for data, target in zip(dataloader_x, dataloader_y):
                prediction, hidden = self.model(data, hidden)
                hidden = self.repackage_hidden(hidden)
                loss = dist(prediction, target)
                loss_distribution.append(loss.item())
                total_loss += loss.item()
        return total_loss / len(idx), loss_distribution


    def train(self, idx, dist):
        self.model.train()
        dataloader_x = DataLoader(self.train_data_x[idx], batch_size=self.batch_size, drop_last=True)
        dataloader_y = DataLoader(self.train_data_y[idx], batch_size=self.batch_size, drop_last=True)
        hidden = self.model.init_hidden(self.batch_size, self.device)  # TODO: check stuff with batch size
        total_loss = 0
        for data, target in zip(dataloader_x, dataloader_y):
            self.optimizer.zero_grad()
            hidden = self.repackage_hidden(hidden)
            prediction, hidden = self.model(data, hidden)
            loss = dist(prediction, target)
            total_loss += loss.item()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
            for p in self.model.parameters():
                p.data.add_(-self.learning_rate, p.grad.data)
        return total_loss / len(idx)

    def start_training(self, no_anomaly):
        raise NotImplementedError


    def write_intermediate_metrics(self, results, eval_loss):
        # plot metrics every 5 epochs
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=np.arange(self.log_frequency_interval,self.num_epochs+1,self.log_frequency_interval),
                                 y=[x.f1 for x in results], mode='lines+markers', name='F1'))
        fig.add_trace(go.Scatter(x=np.arange(self.log_frequency_interval,self.num_epochs+1,self.log_frequency_interval),
                                 y=[x.precision for x in results], mode='lines+markers', name='Precision'))
        fig.add_trace(go.Scatter(x=np.arange(self.log_frequency_interval,self.num_epochs+1,self.log_frequency_interval),
                                 y=[x.recall for x in results], mode='lines+markers', name='Recall'))
        fig.add_trace(go.Scatter(x=np.arange(self.log_frequency_interval,self.num_epochs+1,self.log_frequency_interval),
                                 y=[x.accuracy for x in results], mode='lines+markers', name='Accuracy'))
        fig.write_html(self.results_dir + 'metrics.html')

        # write metrics to file
        with open(self.results_dir + "metrics.csv", 'w') as metrics_file:
            metrics_file.write("Epoch, F1, Precision, Recall, Accuracy\n")
            for i, x in enumerate(results):
                metrics_file.write("{}, {}, {}, {}, {}\n".format((i+1)*self.log_frequency_interval, x.f1, x.precision, x.recall, x.accuracy))

        # plot training loss
        loss_fig = go.Figure()
        loss_fig.add_trace(go.Scatter(x=np.arange(0, self.num_epochs, 1), y=eval_loss, mode='lines+markers', name='Loss'))
        loss_fig.write_html(self.results_dir + 'loss.html')


    def write_final_metrics(self, res):
        scores_file = open(self.results_dir + "scores.txt", "w+")
        scores_file.write("F1-Score: {}\n".format(str(res.f1)))
        scores_file.write("Precision-Score: {}\n".format(str(res.precision)))
        scores_file.write("Recall-Score: {}\n".format(str(res.recall)))
        scores_file.write("Accuracy-Score: {}\n".format(str(res.accuracy)))
        scores_file.write("confusion matrix:\n")
        scores_file.write('\n'.join('\t'.join('%0.3f' % x for x in y) for y in res.confusion_matrix))
        disp = ConfusionMatrixDisplay(confusion_matrix=res.confusion_matrix, display_labels=[0, 1])

        disp = disp.plot(include_values=True, cmap='inferno')
        plt.savefig(self.results_dir + 'confusion_matrix.png')
        plt.clf()
        scores_file.close()


    def final_prediction(self, no_anomaly) -> Tuple[float, float]:
        raise NotImplementedError
