import math
import pickle
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from shared_functions import write_lines_to_file, calculate_anomaly_loss
from torch import optim
import plotly.graph_objects as go

import loganaliser.model as lstm_model
from loganaliser.vanilla_autoencoder import AutoEncoder
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

log_frequency_interval = 10

class AnomalyDetection:
    def __init__(self,
                 num_epochs,
                 n_layers,
                 n_hidden_units,
                 seq_length,
                 batch_size,
                 clip,
                 lines_that_have_anomalies,
                 train_vectors='../data/openstack/utah/padded_embeddings_pickle/openstack_52k_normal.pickle',
                 train_instance_information_file=None,
                 test_vectors=None,
                 test_instance_information_file=None,
                 learning_rate=1e-4,
                 results_dir=None,
                 embeddings_model='glove',
                 loadautoencodermodel='saved_models/openstack_52k_normal_vae.pth',
                 savemodelpath='saved_models/lstm.pth',
                 folds=5,
                 train_mode=False,
                 anomalies_run=False,
                 transfer_learning=False
                 ):
        self.train_vectors = train_vectors
        self.train_instance_information_file = train_instance_information_file
        self.test_vectors = test_vectors
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
        self.anomalies_run = anomalies_run
        self.results_dir = results_dir
        self.lines_that_have_anomalies = lines_that_have_anomalies

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # select word embeddings
        if self.train_instance_information_file is None:
            if embeddings_model == 'glove':
                self.train_data_x, self.train_data_y, self.feature_length = self.prepare_data_latent_space(self.train_vectors)
                self.test_data_x, self.test_data_y, self.feature_length = self.prepare_data_latent_space(self.test_vectors)
            elif embeddings_model == 'bert' or embeddings_model == 'embeddings_layer':
                self.train_data_x, self.train_data_y, self.feature_length = self.prepare_data_raw(self.train_vectors)
                self.test_data_x, self.test_data_y, self.feature_length = self.prepare_data_raw(self.test_vectors)
        else:
            self.train_data_x, self.train_data_y, self.feature_length = self.prepare_data_per_request(self.train_vectors,
                                                                                                      self.train_instance_information_file)
            self.test_data_x, self.test_data_y, self.feature_length = self.prepare_data_per_request(self.test_vectors,
                                                                                                    self.test_instance_information_file)

        self.model = lstm_model.LSTM(n_input=self.feature_length,
                                     n_hidden_units=self.n_hidden_units,
                                     n_layers=self.n_layers,
                                     train_mode=self.train_mode).to(self.device)
        if transfer_learning:
            self.model.load_state_dict(torch.load(self.savemodelpath))
        # self.model = self.model.double()  # TODO: check this double stuff
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        # self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=0.1, betas=(0.9, 0.999))
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min')
        # optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        #  überprüfe was mse genau macht, abspeichern
        #  zb jede 10. epoche die distanz plotten
        #  quadrat mean squared error mal probieren
        self.distance = nn.MSELoss()

        test_set_len = math.floor(self.train_data_x.size(0) / 10)
        train_set_len = self.train_data_x.size(0) - test_set_len

        self.train_indices = range(0, train_set_len)
        self.test_indices = range(train_set_len, train_set_len + test_set_len)

    def prepare_data_per_request(self, vectors, instance_information_file):
        instance_information = pickle.load(open(instance_information_file, 'rb'))
        embeddings = pickle.load(open(vectors, 'rb'))
        feature_length = embeddings[0].size(0)

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
                        data_y.append(embeddings[indices[-1]])
                        self.target_indices.append(indices[-1])
            else:
                for i in range(0, end - begin - self.seq_length - + 1):
                    data_x.append(embeddings[begin + i:begin + i + self.seq_length])
                    data_y.append(embeddings[begin + i + self.seq_length + 1])
                    self.target_indices.append(begin + i + self.seq_length + 1)
        if self.anomalies_run:
            anomaly_indices_file = open(self.results_dir + "anomaly_loss_indices", 'w+')
            for val in self.target_indices:
                anomaly_indices_file.write(str(val) + "\n")
            anomaly_indices_file.close()

        data_x = torch.stack(data_x).to(self.device)
        data_y = torch.stack(data_y).to(self.device)

        return data_x, data_y, feature_length

    def prepare_data_raw(self, vectors):
        embeddings = pickle.load(open(vectors, 'rb'))
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

    def prepare_data_latent_space(self, vectors):
        # load vectors and glove obj
        padded_embeddings = pickle.load(open(vectors, 'rb'))

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
        dataloader_x = DataLoader(self.train_data_x[idx], batch_size=self.batch_size, drop_last=True)
        dataloader_y = DataLoader(self.train_data_y[idx], batch_size=self.batch_size, drop_last=True)
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

    def predict(self, data_x, data_y):
        self.model.eval()
        # TODO: since we want to predict *every* loss of every line, we don't use batches, so here we use batch_size
        #   1 is this ok?
        hidden = self.model.init_hidden(1, self.device)
        loss_distribution = []
        with torch.no_grad():
            for data, target in zip(data_x, data_y):
                data = data.view(1, self.seq_length, self.feature_length)
                prediction, hidden = self.model(data, hidden)
                hidden = self.repackage_hidden(hidden)
                loss = self.distance(prediction.reshape(-1), target.reshape(-1))  # TODO check if reshape is necessary
                loss_distribution.append(loss.item())
        return loss_distribution

    def train(self, idx):
        self.model.train()
        dataloader_x = DataLoader(self.train_data_x[idx], batch_size=self.batch_size, drop_last=True)
        dataloader_y = DataLoader(self.train_data_y[idx], batch_size=self.batch_size, drop_last=True)
        hidden = self.model.init_hidden(self.batch_size, self.device)  # TODO: check stuff with batch size
        total_loss = 0
        for data, target in zip(dataloader_x, dataloader_y):
            self.optimizer.zero_grad()
            hidden = self.repackage_hidden(hidden)
            prediction, hidden = self.model(data, hidden)
            loss = self.distance(prediction.reshape(-1), target.reshape(-1))
            total_loss += loss.item()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
            for p in self.model.parameters():
                p.data.add_(-self.learning_rate, p.grad.data)
        return total_loss / len(idx)

    def start_training(self, no_anomaly):
        best_val_loss = None
        log_output = open(self.results_dir + 'training_output.txt', 'w')
        try:
            loss_values = []
            intermediate_results = []
            train_and_eval_indices = self.split(self.train_indices, self.folds)
            for epoch in range(1, self.num_epochs + 1):
                eval_loss = 0
                train_loss = 0
                loss_this_epoch = []
                epoch_start_time = time.time()
                for i in range(0, self.folds):
                    train_incides = train_and_eval_indices[i][0]
                    eval_indices = train_and_eval_indices[i][1]

                    this_train_loss = self.train(train_incides)
                    this_eval_loss, _ = self.evaluate(eval_indices)
                    loss_this_epoch.append(this_eval_loss)
                    self.optimizer.step()
                    self.scheduler.step(this_eval_loss)
                    eval_loss += this_eval_loss
                    train_loss += this_train_loss
                if epoch % log_frequency_interval == 0:
                    normal_loss_values = self.predict(self.train_data_x, self.train_data_y)
                    anomaly_loss_values = self.predict(self.test_data_x, self.test_data_y)
                    result = calculate_anomaly_loss(anomaly_loss_values, normal_loss_values, self.target_indices,
                                                    self.lines_that_have_anomalies, no_anomaly)
                    intermediate_results.append(result)
                output = '-' * 89 + "\n" + 'LSTM: | end of epoch {:3d} | time: {:5.2f}s | loss {} |\n' \
                       .format(epoch, (time.time() - epoch_start_time), eval_loss / self.folds) \
                       + '-' * 89
                print(output)
                log_output.write(output + "\n")
                if not best_val_loss or eval_loss < best_val_loss:
                    torch.save(self.model.state_dict(), self.savemodelpath)
                    best_val_loss = eval_loss
                loss_values.append(eval_loss / self.folds)
            # training done, do final prediction
            log_output.close()
            normal_loss_values = self.predict(self.train_data_x, self.train_data_y)
            anomaly_loss_values = self.predict(self.test_data_x, self.test_data_y)
            calculate_anomaly_loss(anomaly_loss_values, normal_loss_values, self.target_indices,
                                   self.lines_that_have_anomalies, no_anomaly)
            self.write_train_results(intermediate_results, loss_values)

        except KeyboardInterrupt:
            print('-' * 89)
            print('Exiting from training early')


    def write_train_results(self, results, eval_loss):
        # plot metrics every 5 epochs
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=np.arange(log_frequency_interval,self.num_epochs+1,log_frequency_interval),
                                 y=[x.f1 for x in results], mode='lines+markers', name='F1'))
        fig.add_trace(go.Scatter(x=np.arange(log_frequency_interval,self.num_epochs+1,log_frequency_interval),
                                 y=[x.precision for x in results], mode='lines+markers', name='Precision'))
        fig.add_trace(go.Scatter(x=np.arange(log_frequency_interval,self.num_epochs+1,log_frequency_interval),
                                 y=[x.recall for x in results], mode='lines+markers', name='Recall'))
        fig.add_trace(go.Scatter(x=np.arange(log_frequency_interval,self.num_epochs+1,log_frequency_interval),
                                 y=[x.accuracy for x in results], mode='lines+markers', name='Accuracy'))
        fig.write_html(self.results_dir + 'metrics.html')

        # write metrics to file
        with open(self.results_dir + "metrics.csv", 'w') as metrics_file:
            metrics_file.write("Epoch, F1, Precision, Recall, Accuracy\n")
            for i, x in enumerate(results):
                metrics_file.write("{}, {}, {}, {}, {}\n".format((i+1)*log_frequency_interval, x.f1, x.precision, x.recall, x.accuracy))

        # plot training loss
        loss_fig = go.Figure()
        loss_fig.add_trace(go.Scatter(x=np.arange(0, self.num_epochs, 1), y=eval_loss, mode='lines+markers', name='Loss'))
        loss_fig.write_html(self.results_dir + 'loss.html')


    def write_final_results(self, res):
        write_lines_to_file(self.results_dir + "pred_outliers_indeces.txt", res.predicted_outliers, new_line=True)
        write_lines_to_file(self.results_dir + "pred_outliers_values.txt", res.pred_outliers_values, new_line=True)
        write_lines_to_file(self.results_dir + 'anomaly_loss_values', res.anomaly_loss_values_correct_order, new_line=True)
        write_lines_to_file(self.results_dir + 'normal_loss_values', res.train_loss_values, new_line=True)

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

    def loss_evaluation(self, no_anomaly):
        loss_values_train = self.predict(self.train_data_x, self.train_data_y)
        loss_values_test = self.predict(self.test_data_x, self.test_data_y)
        res = calculate_anomaly_loss(loss_values_test, loss_values_train, self.target_indices, self.lines_that_have_anomalies,
                                     no_anomaly)
        res.train_loss_values = loss_values_train
        self.write_final_results(res)
        return res.f1, res.precision


if __name__ == '__main__':
    ad = AnomalyDetection()
    ad.start_training()
