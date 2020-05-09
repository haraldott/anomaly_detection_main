import math
import pickle
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim

import loganaliser.model as lstm_model
from torch.utils.tensorboard import SummaryWriter

from loganaliser.main import AnomalyDetection
from loganaliser.vanilla_autoencoder import AutoEncoder
from shared_functions import determine_binary_anomalies, write_lines_to_file


class BinaryClassification:
    def __init__(self,
                 num_epochs,
                 n_layers,
                 n_hidden_units,
                 seq_length,
                 batch_size,
                 clip,
                 train_anomaly_lines,
                 test_anomaly_lines,
                 train_vectors,
                 train_instance_information_file,
                 test_vectors,
                 test_instance_information_file,
                 n_input,
                 results_dir,
                 embeddings_model,
                 prediction_only,
                 no_anomaly=False,
                 learning_rate=1e-4,
                 loadautoencodermodel='saved_models/openstack_52k_normal_vae.pth',
                 savemodelpath='saved_models/lstm.pth',
                 folds=5,
                 anomalies_run=False,
                 transfer_learning=False,
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
        self.anomalies_run = anomalies_run
        self.results_dir = results_dir
        self.train_anomaly_lines = train_anomaly_lines
        self.test_anomaly_lines = test_anomaly_lines
        self.log_frequency_interval = 10
        self.no_anomaly = no_anomaly

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_input = n_input

        # select word embeddings
        if self.train_instance_information_file is None:
            if embeddings_model == 'glove':
                self.train_data_x, self.train_data_y, = self.prepare_data_latent_space(self.train_vectors)
                self.test_data_x, self.test_data_y = self.prepare_data_latent_space(self.test_vectors)
            elif embeddings_model == 'bert' or embeddings_model == 'embeddings_layer':
                self.train_data_x, self.train_data_y = self.prepare_data_raw(self.train_vectors)
                self.test_data_x, self.test_data_y = self.prepare_data_raw(self.test_vectors)
        else:
            self.train_data_x, self.train_data_y = self.prepare_data_per_request(self.train_vectors,
                                                                                 self.train_instance_information_file,
                                                                                 self.train_anomaly_lines)
            self.test_data_x, self.test_data_y = self.prepare_data_per_request(self.test_vectors,
                                                                               self.test_instance_information_file,
                                                                               self.test_anomaly_lines)

        self.model = lstm_model.Net(n_input=self.n_input,
                                    seq_len=self.seq_length,
                                    n_hidden_units=self.n_hidden_units,
                                    n_layers=self.n_layers).to(self.device)
        if transfer_learning or prediction_only:
            self.model.load_state_dict(torch.load(self.savemodelpath))
        # self.model = self.model.double()  # TODO: check this double stuff
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        # self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=0.1, betas=(0.9, 0.999))
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', verbose=True)
        # optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        #  überprüfe was mse genau macht, abspeichern
        #  zb jede 10. epoche die distanz plotten
        #  quadrat mean squared error mal probieren
        self.distance = nn.BCELoss()

        test_set_len = math.floor(self.train_data_x.size(0) / 10)
        train_set_len = self.train_data_x.size(0) - test_set_len

        self.train_indices = range(0, train_set_len)
        self.test_indices = range(train_set_len, train_set_len + test_set_len)

    def prepare_data_per_request(self, embeddings, instance_information_file, anomaly_lines):
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
                        data_x.append(torch.cat(data_x_temp))
                        indices = [x for x in indices[:-1]]
                        if set(indices) & set(anomaly_lines):
                            data_y.append(float(1))
                        else:
                            data_y.append(float(0))
                        self.target_indices.append(indices[-2])
            else:
                for i in range(0, end - begin - self.seq_length - 1):
                    data_x_temp = embeddings[begin + i:begin + i + self.seq_length]
                    data_x.append(data_x_temp)
                    data_y_temp = []
                    [data_y_temp.append(float(1)) if x in anomaly_lines else data_y_temp.append(float(0)) for x in (begin + i, begin + i + self.seq_length)]
        data_x = torch.stack(data_x).to(self.device)
        data_y = torch.tensor(data_y).to(self.device)

        return data_x, data_y

    def prepare_data_raw(self, embeddings):
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

    def evaluate(self, idx):
        self.model.eval()
        dataloader_x = DataLoader(self.train_data_x[idx], batch_size=self.batch_size, drop_last=True)
        dataloader_y = DataLoader(self.train_data_y[idx], batch_size=self.batch_size, drop_last=True)
        loss_distribution = []
        total_loss = 0
        with torch.no_grad():
            for data, target in zip(dataloader_x, dataloader_y):
                prediction = self.model(data)
                #pred_label = prediction.cpu().data.max(1)[1].numpy()
                loss = self.distance(prediction, target)
                loss_distribution.append(loss.item())
                total_loss += loss.item()
        return total_loss / len(idx), loss_distribution

    def predict(self):
        self.model.eval()
        # TODO: since we want to predict *every* loss of every line, we don't use batches, so here we use batch_size
        #   1 is this ok?
        predicted_labels = []
        with torch.no_grad():
            for data, target in zip(self.test_data_x, self.test_data_y):
                prediction = self.model(data)
                pred_label = prediction.data.cpu().numpy()[0]
                predicted_labels.append(pred_label)
        return predicted_labels

    def train(self, idx):
        self.model.train()
        dataloader_x = DataLoader(self.train_data_x[idx], batch_size=self.batch_size, drop_last=True)
        dataloader_y = DataLoader(self.train_data_y[idx], batch_size=self.batch_size, drop_last=True)
        total_loss = 0
        for data, target in zip(dataloader_x, dataloader_y):
            self.optimizer.zero_grad()
            prediction = self.model(data)
            #pred_label = prediction.cpu().data.max(1)[1].numpy()
            loss = self.distance(torch.squeeze(prediction), target)
            total_loss += loss.item()
            loss.backward()

            #torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
            for p in self.model.parameters():
                p.data.add_(-self.learning_rate, p.grad.data)
        return total_loss / len(idx)

    def start_training(self):
        best_val_loss = None
        log_output = open(self.results_dir + 'training_output.txt', 'w')
        loss_over_time = open(self.results_dir + 'loss_over_time.txt', 'w')
        writer = SummaryWriter(log_dir="my_experiment")
        try:
            loss_values = []
            intermediate_results = []
            train_and_eval_indices = AnomalyDetection.split(self.train_indices, self.folds)
            for epoch in range(1, self.num_epochs + 1):
                loss_this_epoch = []
                eval_loss = 0
                train_loss = 0
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
                    if epoch % self.log_frequency_interval == 0:
                        predicted_labels = self.predict()
                        result = determine_binary_anomalies(predicted_labels, self.target_indices,
                                                            self.test_anomaly_lines, no_anomaly=self.no_anomaly)
                        intermediate_results.append(result)
                output = '-' * 89 + "\n" + 'LSTM: | end of epoch {:3d} | time: {:5.2f}s | loss {} |\n'\
                       .format(epoch, (time.time() - epoch_start_time), eval_loss / self.folds)\
                       + '-' * 89
                writer.add_scalar("Loss/train", train_loss / self.folds)
                writer.add_scalar("Loss/test", eval_loss / self.folds)
                print(output)
                log_output.write(output + "\n")
                loss_over_time.write(str(eval_loss) + "\n")
                if not best_val_loss or eval_loss < best_val_loss:
                    torch.save(self.model.state_dict(), self.savemodelpath)
                    best_val_loss = eval_loss
                loss_values.append(eval_loss / self.folds)
            log_output.close()
            AnomalyDetection.write_intermediate_metrics(self.log_frequency_interval, self.num_epochs, self.results_dir,
                                            intermediate_results, loss_values)
        except KeyboardInterrupt:
            print('-' * 89)
            print('Exiting from training early')


    def write_classification_metrics(self, res):
        write_lines_to_file(self.results_dir + 'anomaly_labels.txt', res.predicted_labels, new_line=True)
        write_lines_to_file(self.results_dir + "pred_outliers_indeces.txt", res.predicted_outliers, new_line=True)


    def final_prediction(self):
        predicted_labels = self.predict()
        result = determine_binary_anomalies(predicted_labels, self.target_indices, self.test_anomaly_lines,
                                            self.no_anomaly)
        self.write_classification_metrics(result)
        AnomalyDetection.write_final_metrics(self.results_dir, result)
        return result.f1, result.precision


