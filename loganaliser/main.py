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
from shared_functions import DetermineAnomalies, write_lines_to_file
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


class AnomalyDetection:
    def __init__(self,
                 n_classes,
                 target_labels,
                 num_epochs,
                 n_layers,
                 n_hidden_units,
                 seq_length,
                 batch_size,
                 clip,
                 top_k_label_mapping,
                 lines_that_have_anomalies,
                 corpus_of_log_containing_anomalies,
                 learning_rate=1e-4,
                 results_dir=None,
                 train_vectors='../data/openstack/utah/padded_embeddings_pickle/openstack_52k_normal.pickle',
                 test_vectors=None,
                 train_instance_information_file=None,
                 test_instance_information_file=None,
                 savemodelpath='saved_models/lstm.pth',
                 folds=5,
                 train_mode=False,
                 anomalies_run=False,
                 transfer_learning=False
                 ):
        self.train_vectors = train_vectors
        self.test_vectors = test_vectors
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
        self.train_instance_information_file = train_instance_information_file
        self.test_instance_information_file = test_instance_information_file
        self.anomalies_run = anomalies_run
        self.results_dir = results_dir
        self.top_k_label_mapping = top_k_label_mapping
        self.lines_that_have_anomalies = lines_that_have_anomalies
        self.corpus_of_log_containing_anomalies = corpus_of_log_containing_anomalies

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.target_labels = target_labels
        self.n_classes = n_classes

        # select word embeddings
        if self.train_instance_information_file is None:
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
                                     train_mode=self.train_mode,
                                     n_classes=self.n_classes).to(self.device)
        if transfer_learning:
            self.model.load_state_dict(torch.load(self.savemodelpath))
        # self.model = self.model.double()  # TODO: check this double stuff
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        # self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=0.1, betas=(0.9, 0.999))
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', verbose=True)
        #  überprüfe was mse genau macht, abspeichern
        #  zb jede 10. epoche die distanz plotten
        #  quadrat mean squared error mal probieren
        self.distance = nn.NLLLoss()

        eval_set_len = math.floor(self.train_data_x.size(0) / 10)
        train_set_len = self.train_data_x.size(0) - eval_set_len

        self.train_indices = range(0, train_set_len)
        self.test_indices = range(train_set_len, train_set_len + eval_set_len)
        self.determine_anomalies = DetermineAnomalies(lines_that_have_anomalies=self.lines_that_have_anomalies,
                                                      corpus_of_log_containing_anomalies=self.corpus_of_log_containing_anomalies,
                                                      top_k_anomaly_embedding_label_mapping=self.top_k_label_mapping,
                                                      order_of_values_of_file_containing_anomalies=self.target_indices)

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
                        data_y.append(self.target_labels[indices[-1]])
                        self.target_indices.append(indices[-1])
            else:
                for i in range(0, end - begin - self.seq_length - + 1):
                    data_x.append(embeddings[begin + i:begin + i + self.seq_length])
                    data_y.append(self.target_labels[begin + i + self.seq_length + 1])
                    self.target_indices.append(begin + i + self.seq_length + 1)
        if self.anomalies_run:
            anomaly_indices_file = open(self.results_dir + 'anomaly_label_indices', 'w+')
            for val in self.target_indices:
                anomaly_indices_file.write(str(val) + "\n")
            anomaly_indices_file.close()

        data_x = torch.stack(data_x).to(self.device)
        data_y = torch.tensor(data_y).to(self.device)

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
                #pred_label = prediction.cpu().data.max(1)[1].numpy()
                hidden = self.repackage_hidden(hidden)
                loss = self.distance(prediction, target)
                loss_distribution.append(loss.item())
                total_loss += loss.item()
        return total_loss / len(idx), loss_distribution

    def predict(self):
        self.model.eval()
        # TODO: since we want to predict *every* loss of every line, we don't use batches, so here we use batch_size
        #   1 is this ok?
        hidden = self.model.init_hidden(1, self.device)
        predicted_labels = []
        with torch.no_grad():
            for data, target in zip(self.test_data_x, self.test_data_y):
                data = data.view(1, self.seq_length, self.feature_length)
                prediction, hidden = self.model(data, hidden)
                pred_label = prediction.cpu().data.max(1)[1].numpy()[0]
                predicted_labels.append(pred_label)
                hidden = self.repackage_hidden(hidden)
        return predicted_labels

    def train(self, idx):
        self.model.train()
        dataloader_x = DataLoader(self.train_data_x[idx], batch_size=self.batch_size, drop_last=True)
        dataloader_y = DataLoader(self.train_data_y[idx], batch_size=self.batch_size, drop_last=True)
        total_loss = 0
        hidden = self.model.init_hidden(self.batch_size, self.device)  # TODO: check stuff with batch size
        for data, target in zip(dataloader_x, dataloader_y):
            self.optimizer.zero_grad()
            hidden = self.repackage_hidden(hidden)
            prediction, hidden = self.model(data, hidden)
            #pred_label = prediction.cpu().data.max(1)[1].numpy()
            loss = self.distance(prediction, target)
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
            train_and_eval_indices = self.split(self.train_indices, self.folds)
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
                if epoch % 5 == 0:
                    predicted_labels = self.predict()
                    result = self.determine_anomalies.determine(predicted_labels)
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
            # training done, write results
            self.write_all_results(intermediate_results, loss_values)
        except KeyboardInterrupt:
            print('-' * 89)
            print('Exiting from training early')


    def calc_labels(self):
        predicted_labels = self.predict()
        result = self.determine_anomalies.determine(predicted_labels)
        self.write_final_results(result)
        return result.f1, result.precision


    def write_all_results(self, results, eval_loss=None):
        if isinstance(results, list):
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=np.arange(5,self.num_epochs+1,5), y=[x.f1 for x in results], mode='lines+markers', name='F1'))
            fig.add_trace(go.Scatter(x=np.arange(5,self.num_epochs+1,5), y=[x.precision for x in results], mode='lines+markers', name='Precision'))
            fig.add_trace(go.Scatter(x=np.arange(5,self.num_epochs+1,5), y=[x.recall for x in results], mode='lines+markers', name='Recall'))
            fig.add_trace(go.Scatter(x=np.arange(5,self.num_epochs+1,5), y=[x.accuracy for x in results], mode='lines+markers', name='Accuracy'))
            fig.write_html(self.results_dir + 'metrics.html')

        if eval_loss:
            loss_fig = go.Figure()
            loss_fig.add_trace(go.Scatter(x=np.arange(0, self.num_epochs, 1), y=eval_loss, mode='lines+markers', name='Loss'))
            loss_fig.write_html(self.results_dir + 'loss.html')

        write_lines_to_file(self.results_dir + 'anomaly_labels', results[-1].predicted_labels_of_file_containing_anomalies_correct_order, new_line=True)
        write_lines_to_file(self.results_dir + "pred_outliers_indeces.txt", results[-1].predicted_outliers, new_line=True)

        self.write_final_results(results[-1])


    def write_final_results(self, res):
        scores_file = open(self.results_dir + "scores.txt", "w+")
        scores_file.write("F1-Score: {}\n".format(str(res.f1)))
        scores_file.write("Precision-Score: {}\n".format(str(res.precision)))
        scores_file.write("Recall-Score: {}\n".format(str(res.recall)))
        scores_file.write("Accuracy-Score: {}\n".format(str(res.accuracy)))
        scores_file.write("confusion matrix:\n")
        scores_file.write('\n'.join('\t'.join('%0.3f' % x for x in y) for y in res.confusion_matrix))
        disp = ConfusionMatrixDisplay(confusion_matrix=res.confusion_matrix,
                                      display_labels=[0, 1])

        # NOTE: Fill all variables here with default values of the plot_confusion_matrix
        disp = disp.plot(include_values=True)
        plt.savefig(self.results_dir + 'confusion_matrix.png')
        plt.clf()
        scores_file.close()

if __name__ == '__main__':
    ad = AnomalyDetection()
    ad.start_training()
