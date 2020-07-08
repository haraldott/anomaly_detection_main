import time

import torch
import torch.nn as nn

from loganaliser.main import AnomalyDetection
from shared_functions import calculate_anomaly_loss, write_lines_to_file
from tools import distribution_plots as distribution_plots


class Regression(AnomalyDetection):
    def __init__(self, *args, **kwargs):
        self.distance = nn.MSELoss()

        super(Regression, self).__init__(*args, **kwargs)

    def predict(self, data_x, data_y, dist):
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
                loss = dist(prediction.reshape(-1), target.reshape(-1))  # TODO check if reshape is necessary
                loss_distribution.append(loss.item())
        return loss_distribution

    def return_target(self, embeddings):
        return embeddings

    def start_training(self):
        best_val_loss = None
        log_output = open(self.results_dir + 'training_output.txt', 'w')
        loss_over_time = open(self.results_dir + 'loss_over_time.txt', 'w')
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

                    this_train_loss = self.train(train_incides, self.distance)
                    this_eval_loss, _ = self.evaluate(eval_indices, self.distance)
                    loss_this_epoch.append(this_eval_loss)
                    self.optimizer.step()
                    self.scheduler.step(this_eval_loss)
                    eval_loss += this_eval_loss
                    train_loss += this_train_loss
                if epoch % self.log_frequency_interval == 0:
                    normal_loss_values = self.predict(self.train_data_x, self.train_data_y, self.distance)
                    anomaly_loss_values = self.predict(self.test_data_x, self.test_data_y, self.distance)
                    result = calculate_anomaly_loss(anomaly_loss_values, normal_loss_values, self.target_indices,
                                                    self.lines_that_have_anomalies, self.no_anomaly, self.results_dir)
                    intermediate_results.append(result)
                output = '-' * 89 + "\n" + 'LSTM: | end of epoch {:3d} | time: {:5.2f}s | loss {} |\n' \
                    .format(epoch, (time.time() - epoch_start_time), eval_loss / self.folds) \
                         + '-' * 89
                print(output)
                log_output.write(output + "\n")
                loss_over_time.write(str(eval_loss) + "\n")
                if not best_val_loss or eval_loss < best_val_loss:
                    torch.save(self.model.state_dict(), self.savemodelpath)
                    best_val_loss = eval_loss
                loss_values.append(eval_loss / self.folds)
            # training done, do final prediction
            log_output.close()
            if self.test_vectors is not None and self.log_frequency_interval < self.num_epochs:
                self.write_intermediate_metrics(self.log_frequency_interval, self.num_epochs, self.results_dir,
                                            intermediate_results, loss_values)

        except KeyboardInterrupt:
            print('-' * 89)
            print('Exiting from training early')

    def write_regression_metrics(self, res):
        write_lines_to_file(self.results_dir + "pred_outliers_indeces.txt", res.predicted_outliers, new_line=True)
        write_lines_to_file(self.results_dir + "pred_outliers_loss_values.txt", res.pred_outliers_loss_values, new_line=True)
        write_lines_to_file(self.results_dir + 'anomaly_loss_values', res.anomaly_loss_values, new_line=True)
        write_lines_to_file(self.results_dir + 'normal_loss_values', res.train_loss_values, new_line=True)

    def final_prediction(self):
        loss_values_train = self.predict(self.train_data_x, self.train_data_y, self.distance)
        loss_values_test = self.predict(self.test_data_x, self.test_data_y, self.distance)
        res = calculate_anomaly_loss(loss_values_test, loss_values_train, self.target_indices,
                                     self.lines_that_have_anomalies,
                                     self.no_anomaly, self.results_dir)
        distribution_plots(self.results_dir, loss_values_train, loss_values_test, self.num_epochs, self.seq_length,
                           768, 0)
        res.train_loss_values = loss_values_train
        self.write_regression_metrics(res)
        self.write_final_metrics(self.results_dir, res)
        return res.f1, res.precision
