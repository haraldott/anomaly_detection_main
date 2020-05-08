import time

import torch
import torch.nn as nn

from loganaliser.main import AnomalyDetection
from shared_functions import DetermineAnomalies, write_lines_to_file


class Multiclass(AnomalyDetection):
    def __init__(self, target_labels, top_k_label_mapping, corpus_of_log_containing_anomalies, *args, **kwargs):
        self.distance = nn.CrossEntropyLoss()
        self.target_labels = target_labels
        super(Multiclass, self).__init__(*args, **kwargs)

        self.determine_anomalies = DetermineAnomalies(lines_that_have_anomalies=self.lines_that_have_anomalies,
                                                      target_labels=target_labels,
                                                      top_k_anomaly_embedding_label_mapping=top_k_label_mapping,
                                                      order_of_values_of_file_containing_anomalies=self.target_indices,
                                                      results_dir=self.results_dir)




    def predict(self):
        self.model.eval()
        # TODO: since we want to predict *every* loss of every line, we don't use batches, so here we use batch_size
        #   1 is this ok?
        hidden = self.model.init_hidden(1, self.device)
        predicted_labels = []
        with torch.no_grad():
            for data, target in zip(self.test_data_x, self.test_data_y):
                data = data.view(1, self.seq_length, self.n_input)
                prediction, hidden = self.model(data, hidden)
                pred_labels = (-prediction.data.cpu()).numpy().argsort()[0][:3]
                predicted_labels.append(pred_labels)
                hidden = self.repackage_hidden(hidden)
        return predicted_labels

    def return_target(self, embeddings):
        return torch.tensor(self.target_labels)

    def start_training(self):
        best_val_loss = None
        log_output = open(self.results_dir + 'training_output.txt', 'w')
        loss_over_time = open(self.results_dir + 'loss_over_time.txt', 'w')
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

                    this_train_loss = self.train(train_incides, self.distance)
                    this_eval_loss, _ = self.evaluate(eval_indices, self.distance)
                    loss_this_epoch.append(this_eval_loss)
                    self.optimizer.step()
                    self.scheduler.step(this_eval_loss)
                    eval_loss += this_eval_loss
                    train_loss += this_train_loss
                if epoch % self.log_frequency_interval == 0:
                    predicted_labels = self.predict()
                    result = self.determine_anomalies.determine(predicted_labels, self.no_anomaly)
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
            # training done, write results
            log_output.close()
            self.write_intermediate_metrics(self.log_frequency_interval, self.num_epochs, self.results_dir,
                                            intermediate_results, loss_values)
        except KeyboardInterrupt:
            print('-' * 89)
            print('Exiting from training early')

    def write_classification_metrics(self, res):
        write_lines_to_file(self.results_dir + 'anomaly_labels.txt', res.predicted_labels, new_line=True)
        write_lines_to_file(self.results_dir + "pred_outliers_indeces.txt", res.predicted_outliers, new_line=True)


    def final_prediction(self):
        predicted_labels = self.predict()
        result = self.determine_anomalies.determine(predicted_labels, self.no_anomaly)
        self.write_classification_metrics(result)
        self.write_final_metrics(self.results_dir, result)
        return result.f1, result.precision
