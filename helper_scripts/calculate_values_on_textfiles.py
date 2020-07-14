from sklearn.metrics import f1_score, precision_score
import numpy as np
from numpy import percentile
import seaborn as sns
import matplotlib.pyplot as plt


def calc_mean():
    with open('/Users/haraldott/Downloads/results/no finetune/bert_epochs_0_seq_len_7_description__reverse_order_0/anomaly_loss_values') as f:
        reverse_order_loss = [float(x) for x in f.readlines()]
    with open('/Users/haraldott/Downloads/results/no finetune/bert_epochs_100_seq_len_7_description__no_anomaly_1/anomaly_loss_values') as f:
        no_anomaly_loss = [float(x) for x in f.readlines()]

    print("reverse_order_loss: {}, std: {}".format(np.mean(reverse_order_loss), np.std(reverse_order_loss)))
    print("no_anomaly_loss: {}, std: {}".format(np.mean(no_anomaly_loss), np.std(no_anomaly_loss)))



def distribution_plots(results_dir, normal_vals, anomaly_vals, epochs, units, emb_size, precision=0):
    sns.distplot(normal_vals, hist=False, kde=True,
                 kde_kws={'linewidth': 3},
                 label='Dataset A')

    sns.distplot(anomaly_vals, hist=False, kde=True,
                 kde_kws={'linewidth': 3},
                 label='Dataset B')

    plt.legend(prop={'size': 16})
    #plt.title('{} Epochs, {} units, {} emb_size, p {}'.format(epochs, units, emb_size, precision))
    plt.xlabel('Loss value')
    plt.ylabel('Density')
    plt.savefig(results_dir + 'plot')
    plt.clf()
    plt.close('all')


def calc_percentile_outliers(normal_loss_values_path="/Users/haraldott/Downloads/results/regression/bert_epochs_130_seq_len_7_description__random_lines_1/normal_loss_values",
                             anomaly_loss_values_path="/Users/haraldott/Downloads/results/regression/bert_epochs_130_seq_len_7_description__random_lines_1/anomaly_loss_values",
                             perc=97.0):
    with open(normal_loss_values_path) as f:
        normal_loss_values = [float(y) for y in f.readlines()]
    with open(anomaly_loss_values_path) as f:
        anomaly_loss_values = [float(y) for y in f.readlines()]

    per = percentile(normal_loss_values, perc)

    pred_outliers_indeces = [i for i, val in enumerate(anomaly_loss_values) if val > per]
    print(len(pred_outliers_indeces))
    return pred_outliers_indeces



def calc_f1_based_on_percentile(normal_loss_values_path="/Users/haraldott/Downloads/bert_epochs_100_seq_len_7_anomaly_type_random_lines_1_hidden_128_layers_1_clip_1.0_experiment_default/normal_loss_values",
                                anomaly_loss_values_path="/Users/haraldott/Downloads/bert_epochs_100_seq_len_7_anomaly_type_random_lines_1_hidden_128_layers_1_clip_1.0_experiment_default/anomaly_loss_values",
                                ground_truth_path="/Users/haraldott/Downloads/bert_epochs_100_seq_len_7_anomaly_type_random_lines_1_hidden_128_layers_1_clip_1.0_experiment_default/test_anomaly_labels.txt"):
    with open(ground_truth_path) as f:
        ground_truth = [int(y) for y in f.readlines()]
    pred_indeces = calc_percentile_outliers(normal_loss_values_path, anomaly_loss_values_path, 99.1)

    with open(anomaly_loss_values_path) as f:
        anomaly_loss_values = f.readlines()

    pred_labels = np.zeros(len(anomaly_loss_values), dtype=int)
    for index_pred in pred_indeces:
        pred_labels[index_pred] = 1

    true_labels = np.zeros(len(anomaly_loss_values), dtype=int)
    for index_true in ground_truth:
        true_labels[index_true] = 1

    print("f1 score: {}".format(f1_score(true_labels, pred_labels)))
    print("precision: {}".format(precision_score(true_labels, pred_labels)))


def box_plots(normal_loss_values, anomaly_loss_values):
    with open(normal_loss_values) as f:
        normal_loss_values = [float(y) for y in f.readlines()]
    with open(anomaly_loss_values) as f:
        anomaly_loss_values = [float(y) for y in f.readlines()]

    data = [normal_loss_values, anomaly_loss_values]
    fig, ax = plt.subplots()
    ax.set_title("Box plots for training and test data")
    ax.boxplot(data)
    ax.set_xticklabels(["Training Data", "Test Data"])
    plt.savefig("/Users/haraldott/Development/thesis/detector/data/openstack/utah/results/UtahSorted/regression_xl_epochs_60_seq_len_7_anomaly_type_insert_words_1_hidden_512_layers_1_clip_1.0_experiment_x_alteration_ratio_0.05_anomaly_ratio_0.05/boxplots.png", dpi=300)
    plt.clf()

if __name__ == '__main__':
    box_plots('/Users/haraldott/Development/thesis/detector/data/openstack/utah/results/UtahSorted/regression_xl_epochs_60_seq_len_7_anomaly_type_insert_words_1_hidden_512_layers_1_clip_1.0_experiment_x_alteration_ratio_0.05_anomaly_ratio_0.05/normal_loss_values', '/Users/haraldott/Development/thesis/detector/data/openstack/utah/results/UtahSorted/regression_xl_epochs_60_seq_len_7_anomaly_type_insert_words_1_hidden_512_layers_1_clip_1.0_experiment_x_alteration_ratio_0.05_anomaly_ratio_0.05/anomaly_loss_values')

