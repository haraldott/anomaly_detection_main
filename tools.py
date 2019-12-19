import seaborn as sns
import matplotlib.pyplot as plt


def distribution_plots(dir, epochs, units, emb_size, precision=0.0):
    normal_vals = open(dir + 'normal_loss_values', 'r').readlines()
    anomaly_vals = open(dir + 'anomaly_loss_values',
                        'r').readlines()

    normal_vals = [float(x) for x in normal_vals]
    anomaly_vals = [float(x) for x in anomaly_vals]

    sns.distplot(normal_vals, hist=False, kde=True,
                 kde_kws={'linewidth': 3},
                 label='normal')

    sns.distplot(anomaly_vals, hist=False, kde=True,
                 kde_kws={'linewidth': 3},
                 label='anomaly')

    plt.legend(prop={'size': 16}, title='n and a')
    plt.title('{} Epochs, {} units, {} emb_size, p {}'.format(epochs, units, emb_size, precision))
    plt.xlabel('Loss value')
    plt.ylabel('Density')
    plt.savefig(dir + 'plot')
    plt.clf()


def calc_precision_utah(anomalies_file, outliers_file):
    file = open(anomalies_file)
    lines = file.readlines()
    outliers = open(outliers_file)
    outliers = outliers.readlines()

    instances_containing_anomalies = [
        "544fd51c-4edc-4780-baae-ba1d80a0acfc",
        "ae651dff-c7ad-43d6-ac96-bbcd820ccca8",
        "a445709b-6ad0-40ec-8860-bec60b6ca0c2",
        "1643649d-2f42-4303-bfcd-7798baec19f9"
    ]
    anomaly_idx = []
    for i, line in enumerate(lines):
        if any(substring in line for substring in instances_containing_anomalies):
            anomaly_idx.append(i + 1)

    detected_anomalies = []
    for line in outliers:
        x = line.split(',')
        detected_anomalies.append(int(x[0]))

    tp = 0
    fp = 0

    for el in anomaly_idx:
        if el in detected_anomalies:
            tp += 1
        else:
            fp += 1

    if tp is 0:
        return 0
    else:
        return tp / (tp + fp)
