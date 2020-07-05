import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import StrMethodFormatter


def distribution_plots(results_dir, normal_vals, anomaly_vals, epochs, units, emb_size, precision=0):
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
    plt.savefig(results_dir + 'plot')
    plt.clf()


def show_f1_score_injection_ratio(f1_file):
    scores, ratios = [], []
    with open(f1_file, 'r') as file:
        lines = file.readlines()
        for line in lines:
            ratio, f1 = line.split(",")
            ratios.append(float(ratio))
            scores.append(float(f1))

    plt.ylim(bottom=0.0, top=1.0)
    plt.xlim(0.015, 0.2)
    plt.plot(ratios, scores, '.-')
    plt.xlabel('Injection ratio')
    plt.ylabel('F1 score')
    plt.savefig("f1plot.png", dpi=500)

show_f1_score_injection_ratio("/Users/haraldott/Downloads/anomaly_only_results.txt")