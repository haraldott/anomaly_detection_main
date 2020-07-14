import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import StrMethodFormatter
from sklearn.metrics import roc_curve
from typing import Dict, List


def distribution_plots(results_dir, normal_vals, anomaly_vals, epochs, units, emb_size, precision=0):
    plt.figure()
    sns.distplot(normal_vals, hist=False, kde=True,
                 kde_kws={'linewidth': 3},
                 label='train data')

    sns.distplot(anomaly_vals, hist=False, kde=True,
                 kde_kws={'linewidth': 3},
                 label='test data')

    plt.legend()
    #plt.title('{} Epochs, {} units, {} emb_size, p {}'.format(epochs, units, emb_size, precision))
    plt.xlabel('Loss value')
    plt.ylabel('Density')
    plt.savefig(results_dir + 'regression_lost.png', dpi=300)
    plt.close('all')


def show_f1_score_injection_ratio(f1_file):
    scores, ratios = [], []
    with open(f1_file, 'r') as file:
        lines = file.readlines()
        for line in lines:
            ratio, f1 = line.split(",")
            ratios.append(float(ratio))
            scores.append(float(f1))
    plt.figure()
    plt.ylim(bottom=0.0, top=1.0)
    plt.xlim(0.015, 0.2)
    plt.plot(ratios, scores, '.-')
    plt.xlabel('Injection ratio')
    plt.ylabel('F1 score')
    plt.savefig("f1plot.png", dpi=300)
    plt.close('all')


def plot_loss_for_finetuning(loss_eval, training_loss):
    # plot_loss_for_finetuning([4.26, 4.05, 3.9, 3.85],[7.12, 4.45, 2.54, 1.88])
    plt.figure()
    plt.plot([1,2,3,4], loss_eval, 'o-', label='Evaluation Loss', color='red')
    plt.plot([1,2,3,4], training_loss, 'o-', label='Training Loss', color='blue')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig("finetuning_loss.png", dpi=300)
    plt.close('all')

def plot_roc_curve(true_labels, pred_labels, results_dir):
    plt.figure()
    fpr, tpr, thresholds = roc_curve(true_labels, pred_labels)
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.savefig(results_dir + "roc_curve.png", dpi=300)
    plt.close('all')


def compare_approaches(bert: List, xl: List, gpt: List, plotpath):
    #compare_approaches(bert={"precision": 0.5, "f1": 0.6, "recall": 0.9}, bert={"precision": 0.8, "f1": 0.7, "recall": 1.0}, gpt={"precision": 0.7, "f1": 0.6, "recall": 0.97})
    plt.figure()

    labels = ['Bert', 'XL', 'GPT-2']
    f1s = [bert[0], xl[0], gpt[0]]
    precisions = [bert[1], xl[1], gpt[1]]
    recalls = [bert[2], xl[2], gpt[2]]

    x = np.arange(len(labels))  # the label locations
    width = 0.30  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x, precisions, width, label='Precision', color='red')
    rects2 = ax.bar(x - width, f1s, width, label='F1', color='blue')
    rects3 = ax.bar(x + width, recalls, width, label='Recall', color='orange')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Scores')
    #ax.set_title('Scores by group and gender')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    fig.tight_layout()

    plt.savefig(plotpath, dpi=300)
    plt.clf()
    plt.close()

# compare_approaches(bert=[0.70,0.54,1.00], gpt=[0.53,0.39,0.83], xl=[0.56,0.39,1.00], plotpath="/Users/haraldott/Downloads/results_sequential/delete_lines_5_percent.png")

#show_f1_score_injection_ratio("/Users/haraldott/Downloads/anomaly_only_results.txt")