import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import StrMethodFormatter
from sklearn.metrics import roc_curve


def distribution_plots(results_dir, normal_vals, anomaly_vals, epochs, units, emb_size, precision=0):
    plt.figure()
    sns.distplot(normal_vals, hist=False, kde=True,
                 kde_kws={'linewidth': 3},
                 label='normal')

    sns.distplot(anomaly_vals, hist=False, kde=True,
                 kde_kws={'linewidth': 3},
                 label='anomaly')

    plt.legend()
    #plt.title('{} Epochs, {} units, {} emb_size, p {}'.format(epochs, units, emb_size, precision))
    plt.xlabel('Loss value')
    plt.ylabel('Density')
    plt.savefig(results_dir + 'regression_lost.png', dpi=300)
    plt.close()


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
    plt.close()


def plot_loss_for_finetuning(loss_eval, training_loss):
    # plot_loss_for_finetuning([4.26, 4.05, 3.9, 3.85],[7.12, 4.45, 2.54, 1.88])
    plt.figure()
    plt.plot([1,2,3,4], loss_eval, 'o-', label='Evaluation Loss', color='red')
    plt.plot([1,2,3,4], training_loss, 'o-', label='Training Loss', color='blue')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig("finetuning_loss.png", dpi=300)
    plt.close()

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
    plt.close()


def compare_approaches(glove, bert, gpt):
    plt.figure()


#show_f1_score_injection_ratio("/Users/haraldott/Downloads/anomaly_only_results.txt")