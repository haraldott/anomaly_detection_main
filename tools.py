import os
from collections import defaultdict

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import StrMethodFormatter
from sklearn.metrics import roc_curve
from typing import Dict, List
from os import listdir
from os.path import isfile, join


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


# def comapare_seq_len():




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


def read_in_values(*files):
    # bert_vals = defaultdict(dict)
    bert_vals = {{"regression":
                      {"sequential": {"delete_lines": {}, "duplicate_lines": {}, "shuffle_lines": {}},
                       "qualitative": {"insert_words": {}, "remove_words": {}, "replace_words": {}}}},
                 {"multiclass":
                      {"sequential": {"delete_lines": {}, "duplicate_lines": {}, "shuffle_lines": {}},
                       "qualitative": {"insert_words": {}, "remove_words": {}, "replace_words": {}}}}}
    gpt_vals = defaultdict(dict)
    # gpt_vals = {{"regression":
    #                   {"sequential": {"delete_lines": {}, "duplicate_lines": {}, "shuffle_lines": {}},
    #                    "qualitative": {"insert_words": {}, "remove_words": {}, "replace_words": {}}}},
    #              {"multiclass":
    #                   {"sequential": {"delete_lines": {}, "duplicate_lines": {}, "shuffle_lines": {}},
    #                    "qualitative": {"insert_words": {}, "remove_words": {}, "replace_words": {}}}}}
    xl_vals = defaultdict(dict)
    # xl_vals = {{"regression":
    #                   {"sequential": {"delete_lines": {}, "duplicate_lines": {}, "shuffle_lines": {}},
    #                    "qualitative": {"insert_words": {}, "remove_words": {}, "replace_words": {}}}},
    #              {"multiclass":
    #                   {"sequential": {"delete_lines": {}, "duplicate_lines": {}, "shuffle_lines": {}},
    #                    "qualitative": {"insert_words": {}, "remove_words": {}, "replace_words": {}}}}}


    for file in files:
        with open(file, "r") as f:
            line = f.readline()
            split_line = [float(v) for v in line.split(",")]

            classification = os.path.splitext(file)[0].split("_")[1]

            if "delete_lines" in file:
                if "bert" in file:
                    bert_vals[classification]["sequential"]["delete_lines"][split_line[0]] = [split_line[1], split_line[2], split_line[3]]
                elif "gpt" in file:
                    gpt_vals[classification]["sequential"]["delete_lines"][split_line[0]] = [split_line[1], split_line[2], split_line[3]]
                elif "xl" in file:
                    xl_vals[classification]["sequential"]["delete_lines"][split_line[0]] = [split_line[1], split_line[2], split_line[3]]
            elif "duplicate_lines" in file:
                if "bert" in file:
                    bert_vals[classification]["sequential"]["duplicate_lines"][split_line[0]] = [split_line[1], split_line[2], split_line[3]]
                elif "gpt" in file:
                    gpt_vals[classification]["sequential"]["duplicate_lines"][split_line[0]] = [split_line[1], split_line[2], split_line[3]]
                elif "xl" in file:
                    xl_vals[classification]["sequential"]["duplicate_lines"][split_line[0]] = [split_line[1], split_line[2], split_line[3]]
            elif "shuffle" in file:
                if "bert" in file:
                    bert_vals[classification]["sequential"]["shuffle_lines"][split_line[0]] = [split_line[1], split_line[2], split_line[3]]
                elif "gpt" in file:
                    gpt_vals[classification]["sequential"]["shuffle_lines"][split_line[0]] = [split_line[1], split_line[2], split_line[3]]
                elif "xl" in file:
                    xl_vals[classification]["sequential"]["shuffle_lines"][split_line[0]] = [split_line[1], split_line[2], split_line[3]]
            elif "shuffle" in file:
                if "bert" in file:
                    bert_vals[classification]["sequential"]["shuffle_lines"][split_line[0]] = [split_line[1], split_line[2], split_line[3]]
                elif "gpt" in file:
                    gpt_vals[classification]["sequential"]["shuffle_lines"][split_line[0]] = [split_line[1], split_line[2], split_line[3]]
                elif "xl" in file:
                    xl_vals[classification]["sequential"]["shuffle_lines"][split_line[0]] = [split_line[1], split_line[2], split_line[3]]
            elif "insert" in file:
                if "bert" in file:
                    bert_vals[classification]["qualitative"]["insert_words"][split_line[0]] = [split_line[1], split_line[2], split_line[3]]
                elif "gpt" in file:
                    gpt_vals[classification]["qualitative"]["insert_words"][split_line[0]] = [split_line[1], split_line[2], split_line[3]]
                elif "xl" in file:
                    xl_vals[classification]["qualitative"]["insert_words"][split_line[0]] = [split_line[1], split_line[2], split_line[3]]
            elif "remove" in file:
                if "bert" in file:
                    bert_vals[classification]["qualitative"]["remove_words"][split_line[0]] = [split_line[1], split_line[2], split_line[3]]
                elif "gpt" in file:
                    gpt_vals[classification]["qualitative"]["remove_words"][split_line[0]] = [split_line[1], split_line[2], split_line[3]]
                elif "xl" in file:
                    xl_vals[classification]["qualitative"]["remove_words"][split_line[0]] = [split_line[1], split_line[2], split_line[3]]
            elif "replace" in file:
                if "bert" in file:
                    bert_vals[classification]["qualitative"]["replace_words"][split_line[0]] = [split_line[1], split_line[2], split_line[3]]
                elif "gpt" in file:
                    gpt_vals[classification]["qualitative"]["replace_words"][split_line[0]] = [split_line[1], split_line[2], split_line[3]]
                elif "xl" in file:
                    xl_vals[classification]["qualitative"]["replace_words"][split_line[0]] = [split_line[1], split_line[2], split_line[3]]


    val = 1+1



########################################################################
# EINZELN REGRESSION
########################################################################

# REGRESSION QUALITATIVE
# compare_approaches(bert=[0.78,0.64,1.00], gpt=[0.96,0.93,1.00], xl=[0.56,0.53,0.58], plotpath="/Users/haraldott/Development/thesis/detector/results_qualitative/regression_insert_words_anomaly_ratio_0.05.png")
# compare_approaches(bert=[0.74,0.58,1.00], gpt=[0.96,0.91,1.00], xl=[0.52,0.53,0.51], plotpath="/Users/haraldott/Development/thesis/detector/results_qualitative/regression_insert_words_anomaly_ratio_0.10.png")
# compare_approaches(bert=[0.70,0.53,1.00], gpt=[0.97,0.94,1.00], xl=[0.47,0.38,0.60], plotpath="/Users/haraldott/Development/thesis/detector/results_qualitative/regression_insert_words_anomaly_ratio_0.15.png")
#
# compare_approaches(bert=[0.76,0.61,1.00], gpt=[0.87,0.78,1.00], xl=[0.56,0.52,0.60], plotpath="/Users/haraldott/Development/thesis/detector/results_qualitative/regression_remove_words_anomaly_ratio_0.05.png")
# compare_approaches(bert=[0.70,0.54,1.00], gpt=[0.81,0.68,1.00], xl=[0.46,0.37,0.60], plotpath="/Users/haraldott/Development/thesis/detector/results_qualitative/regression_remove_words_anomaly_ratio_0.10.png")
# compare_approaches(bert=[0.66,0.49,1.00], gpt=[0.74,0.59,1.00], xl=[0.43,0.33,0.62], plotpath="/Users/haraldott/Development/thesis/detector/results_qualitative/regression_remove_words_anomaly_ratio_0.15.png")
#
# compare_approaches(bert=[0.76,0.62,1.00], gpt=[0.96,0.91,1.00], xl=[0.50,0.43,0.59], plotpath="/Users/haraldott/Development/thesis/detector/results_qualitative/regression_replace_words_anomaly_ratio_0.05.png")
# compare_approaches(bert=[0.68,0.51,1.00], gpt=[0.93,0.87,1.00], xl=[0.48,0.39,0.62], plotpath="/Users/haraldott/Development/thesis/detector/results_qualitative/regression_replace_words_anomaly_ratio_0.10.png")
# compare_approaches(bert=[0.63,0.46,1.00], gpt=[0.91,0.83,1.00], xl=[0.37,0.26,0.61], plotpath="/Users/haraldott/Development/thesis/detector/results_qualitative/regression_replace_words_anomaly_ratio_0.15.png")
#
#
# # REGRESSION SEQUENTIAL
# compare_approaches(bert=[0.71,0.56,1.00], gpt=[0.96,0.92,1.00], xl=[0.43,0.34,0.60], plotpath="/Users/haraldott/Development/thesis/detector/results_sequential/regression/regression_delete_lines_anomaly_ratio_0.05.png")
# compare_approaches(bert=[0.60,0.43,1.00], gpt=[0.96,0.93,1.00], xl=[0.33,0.22,0.65], plotpath="/Users/haraldott/Development/thesis/detector/results_sequential/regression/regression_delete_lines_anomaly_ratio_0.10.png")
# compare_approaches(bert=[0.52,0.36,1.00], gpt=[0.96,0.92,1.00], xl=[0.27,0.17,0.63], plotpath="/Users/haraldott/Development/thesis/detector/results_sequential/regression/regression_delete_lines_anomaly_ratio_0.15.png")
#
# compare_approaches(bert=[0.75,0.60,1.00], gpt=[0.93,0.87,1.00], xl=[0.67,0.51,0.95], plotpath="/Users/haraldott/Development/thesis/detector/results_sequential/regression/regression_duplicate_lines_anomaly_ratio_0.05.png")
# compare_approaches(bert=[0.60,0.43,0.99], gpt=[0.93,0.86,1.00], xl=[0.51,0.35,0.94], plotpath="/Users/haraldott/Development/thesis/detector/results_sequential/regression/regression_duplicate_lines_anomaly_ratio_0.10.png")
# compare_approaches(bert=[0.53,0.36,1.00], gpt=[0.89,0.80,1.00], xl=[0.44,0.29,0.93], plotpath="/Users/haraldott/Development/thesis/detector/results_sequential/regression/regression_duplicate_lines_anomaly_ratio_0.15.png")
#
# compare_approaches(bert=[0.64,0.47,1.00], gpt=[0.97,0.95,1.00], xl=[0.37,0.26,0.64], plotpath="/Users/haraldott/Development/thesis/detector/results_sequential/regression/regression_shuffle_lines_anomaly_ratio_0.05.png")
# compare_approaches(bert=[0.65,0.48,1.00], gpt=[0.95,0.91,1.00], xl=[0.36,0.26,0.61], plotpath="/Users/haraldott/Development/thesis/detector/results_sequential/regression/regression_shuffle_lines_anomaly_ratio_0.10.png")
# compare_approaches(bert=[0.65,0.48,1.00], gpt=[0.96,0.91,1.00], xl=[0.36,0.28,0.51], plotpath="/Users/haraldott/Development/thesis/detector/results_sequential/regression/regression_shuffle_lines_anomaly_ratio_0.15.png")
#
#
#



#
# # REGRESSION QUALITATIVE
# qualitative 5 percent average
compare_approaches(bert=[round((0.78+0.76+0.76)/3, 2), round((0.64+0.61+0.62) / 3, 2), 1.00], gpt=[round((0.96+0.87+0.96) / 3, 2),round((0.93+0.78+0.91) / 3, 2),1.00], xl=[round((0.56+0.56+0.5) / 3,2), round((0.53+0.56+0.50) / 3, 2),round((0.58+0.6+0.59) / 3, 2)], plotpath="/Users/haraldott/Development/thesis/detector/results_qualitative/regression/regression_qualitative_average_ratio_0.05.png")

# qualitative 10 percent average
compare_approaches(bert=[round((0.74+0.7+0.68) / 3, 2),round((0.58+0.54+0.51) / 3, 2),1.00], gpt=[round((0.96 + 0.81 + 0.93) / 3, 2),(0.91),1.00], xl=[round((0.52+0.46+0.48) / 3, 2), round((0.53+0.37 + 0.39) / 3, 2),round((0.51+0.6+0.62) / 3, 2)], plotpath="/Users/haraldott/Development/thesis/detector/results_qualitative/regression/regression_qualitative_average_ratio_0.10.png")

# qualitative 15 percent average
compare_approaches(bert=[round((0.70 +0.66 +0.63 )/3,2) , round((0.53 +0.49 + 0.46)/3,2) ,round((1.00 +1.00+ 1.00)/3,2) ], gpt=[round((0.97 +0.74 + 0.91)/3,2) ,round((0.94 +0.59 + 0.83)/3,2) ,round((1.00 +1.00 +1.00 )/3,2) ], xl=[round((0.47 +0.43 + 0.37)/3,2) ,round((0.38 + 0.33+0.26 )/3,2) ,round((0.60 + 0.62+ 0.61)/3,2) ], plotpath="/Users/haraldott/Development/thesis/detector/results_qualitative/regression/regression_qualitative_average_ratio_0.15.png")
#
#
# REGRESSION SEQUENTIAL
# sequential 5 percent average
compare_approaches(bert=[round((0.71+0.71+0.64)/3, 2), round((0.56+0.47+0.62) / 3, 2), round((1.00 +1.00+ 1.00)/3,2)], gpt=[round((0.96+0.93+0.97) / 3, 2),round((0.92+0.87+0.95) / 3, 2),round((1.00 +1.00 +1.00 )/3,2)], xl=[round((0.43+0.45+0.37) / 3,2), round((0.34+0.36+0.26) / 3, 2),round((0.60+0.60+0.64) / 3, 2)], plotpath="/Users/haraldott/Development/thesis/detector/results_sequential/regression/regression_sequential_average_ratio_0.05.png")

# sequential 10 percent average
compare_approaches(bert=[round((0.60+0.58+0.65)/3, 2), round((0.43+0.48+0.51) / 3, 2), round((1.00 +1.00+ 1.00)/3,2)], gpt=[round((0.96 + 0.93 + 0.95) / 3, 2),round((0.93+0.86+0.91) / 3),round((1.00 +1.00 +1.00 )/3,2)], xl=[round((0.33+0.34+0.36) / 3, 2), round((0.22+0.23 + 0.26) / 3, 2),round((0.65+0.68+0.61) / 3, 2)], plotpath="/Users/haraldott/Development/thesis/detector/results_sequential/regression/regression_sequential_average_ratio_0.10.png")

# sequential 15 percent average
compare_approaches(bert=[round((0.52+0.52+0.65)/3, 2), round((0.36 +0.48 + 0.46)/3,2) ,round((1.00 +1.00+ 1.00)/3,2) ], gpt=[round((0.96 +0.89 + 0.96)/3,2) ,round((0.92 +0.80 + 0.91)/3,2) ,round((1.00 +1.00 +1.00 )/3,2) ], xl=[round((0.27 +0.30 + 0.36)/3,2) ,round((0.17 + 0.19+0.28)/3,2) ,round((0.63 + 0.74+ .51)/3,2) ], plotpath="/Users/haraldott/Development/thesis/detector/results_sequential/regression/regression_sequential_average_ratio_0.15.png")


# MULTICLASS SEQUENTIAL
# sequential 5 percent average
compare_approaches(bert=[round((0.72+0.69+0.59)/3, 2), round((0.56+0.52+0.42) / 3, 2), round((1.00 +1.00+ 1.00)/3,2)], gpt=[round((0.49+0.47+0.35) / 3, 2),round((0.36+0.36+0.24) / 3, 2),round((0.75 +0.68 +0.65 )/3,2)], xl=[round((0.52+0.53+0.47) / 3,2), round((0.36+0.36+0.31) / 3, 2),round((1.00+1.00+1.00) / 3, 2)], plotpath="/Users/haraldott/Development/thesis/detector/results_sequential/multiclass/multiclass_sequential_average_ratio_0.05.png")

# sequential 10 percent average
compare_approaches(bert=[round((0.61+0.57+0.60)/3, 2), round((0.44+0.40+0.43) / 3, 2), round((1.00 +1.00+ 1.00)/3,2)], gpt=[round((0.38 + 0.41 + 0.39) / 3, 2),round((0.26+0.29+0.27) / 3),round((0.76 +0.71 +0.70 )/3,2)], xl=[round((0.45+0.44+0.47) / 3, 2), round((0.29+0.29 + 0.30) / 3, 2),round((1.00+1.00+1.00) / 3, 2)], plotpath="/Users/haraldott/Development/thesis/detector/results_sequential/multiclass/multiclass_sequential_average_ratio_0.10.png")

# sequential 15 percent average
compare_approaches(bert=[round((0.51+0.49+0.61)/3, 2), round((0.34 +0.33 + 0.44)/3,2) ,round((1.00 +1.00+ 1.00)/3,2) ], gpt=[round((0.32 +0.38 + 0.37)/3,2) ,round((0.21 +0.26 + 0.25)/3,2) ,round((0.69 +0.72 +0.68 )/3,2) ], xl=[round((0.40 +0.39 + 0.45)/3,2) ,round((0.25 + 0.24+0.29)/3,2) ,round((1.00 + 1.00+1.00)/3,2) ], plotpath="/Users/haraldott/Development/thesis/detector/results_sequential/multiclass/multiclass_sequential_average_ratio_0.15.png")


# MULTICLASS QUALITATIVE
# qualitative 5 percent average
compare_approaches(bert=[round((0.80+0.76+0.76)/3, 2), round((0.67+0.61+0.62) / 3, 2), round((1.00 +1.00+ 1.00)/3,2)], gpt=[round((0.51+0.54+0.55) / 3, 2),round((0.41+0.44+0.43) / 3, 2),round((0.69 +0.69 +0.74 )/3,2)], xl=[round((0.60+0.58+0.57) / 3,2), round((0.43+0.41+0.40) / 3, 2),round((1.00+1.00+1.00) / 3, 2)], plotpath="/Users/haraldott/Development/thesis/detector/results_qualitative/multiclass/multiclass_qualitative_average_ratio_0.05.png")

# qualitative 10 percent average
compare_approaches(bert=[round((0.71+0.69+0.69)/3, 2), round((0.55+0.53+0.52) / 3, 2), round((1.00 +1.00+ 1.00)/3,2)], gpt=[round((0.46 + 0.46 + 0.45) / 3, 2),round((0.35+0.35+0.33) / 3, 2),round((0.67 +0.67 +0.68)/3,2)], xl=[round((0.58+0.55+0.51) / 3, 2), round((0.41+0.38 + 0.34) / 3, 2),round((1.00+1.00+1.00) / 3, 2)], plotpath="/Users/haraldott/Development/thesis/detector/results_qualitative/multiclass/multiclass_qualitative_average_ratio_0.10.png")

# qualitative 15 percent average
compare_approaches(bert=[round((0.70+0.67+0.63)/3, 2), round((0.53 +0.50 + 0.46)/3,2) ,round((1.00 +1.00+ 1.00)/3,2) ], gpt=[round((0.44 +0.47 + 0.38)/3,2) ,round((0.32 +0.34 + 0.26)/3,2) ,round((0.70 +0.73 +0.67 )/3,2) ], xl=[round((0.57 +0.53 + 0.48)/3,2) ,round((0.40 +0.36+0.31)/3,2) ,round((1.00 + 1.00+ 1.00)/3,2) ], plotpath="/Users/haraldott/Development/thesis/detector/results_qualitative/multiclass/multiclass_qualitative_average_ratio_0.15.png")

# compare_approaches(bert=[0.80,0.67,1.00], gpt=[0.56,0.44,0.77], xl=[0.48,0.31,1.00], plotpath="/Users/haraldott/Development/thesis/detector/results_qualitative/multiclass/multiclass_insert_words_5_percent.png")
# compare_approaches(bert=[0.71,0.55,1.00], gpt=[0.53,0.39,0.81], xl=[0.45,0.29,1.00], plotpath="/Users/haraldott/Development/thesis/detector/results_qualitative/multiclass/multiclass_insert_words_10_percent.png")
# compare_approaches(bert=[0.70,0.53,1.00], gpt=[0.47,0.33,0.80], xl=[0.43,0.27,1.00], plotpath="/Users/haraldott/Development/thesis/detector/results_qualitative/multiclass/multiclass_insert_words_15_percent.png")
#
# compare_approaches(bert=[0.76,0.61,1.00], gpt=[0.59,0.46,0.82], xl=[0.48,0.32,1.00], plotpath="/Users/haraldott/Development/thesis/detector/results_qualitative/multiclass/multiclass_remove_words_5_percent.png")
# compare_approaches(bert=[0.68,0.52,1.00], gpt=[0.56,0.43,0.81], xl=[0.44,0.28,1.00], plotpath="/Users/haraldott/Development/thesis/detector/results_qualitative/multiclass/multiclass_remove_words_10_percent.png")
# compare_approaches(bert=[0.64,0.47,1.00], gpt=[0.51,0.37,0.81], xl=[0.41,0.26,1.00], plotpath="/Users/haraldott/Development/thesis/detector/results_qualitative/multiclass/multiclass_remove_words_15_percent.png")
#
# compare_approaches(bert=[0.73,0.58,1.00], gpt=[0.57,0.45,0.81], xl=[0.48,0.31,1.00], plotpath="/Users/haraldott/Development/thesis/detector/results_qualitative/multiclass/multiclass_replace_words_5_percent.png")
# compare_approaches(bert=[0.68,0.51,1.00], gpt=[0.52,0.37,0.84], xl=[0.43,0.27,1.00], plotpath="/Users/haraldott/Development/thesis/detector/results_qualitative/multiclass/multiclass_replace_words_10_percent.png")
# compare_approaches(bert=[0.58,0.41,1.00], gpt=[0.47,0.32,0.84], xl=[0.39,0.24,1.00], plotpath="/Users/haraldott/Development/thesis/detector/results_qualitative/multiclass/multiclass_replace_words_15_percent.png")
#
# compare_approaches(bert=[0.84,0.73,1.00], gpt=[0.25,0.19,0.34], xl=[0.89,0.85,0.93], plotpath="/Users/haraldott/Development/thesis/detector/results_sequential/multiclass/multiclass_delete_lines_5_percent.png")
# compare_approaches(bert=[0.79,0.66,1.00], gpt=[0.21,0.14,0.40], xl=[0.84,0.79,0.90], plotpath="/Users/haraldott/Development/thesis/detector/results_sequential/multiclass/multiclass_delete_lines_10_percent.png")
# compare_approaches(bert=[0.74,0.58,1.00], gpt=[0.24,0.15,0.51], xl=[0.79,0.72,0.88], plotpath="/Users/haraldott/Development/thesis/detector/results_sequential/multiclass/multiclass_delete_lines_15_percent.png")
#
# compare_approaches(bert=[0.84,0.73,0.99], gpt=[0.23,0.17,0.37], xl=[0.67,0.51,0.95], plotpath="/Users/haraldott/Development/thesis/detector/results_sequential/multiclass/multiclass_duplicate_lines_5_percent.png")
# compare_approaches(bert=[0.77,0.63,0.99], gpt=[0.21,0.14,0.41], xl=[0.51,0.35,0.94], plotpath="/Users/haraldott/Development/thesis/detector/results_sequential/multiclass/multiclass_duplicate_lines_10_percent.png")
# compare_approaches(bert=[0.73,0.58,0.99], gpt=[0.22,0.16,0.33], xl=[0.44,0.29,0.93], plotpath="/Users/haraldott/Development/thesis/detector/results_sequential/multiclass/multiclass_duplicate_lines_15_percent.png")
#
# compare_approaches(bert=[0.66,0.50,1.00], gpt=[0.23,0.15,0.49], xl=[0.62,0.46,0.96], plotpath="/Users/haraldott/Development/thesis/detector/results_sequential/multiclass/multiclass_shuffle_lines_words_5_percent.png")
# compare_approaches(bert=[0.66,0.50,1.00], gpt=[0.21,0.15,0.37], xl=[0.63,0.49,0.88], plotpath="/Users/haraldott/Development/thesis/detector/results_sequential/multiclass/multiclass_shuffle_lines_words_10_percent.png")
# compare_approaches(bert=[0.69,0.53,1.00], gpt=[0.25,0.18,0.43], xl=[0.59,0.42,0.98], plotpath="/Users/haraldott/Development/thesis/detector/results_sequential/multiclass/multiclass_shuffle_lines_words_15_percent.png")



#show_f1_score_injection_ratio("/Users/haraldott/Downloads/anomaly_only_results.txt")