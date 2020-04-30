import argparse

import matplotlib

matplotlib.use('Agg')
from settings import settings
from wordembeddings.transform_glove import merge_templates
import logparser.Drain.Drain_demo as drain
import wordembeddings.transform_bert as transform_bert
from loganaliser.main import AnomalyDetection
from wordembeddings.bert_finetuning import finetune
from shared_functions import calculate_precision_and_plot, get_cosine_distance, inject_anomalies
import os
from wordembeddings.visualisation import write_to_tsv_files_bert_sentences
from shared_functions import get_embeddings

def experiment(option='Normal', seq_len=7, n_layers=1, n_hidden_units=128, batch_size=64, clip=1.1, epochs=10,
               anomaly_only=False, finetuning=False, anomaly_type='no_anomaly', anomaly_amount=1, embeddings_model='bert',
               experiment='default'):
    cwd = os.getcwd() + "/"
    print("starting {} {}".format(anomaly_type, anomaly_amount))

    no_anomaly = True if anomaly_type == "no_anomaly" else False

    if finetuning:
        results_dir = settings[option]["results_dir"] + "_finetune/"
    else:
        results_dir = settings[option]["results_dir"] + "/"

    results_dir_experiment = "{}_epochs_{}_seq_len_{}_anomaly_type_{}_{}_hidden_{}_layers_{}_clip_{}_experiment: {}/".format(
        results_dir + embeddings_model, epochs, seq_len, anomaly_type, anomaly_amount, n_hidden_units, n_layers, clip, experiment)

    normal = settings[option]["raw_normal"]
    anomaly = settings[option]["raw_anomaly"]

    raw_dir = settings[option]["raw_dir"]

    parsed_dir = settings[option]["parsed_dir"]

    embeddings_dir = settings[option]["embeddings_dir"]

    logtype = settings[option]["logtype"]

    instance_information_file_normal = settings[option]['instance_information_file_normal']
    instance_information_file_anomalies_pre_inject = settings[option][
        'instance_information_file_anomalies_pre_inject']
    instance_information_file_anomalies_injected = settings[option][
                                                       'instance_information_file_anomalies_injected'] + anomaly + "_" + anomaly_type + "_" + str(
        anomaly_amount)

    anomalies_injected_dir = parsed_dir + "anomalies_injected/"
    anomaly_indeces_dir = parsed_dir + "anomalies_injected/anomaly_indeces/"

    # corpus files produced by Drain
    corpus_normal = cwd + parsed_dir + normal + '_corpus'
    corpus_pre_anomaly = cwd + parsed_dir + anomaly + '_corpus'

    # bert vectors as pickle files
    embeddings_normal = cwd + embeddings_dir + normal + '.pickle'
    embeddings_anomalies_injected = cwd + embeddings_dir + anomaly + '.pickle'

    if finetuning:
        finetuning_model_dir = "wordembeddings/finetuning-models/" + normal
        lstm_model_save_path = cwd + 'loganaliser/saved_models/' + normal + '_with_finetune' + '_lstm.pth'
    else:
        finetuning_model_dir = "bert-base-uncased"
        lstm_model_save_path = cwd + 'loganaliser/saved_models/' + normal + "_" + experiment + '_lstm.pth'

    # take corpus parsed by drain, inject anomalies in this file
    anomaly_injected_corpus = cwd + anomalies_injected_dir + anomaly + "_" + anomaly_type
    # put here the information about which line is an anomaly from anomaly injection
    anomaly_indeces = cwd + results_dir_experiment + "true_anomaly_labels.txt"
    # anomaly_indeces_2 = cwd + anomaly_indeces_dir_2 + anomaly_2 + "_" + anomaly_type + "_" + anomaly_amount  + '_anomaly_indeces.txt'

    # create all directories, if they don't exist yet
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(results_dir_experiment, exist_ok=True)
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(parsed_dir, exist_ok=True)
    os.makedirs(embeddings_dir, exist_ok=True)
    os.makedirs(anomalies_injected_dir, exist_ok=True)
    os.makedirs(anomaly_indeces_dir, exist_ok=True)

    ### DRAIN PARSING
    # if not os.path.exists(corpus_normal) or not os.path.exists(corpus_pre_anomaly):
    drain.execute(directory=raw_dir, file=normal, output=parsed_dir, logtype=logtype)
    drain.execute(directory=raw_dir, file=anomaly, output=parsed_dir, logtype=logtype)

    ### INJECT ANOMALIES in dataset 2
    anomalies_true, lines_before_alter, lines_after_alter = inject_anomalies(anomaly_type=anomaly_type,
                                                                             corpus_input=corpus_pre_anomaly,
                                                                             corpus_output=anomaly_injected_corpus,
                                                                             anomaly_indices_output_path=anomaly_indeces,
                                                                             instance_information_in=instance_information_file_anomalies_pre_inject,
                                                                             instance_information_out=instance_information_file_anomalies_injected,
                                                                             anomaly_amount=anomaly_amount,
                                                                             results_dir=results_dir_experiment)

    # produce templates out of the corpuses that we have from the anomaly file
    templates_normal = list(set(open(corpus_normal, 'r').readlines()))
    # merge_templates(templates_normal_1, templates_normal_2, merged_template_path=parsed_dir_1 + "_merged_templates_normal")
    templates_anomalies_injected = list(set(open(anomaly_injected_corpus, 'r').readlines()))
    merged_templates = merge_templates(templates_normal, templates_anomalies_injected,
                                       merged_template_path=None)
    merged_templates = list(merged_templates)

    if finetuning:
        if not os.path.exists(finetuning_model_dir):
            finetune(templates=templates_normal, output_dir=finetuning_model_dir)

    word_embeddings = get_embeddings(embeddings_model, merged_templates, finetuning_model_dir)

    write_to_tsv_files_bert_sentences(vectors=word_embeddings, sentences=merged_templates,
                                      tsv_file_vectors=results_dir_experiment + "visualisation/vectors.tsv",
                                      tsv_file_sentences=results_dir_experiment + "visualisation/sentences.tsv")

    if anomaly_type in ["insert_words", "remove_words", "replace_words"]:
        get_cosine_distance(lines_before_alter, lines_after_alter, merged_templates, results_dir_experiment,
                            word_embeddings)

    # transform output of bert into numpy word embedding vectors
    transform_bert.transform(sentence_embeddings=word_embeddings, logfile=corpus_normal,
                             templates=merged_templates,  outputfile=embeddings_normal)

    transform_bert.transform(sentence_embeddings=word_embeddings, logfile=anomaly_injected_corpus,
                             templates=merged_templates, outputfile=embeddings_anomalies_injected)

    # NORMAL TRAINING with dataset 1
    lstm = AnomalyDetection(train_vectors=embeddings_normal,
                             train_instance_information_file=instance_information_file_normal,
                             test_vectors=embeddings_anomalies_injected,
                             test_instance_information_file=instance_information_file_anomalies_injected,
                             savemodelpath=lstm_model_save_path,
                             seq_length=seq_len,
                             num_epochs=epochs,
                             n_hidden_units=n_hidden_units,
                             n_layers=n_layers,
                             embeddings_model='bert',
                             train_mode=True,
                             clip=clip,
                             results_dir=cwd + results_dir_experiment,
                             batch_size=batch_size,
                             lines_that_have_anomalies=anomalies_true)

    if not anomaly_only:
        lstm.start_training(no_anomaly)

    f1, precision = lstm.loss_evaluation(no_anomaly)
    print("done.")
    calculate_precision_and_plot(results_dir_experiment, epochs, seq_len, embeddings_model, anomaly_type, anomaly_amount, cwd)
    return f1, precision

if __name__ == '__main__':
    experiment()