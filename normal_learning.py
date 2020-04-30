import argparse

import matplotlib

matplotlib.use('Agg')
from settings import settings
from wordembeddings.transform_glove import merge_templates
import logparser.Drain.Drain_demo as drain
import wordembeddings.transform_bert as transform_bert
from loganaliser.main import AnomalyDetection
from wordembeddings.bert_finetuning import finetune
from shared_functions import calculate_precision_and_plot, determine_anomalies, \
    get_cosine_distance, inject_anomalies, get_embeddings, get_labels_from_corpus, pre_process_log_events
import os
from wordembeddings.visualisation import write_to_tsv_files_bert_sentences


predicted_labels_of_file_containing_anomalies = "predicted_labels_of_file_containing_anomalies"

# ---------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------INITIALISE PARAMETERS-------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------

def experiment(option='Normal', seq_len=7, n_layers=1, n_hidden_units=128, batch_size=64, clip=1.22, epochs=100,
               anomaly_only=False, finetuning=False, anomaly_type='random_lines', anomaly_amount=1, embeddings_model='bert',
               label_encoder=None, experiment='default'):

    cwd = os.getcwd() + "/"
    print("starting {} {}".format(anomaly_type, anomaly_amount))

    if finetuning:
        results_dir = settings[option]["results_dir"] + "_finetune/"
    else:
        results_dir = settings[option]["results_dir"] + "/"

    results_dir_experiment = "{}_epochs_{}_seq_len_{}_anomaly_type_{}_{}_hidden_{}_layers_{}_clip_{}_experiment: {}/".format(
        results_dir + embeddings_model, epochs, seq_len, anomaly_type, anomaly_amount, n_hidden_units, n_layers, clip, experiment)

    train_ds = settings[option]["raw_normal"]  # path of normal file for training
    test_ds = settings[option]["raw_anomaly"]  # path of file in which anomalies will be injected
    raw_dir = settings[option]["raw_dir"]  # dir in which training and anomaly files are in, for drain
    parsed_dir = settings[option]["parsed_dir"]  # dir where parsed training and pre-anomaly file will be
    embeddings_dir = settings[option]["embeddings_dir"]  # dir for embeddings vectors
    logtype = settings[option]["logtype"]  # logtype for drain parser
    train_instance_information_clean = settings[option]['instance_information_file_normal']
    train_instance_information_injected = settings[option]['instance_information_file_normal'] + \
                                                   train_ds + "_" + anomaly_type + "_" + str(anomaly_amount)

    test_instance_information_clean = settings[option]['instance_information_file_anomalies_pre_inject']
    test_instance_information_injected = settings[option]['instance_information_file_anomalies_injected'] + \
                                                   test_ds + "_" + anomaly_type + "_" + str(anomaly_amount)

    anomalies_injected_dir = parsed_dir + "anomalies_injected/"
    anomaly_indeces_dir = parsed_dir + "anomalies_injected/anomaly_indeces/"
    # corpus files produced by Drain
    corpus_train = cwd + parsed_dir + train_ds + '_corpus'
    corpus_test = cwd + parsed_dir + test_ds + '_corpus'
    # template files produced by Drain
    templates_train = cwd + parsed_dir + train_ds + '_templates'
    templates_test = cwd + parsed_dir + test_ds + '_templates'
    # bert vectors as pickle files
    embeddings_train = cwd + embeddings_dir + train_ds + '.pickle'
    embeddings_test = cwd + embeddings_dir + test_ds + '.pickle'

    if finetuning:
        finetuning_model_dir = "wordembeddings/finetuning-models/" + train_ds
        lstm_model_save_path = cwd + 'loganaliser/saved_models/' + train_ds + '_with_finetune' + '_lstm.pth'
    else:
        finetuning_model_dir = "bert-base-uncased"
        lstm_model_save_path = cwd + 'loganaliser/saved_models/' + train_ds + "_" + experiment + '_lstm.pth'

    # take corpus parsed by drain, inject anomalies in this file
    corpus_test_injected = cwd + anomalies_injected_dir + test_ds + "_" + anomaly_type
    corpus_train_injected = cwd + anomalies_injected_dir + train_ds + "_" + anomaly_type
    # put here the information about which line is an anomaly from anomaly injection
    train_anomaly_indeces = cwd + results_dir_experiment + "train_anomaly_labels.txt"
    test_anomaly_indeces = cwd + results_dir_experiment + "test_anomaly_labels.txt"
    # anomaly_indeces_2 = cwd + anomaly_indeces_dir_2 + anomaly_2 + "_" + anomaly_type + "_" + anomaly_amount  + '_anomaly_indeces.txt'

    # create all directories, if they don't exist yet
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(results_dir_experiment, exist_ok=True)
    os.makedirs(parsed_dir, exist_ok=True)
    os.makedirs(embeddings_dir, exist_ok=True)
    os.makedirs(anomalies_injected_dir, exist_ok=True)
    os.makedirs(anomaly_indeces_dir, exist_ok=True)

    ### DRAIN PARSING
    if not os.path.exists(corpus_train) or not os.path.exists(corpus_test):
        drain.execute(directory=raw_dir, file=train_ds, output=parsed_dir, logtype=logtype)
        drain.execute(directory=raw_dir, file=test_ds, output=parsed_dir, logtype=logtype)

    pre_process_log_events(corpus_test, corpus_train, templates_train, templates_test)

    ### INJECT ANOMALIES in train dataset
    train_ds_anomaly_lines, train_ds_lines_before_injection, train_ds_lines_after_injection = \
        inject_anomalies(anomaly_type=anomaly_type,
                         corpus_input=corpus_train,
                         corpus_output=corpus_train_injected,
                         anomaly_indices_output_path=train_anomaly_indeces,
                         instance_information_in=train_instance_information_clean,
                         instance_information_out=train_instance_information_injected,
                         anomaly_amount=anomaly_amount,
                         results_dir=results_dir_experiment)

    ### INJECT ANOMALIES in test dataset
    test_ds_anomaly_lines, test_ds_lines_before_injection, test_ds_lines_after_injection = \
        inject_anomalies(anomaly_type=anomaly_type,
                         corpus_input=corpus_test,
                         corpus_output=corpus_test_injected,
                         anomaly_indices_output_path=test_anomaly_indeces,
                         instance_information_in=test_instance_information_clean,
                         instance_information_out=test_instance_information_injected,
                         anomaly_amount=anomaly_amount,
                         results_dir=results_dir_experiment)

    # produce templates out of the corpuses that we have from the anomaly file
    templates_train = list(set(open(corpus_train, 'r').readlines()))
    templates_test = list(set(open(corpus_test_injected, 'r').readlines()))
    merged_templates = list(merge_templates(templates_train, templates_test, merged_template_path=None))

    if finetuning:
        if not os.path.exists(finetuning_model_dir):
            finetune(templates=templates_train, output_dir=finetuning_model_dir)

    word_embeddings = get_embeddings(embeddings_model, merged_templates)

    write_to_tsv_files_bert_sentences(word_embeddings=word_embeddings,
                                      tsv_file_vectors=results_dir_experiment + "visualisation/vectors.tsv",
                                      tsv_file_sentences=results_dir_experiment + "visualisation/sentences.tsv")

    if anomaly_type in ["insert_words", "remove_words", "replace_words"]:
        get_cosine_distance(test_ds_lines_before_injection, test_ds_lines_after_injection, results_dir_experiment, word_embeddings)

    # transform output of bert into numpy word embedding vectors
    transform_bert.transform(sentence_embeddings=word_embeddings, logfile=corpus_train_injected, outputfile=embeddings_train)

    transform_bert.transform(sentence_embeddings=word_embeddings, logfile=corpus_test_injected, outputfile=embeddings_test)

    target_normal_labels, n_classes, normal_label_embeddings_map, _ = \
        get_labels_from_corpus(normal_corpus=open(corpus_train, 'r').readlines(), encoder_path=label_encoder,
                               templates=templates_train, embeddings=word_embeddings)

    train_model = AnomalyDetection(n_classes=n_classes,
                                   target_labels=target_normal_labels,
                                   loadvectors=embeddings_train,
                                   savemodelpath=lstm_model_save_path,
                                   seq_length=seq_len,
                                   num_epochs=epochs,
                                   embeddings_model='bert',
                                   train_mode=True,
                                   instance_information_file=train_instance_information_injected,
                                   results_dir=cwd + results_dir_experiment,
                                   n_layers=n_layers,
                                   n_hidden_units=n_hidden_units,
                                   batch_size=batch_size,
                                   clip=clip,
                                   anomaly_lines=train_ds_anomaly_lines)

    if not anomaly_only:
        train_model.start_training()

    test_model = AnomalyDetection(n_classes=n_classes,
                                  loadvectors=embeddings_test,
                                  target_labels=target_normal_labels, #TODO: eigentlich sollte für anomaly hier nichts geladen werden, da es eh nicht benutzt wird
                                  savemodelpath=lstm_model_save_path,
                                  seq_length=seq_len,
                                  num_epochs=epochs,
                                  embeddings_model='bert',
                                  instance_information_file=test_instance_information_injected,
                                  anomalies_run=True,
                                  results_dir=cwd + results_dir_experiment,
                                  n_layers=n_layers,
                                  n_hidden_units=n_hidden_units,
                                  batch_size=batch_size,
                                  clip=clip,
                                  anomaly_lines=test_ds_anomaly_lines)

    f1_score, precision = determine_anomalies(anomaly_lstm_model=test_model, results_dir=results_dir_experiment,
                                   order_of_values_of_file_containing_anomalies=cwd + results_dir_experiment + 'anomaly_label_indices',
                                   lines_that_have_anomalies=test_ds_anomaly_lines)
    print("done.")
    calculate_precision_and_plot(results_dir_experiment, embeddings_model, epochs, seq_len, anomaly_type, anomaly_amount, cwd)
    return f1_score, precision

if __name__ == '__main__':
    experiment()