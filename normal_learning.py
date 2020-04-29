import argparse

import matplotlib

matplotlib.use('Agg')
from settings import settings
from wordembeddings.transform_glove import merge_templates
import logparser.Drain.Drain_demo as drain
import wordembeddings.transform_bert as transform_bert
from loganaliser.main import AnomalyDetection
from wordembeddings.bert_finetuning import finetune
from shared_functions import calculate_precision_and_plot, \
    get_cosine_distance, inject_anomalies, get_embeddings, get_labels_from_corpus, pre_process_log_events, \
    get_top_k_embedding_label_mapping
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

    results_dir_experiment = "{}_epochs_{}_seq_len:_{}_anomaly_type:{}_{}_experiment: {}/".format(
        results_dir + embeddings_model, epochs, seq_len, anomaly_type, anomaly_amount, experiment)

    normal = settings[option]["raw_normal"]  # path of normal file for training
    anomaly = settings[option]["raw_anomaly"]  # path of file in which anomalies will be injected
    raw_dir = settings[option]["raw_dir"]  # dir in which training and anomaly files are in, for drain
    parsed_dir = settings[option]["parsed_dir"]  # dir where parsed training and pre-anomaly file will be
    embeddings_dir = settings[option]["embeddings_dir"]  # dir for embeddings vectors
    logtype = settings[option]["logtype"]  # logtype for drain parser
    instance_information_file_normal = settings[option]['instance_information_file_normal']
    instance_information_file_anomalies_pre_inject = settings[option][
        'instance_information_file_anomalies_pre_inject']
    instance_information_file_anomalies_injected = settings[option]['instance_information_file_anomalies_injected'] + \
                                                   anomaly + "_" + anomaly_type + "_" + str(anomaly_amount)

    anomalies_injected_dir = parsed_dir + "anomalies_injected/"
    anomaly_indeces_dir = parsed_dir + "anomalies_injected/anomaly_indeces/"
    # corpus files produced by Drain
    corpus_normal = cwd + parsed_dir + normal + '_corpus'
    corpus_pre_anomaly = cwd + parsed_dir + anomaly + '_corpus'
    # template files produced by Drain
    templates_normal = cwd + parsed_dir + normal + '_templates'
    templates_pre_anomaly = cwd + parsed_dir + anomaly + '_templates'
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
    os.makedirs(parsed_dir, exist_ok=True)
    os.makedirs(embeddings_dir, exist_ok=True)
    os.makedirs(anomalies_injected_dir, exist_ok=True)
    os.makedirs(anomaly_indeces_dir, exist_ok=True)

    ### DRAIN PARSING
    if not os.path.exists(corpus_normal) or not os.path.exists(corpus_pre_anomaly):
        drain.execute(directory=raw_dir, file=normal, output=parsed_dir, logtype=logtype)
        drain.execute(directory=raw_dir, file=anomaly, output=parsed_dir, logtype=logtype)

    pre_process_log_events(corpus_pre_anomaly, corpus_normal, templates_normal, templates_pre_anomaly)

    ### INJECT ANOMALIES in dataset 2
    anomaly_lines, lines_before_alter, lines_after_alter = \
        inject_anomalies(anomaly_type=anomaly_type,
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
    merged_templates = merge_templates(templates_normal, templates_anomalies_injected, merged_template_path=None)
    merged_templates = list(merged_templates)

    if finetuning:
        if not os.path.exists(finetuning_model_dir):
            finetune(templates=templates_normal, output_dir=finetuning_model_dir)

    word_embeddings = get_embeddings(embeddings_model, merged_templates)

    write_to_tsv_files_bert_sentences(word_embeddings=word_embeddings,
                                      tsv_file_vectors=results_dir_experiment + "visualisation/vectors.tsv",
                                      tsv_file_sentences=results_dir_experiment + "visualisation/sentences.tsv")

    if anomaly_type in ["insert_words", "remove_words", "replace_words"]:
        get_cosine_distance(lines_before_alter, lines_after_alter, results_dir_experiment, word_embeddings)

    # transform output of bert into numpy word embedding vectors
    transform_bert.transform(sentence_embeddings=word_embeddings, logfile=corpus_normal, outputfile=embeddings_normal)

    transform_bert.transform(sentence_embeddings=word_embeddings, logfile=anomaly_injected_corpus, outputfile=embeddings_anomalies_injected)

    target_normal_labels, n_classes, normal_label_embeddings_map, _ = \
        get_labels_from_corpus(normal_corpus=open(corpus_normal, 'r').readlines(),
                               encoder_path=label_encoder,
                               templates=templates_normal,
                               embeddings=word_embeddings)

    top_k_label_mapping = get_top_k_embedding_label_mapping(set_embeddings_of_log_containing_anomalies=word_embeddings,
                                                            normal_label_embedding_mapping=normal_label_embeddings_map)

    lstm = AnomalyDetection(n_classes=n_classes,
                                target_labels=target_normal_labels,
                                train_vectors=embeddings_normal,
                                train_instance_information_file=instance_information_file_normal,
                                test_vectors=embeddings_anomalies_injected,
                                test_instance_information_file=instance_information_file_anomalies_injected,
                                savemodelpath=lstm_model_save_path,
                                seq_length=seq_len,
                                num_epochs=epochs,
                                train_mode=True,
                                results_dir=cwd + results_dir_experiment,
                                n_layers=n_layers,
                                n_hidden_units=n_hidden_units,
                                batch_size=batch_size,
                                clip=clip,
                                top_k_label_mapping=top_k_label_mapping,
                                lines_that_have_anomalies=anomaly_lines,
                                corpus_of_log_containing_anomalies=anomaly_injected_corpus)

    if not anomaly_only:
        lstm.start_training()

    f1_score, precision = lstm.calc_labels()
    print("done.")
    calculate_precision_and_plot(results_dir_experiment, embeddings_model, epochs, seq_len, anomaly_type, anomaly_amount, cwd)
    return f1_score, precision

if __name__ == '__main__':
    experiment()