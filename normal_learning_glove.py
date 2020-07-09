import subprocess

import matplotlib

from loganaliser.binary import BinaryClassification

#matplotlib.use('Agg')
from loganaliser.vanilla_autoencoder import VanillaAutoEncoder
from settings import settings
from wordembeddings import transform_glove
from wordembeddings.transform_glove import merge_templates
import logparser.Drain.Drain_demo as drain
import wordembeddings.transform_bert as transform_bert
from loganaliser.regression import Regression
from loganaliser.multiclass import Multiclass
from wordembeddings.bert_finetuning import finetune
from shared_functions import calculate_precision_and_plot, get_cosine_distance, inject_anomalies, get_labels_from_corpus, \
    pre_process_log_events, get_top_k_embedding_label_mapping
import os
from wordembeddings.visualisation import write_to_tsv_files_bert_sentences
from shared_functions import get_embeddings


def experiment(epochs=10,
               mode="multiclass",
               anomaly_type='random_lines',
               anomaly_amount=1,
               clip=1.0,
               attention=False,
               prediction_only=False,
               alteration_ratio=0.05,
               anomaly_ratio=0.04,
               option='Normal', seq_len=7, n_layers=1, n_hidden_units=128, batch_size=64, finetuning=False,
               embeddings_model='glove', experiment='x', embeddings_dim=200):
    cwd = os.getcwd() + "/"
    print("############\n STARTING\n Epochs:{}, Mode:{}, Attention:{}, Anomaly Type:{}"
          .format(epochs, mode, attention, anomaly_type))

    no_anomaly = True if anomaly_type == "no_anomaly" else False

    if finetuning:
        results_dir = settings[option]["results_dir"] + "_finetune/"
    else:
        results_dir = settings[option]["results_dir"] + "/"

    results_dir_experiment = "{}_{}_epochs_{}_seq_len_{}_anomaly_type_{}_{}_hidden_{}_layers_{}_clip_{}_experiment_{}/".format(
        results_dir + mode, embeddings_model, epochs, seq_len, anomaly_type, anomaly_amount, n_hidden_units, n_layers, clip, experiment)

    train_ds = settings[option]["raw_normal"]  # path of normal file for training
    test_ds = settings[option]["raw_anomaly"]  # path of file in which anomalies will be injected
    raw_dir = settings[option]["raw_dir"]  # dir in which training and anomaly files are in, for drain
    parsed_dir = settings[option]["parsed_dir"]  # dir where parsed training and pre-anomaly file will be
    embeddings_dir = settings[option]["embeddings_dir"]  # dir for embeddings vectors
    logtype = settings[option]["logtype"]  # logtype for drain parser

    train_instance_information = settings[option]['instance_information_file_normal']
    # for binary
    train_instance_information_injected = settings[option]['instance_information_file_normal'] + \
                                          train_ds + "_" + anomaly_type + "_" + str(anomaly_amount)
    test_instance_information = settings[option]['instance_information_file_anomalies_pre_inject']
    test_instance_information_injected = settings[option]['instance_information_file_anomalies_injected'] + \
                                         test_ds + "_" + anomaly_type + "_" + str(anomaly_amount)

    anomalies_injected_dir = parsed_dir + "anomalies_injected/"
    anomaly_indeces_dir = parsed_dir + "anomalies_injected/anomaly_indeces/"

    # corpus files produced by Drain
    corpus_train = cwd + parsed_dir + train_ds + '_corpus'
    corpus_test = cwd + parsed_dir + test_ds + '_corpus'

    # template files produced by Drain
    templates_normal = cwd + parsed_dir + train_ds + '_templates'
    templates_pre_anomaly = cwd + parsed_dir + test_ds + '_templates'

    # bert vectors as pickle files
    embeddings_train = cwd + embeddings_dir + train_ds + '.pickle'
    embeddings_test = cwd + embeddings_dir + test_ds + '.pickle'
    embeddings_templates_combined = cwd + embeddings_dir + train_ds + test_ds + 'for_vae.pickle'

    embeddingsfile_for_glove = '../' + embeddings_dir + train_ds + '_combined_vectors'
    embeddingsfile_for_transformer = cwd + embeddings_dir + train_ds + '_combined_vectors.txt'

    lstm_model_save_path = cwd + 'loganaliser/saved_models/' + train_ds + "_" + experiment + '_lstm.pth'

    vae_model_save_path = cwd + 'loganaliser/saved_models/' + train_ds + '_vae.pth'

    # take corpus parsed by drain, inject anomalies in this file
    corpus_test_injected = cwd + anomalies_injected_dir + test_ds + "_" + anomaly_type
    corpus_train_injected = cwd + anomalies_injected_dir + train_ds + "_" + anomaly_type # for binary
    train_anomaly_indeces = cwd + results_dir_experiment + "train_anomaly_labels.txt" # for binary
    test_anomaly_indeces = cwd + results_dir_experiment + "test_anomaly_labels.txt"

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(results_dir_experiment, exist_ok=True)
    os.makedirs(parsed_dir, exist_ok=True)
    os.makedirs(embeddings_dir, exist_ok=True)
    os.makedirs(anomalies_injected_dir, exist_ok=True)
    os.makedirs(anomaly_indeces_dir, exist_ok=True)

    ### DRAIN PARSING
    #if not os.path.exists(corpus_train) or not os.path.exists(corpus_test):

    #drain.execute(directory=raw_dir, file=train_ds, output=parsed_dir, logtype=logtype)
    #drain.execute(directory=raw_dir, file=test_ds, output=parsed_dir, logtype=logtype)

    pre_process_log_events(corpus_test, corpus_train, templates_normal, templates_pre_anomaly)



    ### INJECT ANOMALIES in test ds
    if anomaly_type is not "random_lines":
        _, test_ds_liens_before_injection, train_ds_lines_after_injection = \
            inject_anomalies(anomaly_type=anomaly_type,
                             corpus_input=corpus_test,
                             corpus_output=corpus_test_injected,
                             anomaly_indices_output_path=test_anomaly_indeces,
                             instance_information_in=test_instance_information,
                             instance_information_out=test_instance_information_injected,
                             anomaly_amount=anomaly_amount,
                             results_dir=results_dir_experiment,
                             alteration_ratio=alteration_ratio,
                             anomaly_ratio=anomaly_ratio)

        # INJECT ANOMALIES in test ds
        if anomaly_type is not "no_anomaly":
            test_ds_anomaly_lines, _, _ = \
                inject_anomalies(anomaly_type="random_lines",
                                 corpus_input=corpus_test_injected,
                                 corpus_output=corpus_test_injected,
                                 anomaly_indices_output_path=test_anomaly_indeces,
                                 instance_information_in=test_instance_information_injected,
                                 instance_information_out=test_instance_information_injected,
                                 anomaly_amount=anomaly_amount,
                                 results_dir=results_dir_experiment,
                                 alteration_ratio=alteration_ratio,
                                 anomaly_ratio=anomaly_ratio)

    else:
        # INJECT ANOMALIES in test ds
        test_ds_anomaly_lines, _, _ = \
            inject_anomalies(anomaly_type="random_lines",
                             corpus_input=corpus_test,
                             corpus_output=corpus_test_injected,
                             anomaly_indices_output_path=test_anomaly_indeces,
                             instance_information_in=test_instance_information,
                             instance_information_out=test_instance_information_injected,
                             anomaly_amount=anomaly_amount,
                             results_dir=results_dir_experiment,
                             alteration_ratio=alteration_ratio,
                             anomaly_ratio=anomaly_ratio)


    # produce templates out of the corpuses that we have from the anomaly file
    templates_train = list(set(open(corpus_train, 'r').readlines()))
    # merge_templates(templates_normal_1, templates_normal_2, merged_template_path=parsed_dir_1 + "_merged_templates_normal")
    templates_test_anomalies_injected = list(set(open(corpus_test_injected, 'r').readlines()))
    merged_templates = merge_templates(templates_train, templates_test_anomalies_injected, merged_template_path=None)
    merged_templates = list(merged_templates)
    # write merged templates to file so glove can open
    merged_templates_file = templates_normal + "_anomaly_merged"
    with open(merged_templates_file, "w") as f:
        for template in merged_templates:
            f.write(template)

    subprocess.call(['glove-c/word_embeddings.sh',
                     '-c', merged_templates_file,
                     '-s', embeddingsfile_for_glove,
                     '-v', str(embeddings_dim)])

    # if anomaly_type in ["insert_words", "remove_words", "replace_words"]:
    #     get_cosine_distance(test_ds_liens_before_injection, train_ds_lines_after_injection, results_dir_experiment, word_embeddings)

    # transform output of bert into numpy word embedding vectors
    transform_glove.transform(vectorsfile=embeddingsfile_for_transformer, logfile=corpus_train, outputfile=embeddings_train, templates=merged_templates)

    transform_glove.transform(vectorsfile=embeddingsfile_for_transformer, logfile=corpus_test_injected, outputfile=embeddings_test, templates=merged_templates)

    transform_glove.transform(vectorsfile=embeddingsfile_for_transformer, logfile=merged_templates, outputfile=embeddings_templates_combined, templates=merged_templates)

    vae = VanillaAutoEncoder(load_vectors=embeddings_templates_combined,
                             model_save_path=vae_model_save_path,
                             train_mode=True)
    vae.start()

    lstm = Regression(train_vectors=embeddings_train,
                      train_instance_information_file=train_instance_information,
                      test_vectors=embeddings_test,
                      test_instance_information_file=test_instance_information_injected,
                      savemodelpath=lstm_model_save_path,
                      seq_length=seq_len,
                      num_epochs=epochs,
                      n_hidden_units=n_hidden_units,
                      n_layers=n_layers,
                      embeddings_model='bert',
                      no_anomaly=no_anomaly,
                      clip=clip,
                      results_dir=cwd + results_dir_experiment,
                      batch_size=batch_size,
                      lines_that_have_anomalies=test_ds_anomaly_lines,
                      n_input=embeddings_dim,
                      n_features=embeddings_dim,
                      transfer_learning=False,
                      attention=attention,
                      prediction_only=prediction_only,
                      loadautoencodermodel=vae_model_save_path)

    if not prediction_only:
        lstm.start_training()

    f1, precision = lstm.final_prediction()
    calculate_precision_and_plot(results_dir_experiment, cwd, embeddings_model, epochs, seq_len, anomaly_type,
                                 anomaly_amount, n_hidden_units, n_layers, clip, experiment, mode)
    print("done.")
    return f1, precision

if __name__ == '__main__':
    experiment()
