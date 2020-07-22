import matplotlib

from logparser.anomaly_injector import transfer_train_log

#matplotlib.use('Agg')
from settings import settings
from wordembeddings.transform_glove import merge_templates
import logparser.Drain.Drain_demo as drain
import wordembeddings.transform_bert as transform_bert
from loganaliser.regression import Regression
from loganaliser.multiclass import Multiclass
from shared_functions import calculate_precision_and_plot, get_cosine_distance, inject_anomalies, get_labels_from_corpus, \
    pre_process_log_events, get_nearest_neighbour_embedding_label_mapping
import os
from wordembeddings.visualisation import write_to_tsv_files_bert_sentences
from shared_functions import get_embeddings


def experiment(epochs=60,
               mode="multiclass",
               anomaly_type='insert_words',
               anomaly_amount=1,
               clip=1.0,
               attention=False,
               prediction_only=False,
               alteration_ratio=0.04,
               anomaly_ratio=0.05,
               option='UtahUtahTransfer', seq_len=7, n_layers=1, n_hidden_units=512, batch_size=64, finetuning=False,
               embeddings_model='bert', experiment='x', label_encoder=None):
    cwd = os.getcwd() + "/"
    print("############\n STARTING\n Epochs:{}, Mode:{}, Attention:{}, Anomaly Type:{}"
          .format(epochs, mode, attention, anomaly_type))

    no_anomaly = True if anomaly_type == "no_anomaly" else False

    if finetuning:
        results_dir = settings[option]["dataset_2"]["results_dir"] + "_finetune/"
    else:
        results_dir = settings[option]["dataset_2"]["results_dir"] + "/"

    results_dir_experiment = "{}_{}_epochs_{}_seq_len_{}_anomaly_type_{}_{}_hidden_{}_layers_{}_clip_{}_experiment_{}_alteration_ratio_{}_anomaly_ratio_{}/".format(
        results_dir + mode, embeddings_model, epochs, seq_len, anomaly_type, anomaly_amount, n_hidden_units, n_layers, clip, experiment, alteration_ratio, anomaly_ratio)

    train_ds_1 = settings[option]["dataset_1"]["raw_normal"]
    train_ds_2 = settings[option]["dataset_2"]["raw_normal"]
    test_ds_2 = settings[option]["dataset_2"]["raw_anomaly"]

    raw_dir_1 = settings[option]["dataset_1"]["raw_dir"]
    raw_dir_2 = settings[option]["dataset_2"]["raw_dir"]

    parsed_dir_1 = settings[option]["dataset_1"]["parsed_dir"]
    parsed_dir_2 = settings[option]["dataset_2"]["parsed_dir"]

    embeddings_dir_1 = settings[option]["dataset_1"]["embeddings_dir"]
    embeddings_dir_2 = settings[option]["dataset_2"]["embeddings_dir"]

    logtype_1 = settings[option]["dataset_1"]["logtype"]
    logtype_2 = settings[option]["dataset_2"]["logtype"]

    train_instance_information_1 = settings[option]["dataset_1"]['instance_information_file_normal']
    train_instance_information_2 = settings[option]["dataset_2"]['instance_information_file_normal']

    train_instance_information_injected_1 = settings[option]["dataset_1"]['instance_information_file_normal_after_injections']

    # for binary
    BINARY_train_instance_information_injected_1 = settings[option]["dataset_1"]['instance_information_file_normal'] +  train_ds_1 + "_" + anomaly_type + "_" + str(anomaly_amount)
    BINARY_train_instance_information_injected_2 = settings[option]["dataset_2"]['instance_information_file_normal'] +  train_ds_2 + "_" + anomaly_type + "_" + str(anomaly_amount)

    test_instance_information_2 = settings[option]["dataset_2"]['instance_information_file_anomalies_pre_inject']
    test_instance_information_injected_2 = settings[option]["dataset_2"]['instance_information_file_anomalies_injected'] + test_ds_2 + "_" + anomaly_type + "_" + str(anomaly_amount)


    anomalies_injected_dir_2 = parsed_dir_2 + "anomalies_injected/"
    anomaly_indeces_dir_2 = parsed_dir_2 + "anomalies_injected/anomaly_indeces/"

    # corpus files produced by Drain
    corpus_train_1 = cwd + parsed_dir_1 + train_ds_1 + '_corpus'
    corpus_train_2 = cwd + parsed_dir_2 + train_ds_2 + '_corpus'
    corpus_test_2 = cwd + parsed_dir_2 + test_ds_2 + '_corpus'

    # bert vectors as pickle files
    embeddings_train_1 = cwd + embeddings_dir_1 + train_ds_1 + '.pickle'
    embeddings_train_2 = cwd + embeddings_dir_2 + train_ds_2 + '.pickle'
    embeddings_test_2 = cwd + embeddings_dir_2 + test_ds_2 + '.pickle'

    if finetuning:
        finetuning_model_dir = "wordembeddings/finetuning-models/" + train_ds_1
        lstm_model_save_path = cwd + 'loganaliser/saved_models/' + train_ds_1 + '_with_finetune' + '_lstm.pth'
    else:
        finetuning_model_dir = "bert-base-uncased"
        lstm_model_save_path = cwd + 'loganaliser/saved_models/transfer_' + train_ds_1 + "_" + experiment + '_lstm.pth'

    # take corpus parsed by drain, inject anomalies in this file
    corpus_test_injected = cwd + anomalies_injected_dir_2 + test_ds_2 + "_" + anomaly_type
    BINARY_corpus_train_injected = cwd + anomalies_injected_dir_2 + train_ds_1 + "_" + anomaly_type
    BINARY_train_anomaly_indeces = cwd + results_dir_experiment + "train_anomaly_labels.txt"
    test_anomaly_indeces = cwd + results_dir_experiment + "test_anomaly_labels.txt"

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(results_dir_experiment, exist_ok=True)
    os.makedirs(raw_dir_1, exist_ok=True)
    os.makedirs(raw_dir_2, exist_ok=True)
    os.makedirs(parsed_dir_1, exist_ok=True)
    os.makedirs(parsed_dir_2, exist_ok=True)
    os.makedirs(embeddings_dir_1, exist_ok=True)
    os.makedirs(embeddings_dir_2, exist_ok=True)
    os.makedirs(anomalies_injected_dir_2, exist_ok=True)
    os.makedirs(anomaly_indeces_dir_2, exist_ok=True)

    ### DRAIN PARSING
    #if not os.path.exists(corpus_train_1) or not os.path.exists(corpus_test_2):
    drain.execute(directory=raw_dir_1, file=train_ds_1, output=parsed_dir_1, logtype=logtype_1)
    drain.execute(directory=raw_dir_2, file=train_ds_2, output=parsed_dir_2, logtype=logtype_2)
    drain.execute(directory=raw_dir_2, file=test_ds_2, output=parsed_dir_2, logtype=logtype_2)

    pre_process_log_events(corpus_test_2, corpus_train_1, corpus_train_2)

    # manipulate train ds 1
    if option != "UtahSashoTransfer":
        transfer_train_log(corpus_train_1, corpus_train_1)

    ### INJECT ALTERATIONS in test ds
    if anomaly_type is not "random_lines":
        _, test_ds_lines_before_injection, train_ds_lines_after_injection = \
            inject_anomalies(anomaly_type="shuffle",
                             corpus_input=corpus_train_1,
                             corpus_output=corpus_train_1,
                             anomaly_indices_output_path=test_anomaly_indeces,
                             instance_information_in=train_instance_information_1,
                             instance_information_out=train_instance_information_injected_1,
                             anomaly_amount=anomaly_amount,
                             results_dir=results_dir_experiment,
                             alteration_ratio=alteration_ratio,
                             anomaly_ratio=anomaly_ratio)

        _, test_ds_lines_before_injection, train_ds_lines_after_injection = \
                inject_anomalies(anomaly_type="duplicate_lines",
                                 corpus_input=corpus_train_1,
                                 corpus_output=corpus_train_1,
                                 anomaly_indices_output_path=test_anomaly_indeces,
                                 instance_information_in=train_instance_information_injected_1,
                                 instance_information_out=train_instance_information_injected_1,
                                 anomaly_amount=anomaly_amount,
                                 results_dir=results_dir_experiment,
                                 alteration_ratio=alteration_ratio,
                                 anomaly_ratio=anomaly_ratio)

        _, _, train_ds_lines_after_injection = \
                inject_anomalies(anomaly_type="delete_lines",
                                 corpus_input=corpus_train_1,
                                 corpus_output=corpus_train_1,
                                 anomaly_indices_output_path=test_anomaly_indeces,
                                 instance_information_in=train_instance_information_injected_1,
                                 instance_information_out=train_instance_information_injected_1,
                                 anomaly_amount=anomaly_amount,
                                 results_dir=results_dir_experiment,
                                 alteration_ratio=alteration_ratio,
                                 anomaly_ratio=anomaly_ratio)

        _, _, train_ds_lines_after_injection = \
            inject_anomalies(anomaly_type="insert_words",
                             corpus_input=corpus_train_1,
                             corpus_output=corpus_train_1,
                             anomaly_indices_output_path=test_anomaly_indeces,
                             instance_information_in=train_instance_information_injected_1,
                             instance_information_out=train_instance_information_injected_1,
                             anomaly_amount=anomaly_amount,
                             results_dir=results_dir_experiment,
                             alteration_ratio=alteration_ratio,
                             anomaly_ratio=anomaly_ratio)

        _, _, train_ds_lines_after_injection = \
            inject_anomalies(anomaly_type="remove_words",
                             corpus_input=corpus_train_1,
                             corpus_output=corpus_train_1,
                             anomaly_indices_output_path=test_anomaly_indeces,
                             instance_information_in=train_instance_information_injected_1,
                             instance_information_out=train_instance_information_injected_1,
                             anomaly_amount=anomaly_amount,
                             results_dir=results_dir_experiment,
                             alteration_ratio=alteration_ratio,
                             anomaly_ratio=anomaly_ratio)

        _, _, train_ds_lines_after_injection = \
            inject_anomalies(anomaly_type="replace_words",
                             corpus_input=corpus_train_1,
                             corpus_output=corpus_train_1,
                             anomaly_indices_output_path=test_anomaly_indeces,
                             instance_information_in=train_instance_information_injected_1,
                             instance_information_out=train_instance_information_injected_1,
                             anomaly_amount=anomaly_amount,
                             results_dir=results_dir_experiment,
                             alteration_ratio=alteration_ratio,
                             anomaly_ratio=anomaly_ratio)

        if anomaly_type != "reverse_order":
            # INJECT ANOMALIES in test ds
            test_ds_anomaly_lines, _, _ = \
                    inject_anomalies(anomaly_type="random_lines",
                                     corpus_input=corpus_test_2,
                                     corpus_output=corpus_test_injected,
                                     anomaly_indices_output_path=test_anomaly_indeces,
                                     instance_information_in=test_instance_information_2,
                                     instance_information_out=test_instance_information_injected_2,
                                     anomaly_amount=anomaly_amount,
                                     results_dir=results_dir_experiment,
                                     alteration_ratio=alteration_ratio,
                                     anomaly_ratio=anomaly_ratio)

        else:
            # INJECT ANOMALIES in test ds
            test_ds_anomaly_lines, _, _ = \
                inject_anomalies(anomaly_type="reverse_order",
                                 corpus_input=corpus_test_2,
                                 corpus_output=corpus_test_injected,
                                 anomaly_indices_output_path=test_anomaly_indeces,
                                 instance_information_in=test_instance_information_2,
                                 instance_information_out=test_instance_information_injected_2,
                                 anomaly_amount=anomaly_amount,
                                 results_dir=results_dir_experiment,
                                 alteration_ratio=alteration_ratio,
                                 anomaly_ratio=anomaly_ratio)

    else:
        # INJECT ANOMALIES in test ds
        test_ds_anomaly_lines, _, _ = \
            inject_anomalies(anomaly_type="random_lines",
                             corpus_input=corpus_test_2,
                             corpus_output=corpus_test_injected,
                             anomaly_indices_output_path=test_anomaly_indeces,
                             instance_information_in=test_instance_information_2,
                             instance_information_out=test_instance_information_injected_2,
                             anomaly_amount=anomaly_amount,
                             results_dir=results_dir_experiment,
                             alteration_ratio=alteration_ratio,
                             anomaly_ratio=anomaly_ratio)

    ### if in binary mode, inject anomalies also in train ds
    if mode == "binary":
        train_ds_anomaly_lines, train_ds_lines_before_injection, train_ds_lines_after_injection = \
            inject_anomalies(
                anomaly_type=anomaly_type,
                corpus_input=corpus_train_1,
                corpus_output=BINARY_corpus_train_injected,
                anomaly_indices_output_path=BINARY_train_anomaly_indeces,
                instance_information_in=train_instance_information_1,
                instance_information_out=BINARY_train_instance_information_injected_1,
                anomaly_amount=anomaly_amount,
                results_dir=results_dir_experiment)

        train_ds_2_anomaly_lines, train_ds_2_lines_before_injection, train_ds_2_lines_after_injection = \
            inject_anomalies(
                anomaly_type=anomaly_type,
                corpus_input=corpus_train_2,
                corpus_output=BINARY_corpus_train_injected,
                anomaly_indices_output_path=BINARY_train_anomaly_indeces,
                instance_information_in=train_instance_information_2,
                instance_information_out=BINARY_train_instance_information_injected_2,
                anomaly_amount=anomaly_amount,
                results_dir=results_dir_experiment)


    # produce templates out of the corpuses that we have from the anomaly file
    templates_train_1 = list(set(open(corpus_train_1, 'r').readlines()))
    templates_train_2 = list(set(open(corpus_train_2, 'r').readlines()))
    # merge_templates(templates_normal_1, templates_normal_2, merged_template_path=parsed_dir_1 + "_merged_templates_normal")
    templates_test_anomalies_injected = list(set(open(corpus_test_injected, 'r').readlines()))
    merged_templates = merge_templates(templates_train_1, templates_train_2, templates_test_anomalies_injected, merged_template_path=None)
    merged_templates = list(merged_templates)

    # if finetuning:
    #     if not os.path.exists(finetuning_model_dir):
    #         finetune(templates=templates_train_1, output_dir=finetuning_model_dir)

    sentence_to_embeddings_mapping = get_embeddings(embeddings_model, merged_templates)

    write_to_tsv_files_bert_sentences(word_embeddings=sentence_to_embeddings_mapping,
                                      tsv_file_vectors=results_dir_experiment + "visualisation/vectors.tsv",
                                      tsv_file_sentences=results_dir_experiment + "visualisation/sentences.tsv")

    embeddings_dim = list(sentence_to_embeddings_mapping.values())[0].size()[0]

    # if anomaly_type in ["insert_words", "remove_words", "replace_words"]:
    #     get_cosine_distance(test_ds_lines_before_injection, train_ds_lines_after_injection, results_dir_experiment, sentence_to_embeddings_mapping)

    # transform output of bert into numpy word embedding vectors
    transform_bert.transform(sentence_embeddings=sentence_to_embeddings_mapping, logfile=corpus_train_1, outputfile=embeddings_train_1)

    transform_bert.transform(sentence_embeddings=sentence_to_embeddings_mapping, logfile=corpus_train_2, outputfile=embeddings_train_2)

    transform_bert.transform(sentence_embeddings=sentence_to_embeddings_mapping, logfile=corpus_test_injected, outputfile=embeddings_test_2)

    ############################
    ###### MULTICLASS
    ############################
    if mode == "multiclass":
        target_normal_labels, n_classes, normal_label_embeddings_map, _ = get_labels_from_corpus(normal_corpus=open(corpus_train_1, 'r').readlines(),
                                                                                                 encoder_path=label_encoder,
                                                                                                 templates=templates_train_1,
                                                                                                 embeddings=sentence_to_embeddings_mapping)

        sentences_ds2_label_ds1_mapping = get_nearest_neighbour_embedding_label_mapping(sentence_to_embeddings_mapping,
                                                                                        normal_label_embeddings_map,
                                                                                        templates_train_2 + templates_test_anomalies_injected)

        # Use sentence_label_mapping to map every sentence from corpus to labels
        ds_2_target_labels = [sentences_ds2_label_ds1_mapping.get(sentence) for sentence in open(corpus_train_2, 'r').readlines()]

        lstm_ds_1 = Multiclass(n_features=n_classes,
                               n_input=embeddings_dim,
                               target_labels=target_normal_labels,
                               train_vectors=embeddings_train_1,
                               train_instance_information_file=train_instance_information_injected_1,
                               test_vectors=None,
                               test_instance_information_file=None,
                               savemodelpath=lstm_model_save_path,
                               seq_length=seq_len,
                               num_epochs=epochs,
                               no_anomaly=no_anomaly,
                               results_dir=cwd + results_dir_experiment,
                               embeddings_model='bert',
                               n_layers=n_layers,
                               n_hidden_units=n_hidden_units,
                               batch_size=batch_size,
                               clip=clip,
                               normal_label_embeddings_map=normal_label_embeddings_map,
                               lines_that_have_anomalies=test_ds_anomaly_lines,
                               corpus_of_log_containing_anomalies=corpus_test_injected,
                               transfer_learning=False,
                               attention=attention,
                               prediction_only=prediction_only,
                               transfer_learning_initial_training=True,
                               mode=mode,
                               sentence_to_embeddings_mapping=sentence_to_embeddings_mapping)

        if not prediction_only:
            lstm_ds_1.start_training()

        lstm_ds_2 = Multiclass(n_features=n_classes,
                               n_input=embeddings_dim,
                               target_labels=ds_2_target_labels,
                               train_vectors=embeddings_train_2,
                               train_instance_information_file=train_instance_information_2,
                               test_vectors=embeddings_test_2,
                               test_instance_information_file=test_instance_information_injected_2,
                               savemodelpath=lstm_model_save_path,
                               seq_length=seq_len,
                               num_epochs=5,
                               no_anomaly=no_anomaly,
                               results_dir=cwd + results_dir_experiment,
                               embeddings_model='bert',
                               n_layers=n_layers,
                               n_hidden_units=n_hidden_units,
                               batch_size=batch_size,
                               clip=clip,
                               normal_label_embeddings_map=normal_label_embeddings_map,
                               lines_that_have_anomalies=test_ds_anomaly_lines,
                               corpus_of_log_containing_anomalies=corpus_test_injected,
                               transfer_learning=True,
                               attention=attention,
                               prediction_only=prediction_only,
                               mode=mode,
                               sentence_to_embeddings_mapping=sentence_to_embeddings_mapping)

        if not prediction_only:
            lstm_ds_2.start_training()


    ############################
    ###### REGRESSION
    ############################
    elif mode == "regression":
        lstm_ds_1 = Regression(train_vectors=embeddings_train_1,
                               train_instance_information_file=train_instance_information_injected_1,
                               test_vectors=None,
                               test_instance_information_file=None,
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
                               mode=mode,
                               transfer_learning_initial_training=True
                               )

        if not prediction_only:
            lstm_ds_1.start_training()

        lstm_ds_2 = Regression(train_vectors=embeddings_train_2,
                               train_instance_information_file=train_instance_information_2,
                               test_vectors=embeddings_test_2,
                               test_instance_information_file=test_instance_information_injected_2,
                               savemodelpath=lstm_model_save_path,
                               seq_length=seq_len,
                               num_epochs=5,
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
                               transfer_learning=True,
                               attention=attention,
                               prediction_only=prediction_only,
                               mode=mode)

        if not prediction_only:
            lstm_ds_2.start_training()




    f1, precision, recall = lstm_ds_2.final_prediction()
    calculate_precision_and_plot(results_dir_experiment, cwd, embeddings_model, epochs, seq_len, anomaly_type,
                                 anomaly_amount, n_hidden_units, n_layers, clip, experiment, mode)
    print("done.")
    return f1, precision, recall

if __name__ == '__main__':
    experiment()
