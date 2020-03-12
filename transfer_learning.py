import argparse

import matplotlib

matplotlib.use('Agg')
from settings import settings
from wordembeddings.transform_glove import merge_templates
import logparser.Drain.Drain_demo as drain
import wordembeddings.transform_bert as transform_bert
from loganaliser.main import AnomalyDetection
from wordembeddings.bert_finetuning import finetune
from shared_functions import calculate_precision_and_plot, calculate_anomaly_loss, calculate_normal_loss, \
    get_cosine_distance, inject_anomalies
import os

predicted_labels_of_file_containing_anomalies = "predicted_labels_of_file_containing_anomalies"

# -----------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------INITIALISE PARAMETERS---------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------
cwd = os.getcwd() + "/"
parser = argparse.ArgumentParser()
parser.add_argument('-option', type=str, default='UtahSashoTransfer')
parser.add_argument('-seq_len', type=int, default=7)
parser.add_argument('-reduced', action='store_true')
parser.add_argument('-epochs', type=int, default=100)
parser.add_argument('-hiddenunits', type=int, default=250)
parser.add_argument('-hiddenlayers', type=int, default=4)
parser.add_argument('-transferlearning', action='store_true')
parser.add_argument('-anomaly_only', action='store_true')
parser.add_argument('-corpus_anomaly_inputfile', type=str)
parser.add_argument('-instance_information_file_anomalies', type=str)
parser.add_argument('-bert_model_finetune', type=str, default='bert-base-uncased')
parser.add_argument('-finetune', action='store_true')
parser.add_argument('-anomaly_type', type=str, default='insert_words')
parser.add_argument('-anomaly_amount', type=int, default=8)
args = parser.parse_args()

print("starting {} {}".format(args.anomaly_type, args.anomaly_amount))
option = args.option
results_dir_experiment = "{}_epochs_{}_seq_len:_{}_anomaly_type:{}_{}/"\
    .format(settings[option]["dataset_2"]["results_dir"] + 'bert', args.epochs, args.seq_len, args.anomaly_type, args.anomaly_amount)

normal_1 = settings[option]["dataset_1"]["raw_normal"]
normal_2 = settings[option]["dataset_2"]["raw_normal"]
anomaly_2 = settings[option]["dataset_2"]["raw_anomaly"]

raw_dir_1 = settings[option]["dataset_1"]["raw_dir"]
raw_dir_2 = settings[option]["dataset_2"]["raw_dir"]

parsed_dir_1 = settings[option]["dataset_1"]["parsed_dir"]
parsed_dir_2 = settings[option]["dataset_2"]["parsed_dir"]

results_dir = settings[option]["dataset_2"]["results_dir"]

embeddings_dir_1 = settings[option]["dataset_1"]["embeddings_dir"]
embeddings_dir_2 = settings[option]["dataset_2"]["embeddings_dir"]

logtype_1 = settings[option]["dataset_1"]["logtype"]
logtype_2 = settings[option]["dataset_2"]["logtype"]

instance_information_file_normal_1 = settings[option]["dataset_1"]['instance_information_file_normal']
instance_information_file_normal_2 = settings[option]["dataset_2"]['instance_information_file_normal']
instance_information_file_anomalies_pre_inject_2 = settings[option]["dataset_2"]['instance_information_file_anomalies_pre_inject']
instance_information_file_anomalies_injected_2 = \
    settings[option]["dataset_2"]['instance_information_file_anomalies_injected'] + anomaly_2 + "_" + args.anomaly_type + "_" + str(args.anomaly_amount)

anomalies_injected_dir_2 = parsed_dir_2 + "anomalies_injected/"
anomaly_indeces_dir_2 = parsed_dir_2 + "anomalies_injected/anomaly_indeces/"

# corpus files produced by Drain
corpus_normal_1 = cwd + parsed_dir_1 + normal_1 + '_corpus'
corpus_normal_2 = cwd + parsed_dir_2 + normal_2 + '_corpus'
corpus_pre_anomaly_2 = cwd + parsed_dir_2 + anomaly_2 + '_corpus'

# bert vectors as pickle files
embeddings_normal_1 = cwd + embeddings_dir_1 + normal_1 + '.pickle'
embeddings_normal_2 = cwd + embeddings_dir_2 + normal_2 + '.pickle'
embeddings_anomalies_injected_2 = cwd + embeddings_dir_2 + anomaly_2 + '.pickle'
finetuning_model_dir = "wordembeddings/finetuning-models/" + normal_1

if args.finetune:
    lstm_model_save_path = cwd + 'loganaliser/saved_models/' + normal_1 + '_with_finetune' + '_lstm.pth'
else:
    lstm_model_save_path = cwd + 'loganaliser/saved_models/' + normal_1 + '_lstm.pth'
vae_model_save_path = cwd + 'loganaliser/saved_models/' + normal_1 + '_vae.pth'

# take corpus parsed by drain, inject anomalies in this file
anomaly_injected_corpus_2 = cwd + anomalies_injected_dir_2 + anomaly_2 + "_" + args.anomaly_type
# put here the information about which line is an anomaly from anomaly injection
anomaly_indeces_2 = cwd + results_dir_experiment + "true_anomaly_labels.txt"
# anomaly_indeces_2 = cwd + anomaly_indeces_dir_2 + anomaly_2 + "_" + args.anomaly_type + "_" + args.anomaly_amount  + '_anomaly_indeces.txt'

# create all directories, if they don't exist yet
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
if not os.path.exists(corpus_normal_1) or not os.path.exists(corpus_normal_2) or not os.path.exists(
        corpus_pre_anomaly_2):
    drain.execute(directory=raw_dir_1, file=normal_1, output=parsed_dir_1, logtype=logtype_1)
    drain.execute(directory=raw_dir_2, file=normal_2, output=parsed_dir_2, logtype=logtype_2)
    drain.execute(directory=raw_dir_2, file=anomaly_2, output=parsed_dir_2, logtype=logtype_2)

### INJECT ANOMALIES in dataset 2
anomalies_true_labels, lines_before_alter, lines_after_alter = inject_anomalies(anomaly_type=args.anomaly_type,
                                                                                corpus_input=corpus_pre_anomaly_2,
                                                                                corpus_output=anomaly_injected_corpus_2,
                                                                                anomaly_indices_output_path=anomaly_indeces_2,
                                                                                instance_information_in=instance_information_file_anomalies_pre_inject_2,
                                                                                instance_information_out=instance_information_file_anomalies_injected_2,
                                                                                anomaly_amount=args.anomaly_amount,
                                                                                results_dir=results_dir_experiment)

# produce templates out of the corpuses that we have from the anomaly file
templates_normal_1 = list(set(open(corpus_normal_1, 'r').readlines()))
templates_normal_2 = list(set(open(corpus_normal_2, 'r').readlines()))
# merge_templates(templates_normal_1, templates_normal_2, merged_template_path=parsed_dir_1 + "_merged_templates_normal")
templates_anomalies_injected = list(set(open(anomaly_injected_corpus_2, 'r').readlines()))
merged_templates = merge_templates(templates_normal_1, templates_normal_2, templates_anomalies_injected,
                                   merged_template_path=None)
merged_templates = list(merged_templates)

if args.finetune:
    if not os.path.exists(finetuning_model_dir):
        finetune(templates=templates_normal_1, output_dir=finetuning_model_dir)

bert_vectors, _, _, _ = transform_bert.get_bert_vectors(merged_templates, bert_model=finetuning_model_dir)

if args.anomaly_type in ["insert_words", "remove_words"]:
    get_cosine_distance(lines_before_alter, lines_after_alter, merged_templates, results_dir_experiment, bert_vectors)

# transform output of bert into numpy word embedding vectors
transform_bert.transform(sentence_embeddings=bert_vectors,
                         logfile=corpus_normal_1,
                         templates=merged_templates,
                         outputfile=embeddings_normal_1)

transform_bert.transform(sentence_embeddings=bert_vectors,
                         logfile=corpus_normal_2,
                         templates=merged_templates,
                         outputfile=embeddings_normal_2)

transform_bert.transform(sentence_embeddings=bert_vectors,
                         logfile=anomaly_injected_corpus_2,
                         templates=merged_templates,
                         outputfile=embeddings_anomalies_injected_2)

if not args.anomaly_only:
    # NORMAL TRAINING with dataset 1
    ad_normal = AnomalyDetection(loadautoencodermodel=vae_model_save_path,
                                 loadvectors=embeddings_normal_1,
                                 savemodelpath=lstm_model_save_path,
                                 seq_length=args.seq_len,
                                 num_epochs=args.epochs,
                                 n_hidden_units=args.hiddenunits,
                                 n_layers=args.hiddenlayers,
                                 embeddings_model='bert',
                                 train_mode=True,
                                 instance_information_file=instance_information_file_normal_1)

    ad_normal.start_training()
# FEW SHOT TRAINING with dataset 2
ad_normal_transfer = AnomalyDetection(loadautoencodermodel=vae_model_save_path,
                                      loadvectors=embeddings_normal_2,
                                      savemodelpath=lstm_model_save_path,
                                      seq_length=args.seq_len,
                                      num_epochs=5,
                                      n_hidden_units=args.hiddenunits,
                                      n_layers=args.hiddenlayers,
                                      embeddings_model='bert',
                                      train_mode=True,
                                      instance_information_file=instance_information_file_normal_2,
                                      transfer_learning=True)
if not args.anomaly_only:
    ad_normal_transfer.start_training()

normal_loss_values = calculate_normal_loss(normal_lstm_model=ad_normal_transfer,
                                           results_dir=results_dir_experiment,
                                           values_type='normal_loss_values',
                                           cwd=cwd)
ad_anomaly = AnomalyDetection(loadautoencodermodel=vae_model_save_path,
                              loadvectors=embeddings_anomalies_injected_2,
                              savemodelpath=lstm_model_save_path,
                              seq_length=args.seq_len,
                              num_epochs=args.epochs,
                              n_hidden_units=args.hiddenunits,
                              n_layers=args.hiddenlayers,
                              embeddings_model='bert',
                              instance_information_file=instance_information_file_anomalies_injected_2,
                              anomalies_run=True,
                              results_dir=cwd + results_dir_experiment + 'anomaly_loss_indices')
calculate_anomaly_loss(anomaly_lstm_model=ad_anomaly, results_dir=results_dir_experiment,
                       normal_loss_values=normal_loss_values,
                       anomaly_loss_order=cwd + results_dir_experiment + 'anomaly_loss_indices',
                       anomaly_true_labels=anomalies_true_labels)
print("done.")
calculate_precision_and_plot(results_dir_experiment, args, cwd)
