import datetime

import numpy as np
import torch
from keras.preprocessing.sequence import pad_sequences
from pytorch_pretrained_bert import BertModel
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn import MSELoss
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertPreTrainedModel, BertModel, AdamW
from transformers import BertTokenizer as transformers_BertTokenizer
from transformers import get_linear_schedule_with_warmup
import time

BERT_MAX_TOKEN_LEN = 512


def format_time(elapsed):
    """
    Takes a time in seconds and returns a string hh:mm:ss
    """
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def get_bert_vectors_for_fine_tuning_task(
        templates_location='../data/openstack/sasho/parsed/logs_aggregated_full.csv_templates') -> (list, list, int):
    """

    :param templates_location:
    :return: 3-tuple containing: - list: padded_input_ids, containing the sentences which were converted into ids
                                         and are padded
                                 - list: attention_masks, containing index 1 for token ids and 0 for padding 0
                                 - int:  number_of_concat_sent, how many sentences were concatenated
    """
    temp_max_concat_sent_len = 0
    number_of_concat_sent = 1

    tokenizer = transformers_BertTokenizer.from_pretrained('bert-base-uncased')
    sentences = open(templates_location, 'r').readlines()

    # input_ids = None
    # target_ids = None
    # # find the maximum number of possible concatenation of sentences with which we can stay below the maximum threshold
    # # of BERT_MAX_TOKEN_LEN
    # while temp_max_concat_sent_len < BERT_MAX_TOKEN_LEN:
    #     input_ids_temp = []
    #     target_ids_temp = []
    #     for i in range(0, len(sentences) - number_of_concat_sent):
    #         concatenated_sentence = ""
    #         for j in range(0, number_of_concat_sent):
    #             concatenated_sentence += sentences[i + j]
    #             # append space after every sentence, but not after the last one
    #             if j != number_of_concat_sent - 1:
    #                 concatenated_sentence += " "
    #             # append target sentence
    #             if j == number_of_concat_sent - 1:
    #                 target_sentence = sentences[j + i + 1]
    #         encoded_concatenated_sentence = tokenizer.encode(concatenated_sentence, add_special_tokens=True)
    #         input_ids_temp.append(encoded_concatenated_sentence)
    #         assert target_sentence is not None
    #         encoded_target_sentence = tokenizer.encode(target_sentence, add_special_tokens=True)
    #         target_ids_temp.append(encoded_target_sentence)
    #
    #     temp_max_concat_sent_len = max([len(sen) for sen in input_ids_temp])
    #     if temp_max_concat_sent_len < BERT_MAX_TOKEN_LEN:
    #         input_ids = input_ids_temp
    #         target_ids = target_ids_temp
    #         number_of_concat_sent += 1
    #         max_concat_sent_len = temp_max_concat_sent_len

    input_ids = []
    target_ids = []
    seq_len = 7

    for i in range(0, len(sentences) - seq_len):
        input_sentences = sentences[i:i+seq_len]
        target_sentence = sentences[i+seq_len].strip('\n')
        concat_sentences = ""
        for j, sent in enumerate(input_sentences):
            concat_sentences += sent
            if j != seq_len - 1:
                concat_sentences += " "
        encoded_concatenated_sentence = tokenizer.encode(concat_sentences, add_special_tokens=True)
        input_ids.append(encoded_concatenated_sentence)
        assert target_sentence is not None
        encoded_target_sentence = tokenizer.encode(target_sentence, add_special_tokens=True)
        target_ids.append(encoded_target_sentence)

    max_concat_sent_len = max([len(sen) for sen in input_ids])

    # for i in range(0, len(sentences) - seq_len):
    #     concatenated_sentence = ""
    #     for j in range(0, seq_len):
    #         concatenated_sentence += sentences[i + j]
    #         # append space after every sentence, but not after the last one
    #         if j != seq_len - 1:
    #             concatenated_sentence += " "
    #         # append target sentence
    #         if j == number_of_concat_sent - 1:
    #             target_sentence = sentences[j + i + 1]
    #         encoded_concatenated_sentence = tokenizer.encode(concatenated_sentence, add_special_tokens=True)
    #         input_ids.append(encoded_concatenated_sentence)
    #         assert target_sentence is not None
    #         encoded_target_sentence = tokenizer.encode(target_sentence, add_special_tokens=True)
    #         target_ids.append(encoded_target_sentence)

    print("number_of_concat_sent: {}".format(number_of_concat_sent))
    assert input_ids is not None, "The longest sentence already produced" \
                                  "a tokenised sentence that's longer than {}".format(BERT_MAX_TOKEN_LEN)

    padded_input_ids = pad_sequences(input_ids, maxlen=max_concat_sent_len, dtype="long",
                                     value=0, truncating="post", padding="post")

    max_target_length = max([len(sen) for sen in target_ids])

    padded_target_ids = pad_sequences(target_ids, maxlen=max_target_length, dtype="long",
                                      value=0, truncating="post", padding="post")

    attention_masks = []

    for sent in padded_input_ids:
        # Create the attention mask.
        #   - If a token ID is 0, then it's padding, set the mask to 0.
        #   - If a token ID is > 0, then it's a real token, set the mask to 1.
        att_mask = [int(token_id > 0) for token_id in sent]

        # Store the attention mask for this sentence.
        attention_masks.append(att_mask)

    return padded_input_ids, padded_target_ids, attention_masks, seq_len, max_target_length


class BertNSPHead(nn.Module):
    def __init__(self, config):
        super(BertNSPHead, self).__init__()
        # TODO: this has been copied from modeling_bert.BertOnlyNSPHead, where it takes one sentence, and predicts
        #  the next one, i.e. output of the Linear layer is 2, we change it here according to seq_len + 1
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, pooled_output):
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score


class BertForNextSequenceClassification(BertPreTrainedModel):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the sequence classification/regression loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.
            If ``config.num_labels == 1`` a regression loss is computed (Mean-Square loss),
            If ``config.num_labels > 1`` a classification loss is computed (Cross-Entropy).

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification (or regression if config.num_labels==1) loss.
        **logits**: ``torch.FloatTensor`` of shape ``(batch_size, config.num_labels)``
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, logits = outputs[:2]

    """

    def __init__(self, config):
        super(BertForNextSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)

        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None, labels=None):
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        loss_fct = MSELoss()
        loss = loss_fct(logits.view(-1), labels.view(-1).float())
        outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pad_input_ids, trgt_ids, attent_masks, no_of_concat_sents, max_target_length = get_bert_vectors_for_fine_tuning_task()

# Use 90% for training and 10% for validation.
train_inputs, validation_inputs, train_targets, validation_targets = train_test_split(pad_input_ids, trgt_ids,
                                                                                      test_size=0.1, shuffle=False)
# Do the same for the masks.
train_masks, validation_masks, _, _ = train_test_split(attent_masks, trgt_ids, test_size=0.1, shuffle=False)

train_inputs = torch.tensor(train_inputs)
validation_inputs = torch.tensor(validation_inputs)

train_targets = torch.tensor(train_targets)
validation_targets = torch.tensor(validation_targets)

train_masks = torch.tensor(train_masks)
validation_masks = torch.tensor(validation_masks)

batch_size = 32

# Create the DataLoader for our training set.
train_data = TensorDataset(train_inputs, train_masks, train_targets)
train_dataloader = DataLoader(train_data, batch_size=batch_size)

# Create the DataLoader for our validation set.
validation_data = TensorDataset(validation_inputs, validation_masks, validation_targets)
validation_dataloader = DataLoader(validation_data, batch_size=batch_size)

# Load BertForSequenceClassification, the pretrained BERT model with a single
# linear classification layer on top.
model = BertForNextSequenceClassification.from_pretrained(
    "bert-base-uncased",  # Use the 12-layer BERT model, with an uncased vocab.
    num_labels=max_target_length,  # The number of output labels--2 for binary classification.
    # You can increase this for multi-class tasks.
    output_attentions=False,  # Whether the model returns attentions weights.
    output_hidden_states=False,  # Whether the model returns all hidden-states.
)

# Tell pytorch to run this model on the GPU.
model.to(device)

optimizer = AdamW(model.parameters(),
                  lr=2e-5,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
                  eps=1e-8  # args.adam_epsilon  - default is 1e-8.
                  )

# Number of training epochs (authors recommend between 2 and 4)
epochs = 4

# Total number of training steps is number of batches * number of epochs.
total_steps = len(train_dataloader) * epochs

# Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=0,  # Default value in run_glue.py
                                            num_training_steps=total_steps)

loss_values = []

for epoch_i in range(0, epochs):

    # ========================================
    #               Training
    # ========================================

    # Perform one full pass over the training set.

    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')

    # Measure how long the training epoch takes.
    t0 = time.time()

    # Reset the total loss for this epoch.
    total_loss = 0

    # Put the model into training mode. Don't be mislead--the call to
    # `train` just changes the *mode*, it doesn't *perform* the training.
    # `dropout` and `batchnorm` layers behave differently during training
    # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
    model.train()

    # For each batch of training data...
    for step, batch in enumerate(train_dataloader):

        # Progress update every 40 batches.
        if step % 40 == 0 and not step == 0:
            # Calculate elapsed time in minutes.
            elapsed = format_time(time.time() - t0)

            # Report progress.
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

        # Unpack this training batch from our dataloader.
        #
        # As we unpack the batch, we'll also copy each tensor to the GPU using the
        # `to` method.
        #
        # `batch` contains three pytorch tensors:
        #   [0]: input ids
        #   [1]: attention masks
        #   [2]: labels
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        # Always clear any previously calculated gradients before performing a
        # backward pass. PyTorch doesn't do this automatically because
        # accumulating the gradients is "convenient while training RNNs".
        # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
        model.zero_grad()

        # Perform a forward pass (evaluate the model on this training batch).
        # This will return the loss (rather than the model output) because we
        # have provided the `labels`.
        # The documentation for this `model` function is here:
        # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
        outputs = model(b_input_ids,
                        token_type_ids=None,
                        attention_mask=b_input_mask,
                        labels=b_labels)

        # The call to `model` always returns a tuple, so we need to pull the
        # loss value out of the tuple.
        loss = outputs[0]

        # Accumulate the training loss over all of the batches so that we can
        # calculate the average loss at the end. `loss` is a Tensor containing a
        # single value; the `.item()` function just returns the Python value
        # from the tensor.
        total_loss += loss.item()

        # Perform a backward pass to calculate the gradients.
        loss.backward()

        # Clip the norm of the gradients to 1.0.
        # This is to help prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Update parameters and take a step using the computed gradient.
        # The optimizer dictates the "update rule"--how the parameters are
        # modified based on their gradients, the learning rate, etc.
        optimizer.step()

        # Update the learning rate.
        scheduler.step()

    # Calculate the average loss over the training data.
    avg_train_loss = total_loss / len(train_dataloader)

    # Store the loss value for plotting the learning curve.
    loss_values.append(avg_train_loss)

    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epcoh took: {:}".format(format_time(time.time() - t0)))

    # ========================================
    #               Validation
    # ========================================
    # After the completion of each training epoch, measure our performance on
    # our validation set.

    print("")
    print("Running Validation...")

    t0 = time.time()

    # Put the model in evaluation mode--the dropout layers behave differently
    # during evaluation.
    model.eval()

    # Tracking variables
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0

    # Evaluate data for one epoch
    for batch in validation_dataloader:
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)

        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch

        # Telling the model not to compute or store gradients, saving memory and
        # speeding up validation
        with torch.no_grad():
            # Forward pass, calculate logit predictions.
            # This will return the logits rather than the loss because we have
            # not provided labels.
            # token_type_ids is the same as the "segment ids", which
            # differentiates sentence 1 and 2 in 2-sentence tasks.
            # The documentation for this `model` function is here:
            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
            outputs = model(b_input_ids,
                            token_type_ids=None,
                            attention_mask=b_input_mask)

        # Get the "logits" output by the model. The "logits" are the output
        # values prior to applying an activation function like the softmax.
        logits = outputs[0]

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        # Calculate the accuracy for this batch of test sentences.
        tmp_eval_accuracy = flat_accuracy(logits, label_ids)

        # Accumulate the total accuracy.
        eval_accuracy += tmp_eval_accuracy

        # Track the number of batches
        nb_eval_steps += 1

    # Report the final accuracy for this validation run.
    print("  Accuracy: {0:.2f}".format(eval_accuracy / nb_eval_steps))
    print("  Validation took: {:}".format(format_time(time.time() - t0)))

print("")
print("Training complete!")
