import datetime
import os
import random
import time
import argparse

import numpy as np
import torch
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from transformers import AdamW, BertForMaskedLM
from transformers import BertTokenizer as transformers_BertTokenizer
from transformers import get_linear_schedule_with_warmup
from typing import Tuple
import math

epochs = 4


def format_time(elapsed):
    """
    Takes a time in seconds and returns a string hh:mm:ss
    """
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def mask_tokens(inputs: torch.Tensor, tokenizer: transformers_BertTokenizer) -> Tuple[torch.Tensor, torch.Tensor]:
    # """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
    labels = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training
    # (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    probability_matrix = torch.full(labels.shape, 0.15)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
    ]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
    if tokenizer._pad_token is not None:
        padding_mask = labels.eq(tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -1  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]
    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels

########################################################################################################################
#                                               ARGUMENT PARSE
########################################################################################################################

def finetune(templates='../data/openstack/utah/parsed/merged_templates/137k+18k_spr_injected_2_words',
             output_dir='finetuning-models/137k+18k_spr_injected_2_words'):

    # arg parsing and assignment
    if type(templates) == str:
        sentences = open(templates, 'r')
    elif type(templates) == list:
        sentences = templates
    else:
        print("templates must be either a list of templates or a path str")
        raise
    os.makedirs(output_dir, exist_ok=True)
    logfile_path = open(output_dir + "/log.txt", 'w')


    tokenizer = transformers_BertTokenizer.from_pretrained('bert-base-uncased')
    sentences_duplicated = []

    for sent in sentences:
        sent_len = len(sent.strip())
        number_of_repititions = math.floor(len(sent.split()) * 0.3)
        if number_of_repititions == 0:
            sentences_duplicated.extend([sent for _ in range(epochs)])
        else:
            sentences_duplicated.extend([sent for _ in range(number_of_repititions * 3)])

    tokenized_text = []
    for sent in sentences_duplicated:
        tokenized_text.append(
            tokenizer.build_inputs_with_special_tokens(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sent))))
    max_sent_len = max(len(sen) for sen in tokenized_text)
    tokenized_text = pad_sequences(tokenized_text, maxlen=max_sent_len, dtype="long",
                                   value=0, truncating="post", padding="post")
    tokenized_text = torch.tensor(tokenized_text)

    device = torch.device("cpu")

    inputs, labels = mask_tokens(tokenized_text, tokenizer)

    train_inputs, validation_inputs = train_test_split(inputs, random_state=2020, test_size=0.1)
    train_labels, validation_labels = train_test_split(labels, random_state=2020, test_size=0.1)

    batch_size = 8

    # Create the DataLoader for our training set.
    train_data = TensorDataset(train_inputs, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    validation_data = TensorDataset(validation_inputs, validation_labels)
    validation_sampler = RandomSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

    model = BertForMaskedLM.from_pretrained(
        "bert-base-uncased",
        output_attentions=False,
        output_hidden_states=False
    )

    # Tell pytorch to run this model on the GPU.
    model.to(device)

    optimizer = AdamW(model.parameters(),
                      lr=2e-5,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
                      eps=1e-8  # args.adam_epsilon  - default is 1e-8.
                      )

    # Total number of training steps is number of batches * number of epochs.
    total_steps = len(train_dataloader) * epochs

    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,  # Default value in run_glue.py
                                                num_training_steps=total_steps)

    seed_val = 42

    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    loss_values = []

    for epoch_i in range(0, epochs):

        # ========================================
        #               Training
        # ========================================

        # Perform one full pass over the training set.
        best_val_loss = None
        print("")
        logfile_path.write("\n")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        logfile_path.write('======== Epoch {:} / {:} ========\n'.format(epoch_i + 1, epochs))
        print('Training...')
        logfile_path.write("Training...\n")

        # Measure how long the training epoch takes.
        t0 = time.time()

        # Reset the total loss for this epoch.
        total_loss = 0

        model.train()

        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):

            # Progress update every 40 batches.
            if step % 40 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)

                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))
                logfile_path.write('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.\n'.format(step, len(train_dataloader), elapsed))

            # batch: padded_input_ids, mask_positions, mask_words
            # Unpack this training batch from our dataloader.
            b_input_ids = batch[0].to(device)
            b_input_masks = batch[1].to(device)

            model.zero_grad()

            outputs = model(b_input_ids, masked_lm_labels=b_input_masks)
            loss = outputs[0]
            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        # Calculate the average loss over the training data.
        avg_train_loss = total_loss / len(train_dataloader)

        # Store the loss value for plotting the learning curve.
        loss_values.append(avg_train_loss)

        print("")
        logfile_path.write("\n")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        logfile_path.write('  Average training loss: {0:.2f}\n'.format(avg_train_loss))
        print("  Training epoch took: {:}".format(format_time(time.time() - t0)))
        logfile_path.write('  Training epoch took: {:}\n'.format(format_time(time.time() - t0)))

        # ========================================
        #               Validation
        # ========================================
        # After the completion of each training epoch, measure our performance on
        # our validation set.

        print("")
        logfile_path.write("\n")
        print("Running Validation...")
        logfile_path.write("Running Validation...\n")

        t0 = time.time()

        # Put the model in evaluation mode--the dropout layers behave differently
        # during evaluation.
        model.eval()

        # Tracking variables
        eval_loss = 0
        nb_eval_steps = 0

        # Evaluate data for one epoch
        for batch in validation_dataloader:
            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)

            # Unpack the inputs from our dataloader
            b_input_ids, b_input_labels = batch

            # Telling the model not to compute or store gradients, saving memory and
            # speeding up validation
            with torch.no_grad():
                outputs = model(b_input_ids, masked_lm_labels=b_input_labels)
                lm_loss = outputs[0]
                eval_loss = lm_loss.mean().item()

            nb_eval_steps += 1

        if not best_val_loss or eval_loss < best_val_loss:
            best_val_loss = eval_loss
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)


        # Report the final accuracy for this validation run.
        print("  Loss eval: {0:.2f}".format(lm_loss / nb_eval_steps))
        logfile_path.write('  Loss eval: {0:.2f}\n'.format(lm_loss / nb_eval_steps))
        print("  Validation took: {:}".format(format_time(time.time() - t0)))
        logfile_path.write('  Validation took: {:}\n'.format(format_time(time.time() - t0)))

    print("")
    logfile_path.write("\n")
    print("Training complete!")
    logfile_path.write("Training complete!\n")
    logfile_path.close()

if __name__ == "__main__":
    finetune()