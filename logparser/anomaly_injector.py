import math
import os
import pickle
import random
from shutil import copyfile
import numpy as np

utah_new_random_line = "My personal randomly injected line.\n"
number_of_instances_to_inject_anomalies_in = 20
max_number_of_anomalies_per_instance = 4
number_of_swaps_per_instance = 4
overall_anomaly_ratio = 0.02
words_for_random_insert = ["time <*>", "for", "when", "during <*>", "deleted", "random", "bullshit", "this", "after",
                           "brain", "cell", "whatever"]
words_for_random_replace = ["bullshit", "brain", "cell", "whatever"]
max_number_of_words_to_be_altered = 1
ratio_of_words_to_be_altered_per_line = 0.15


# this will alter indices file
def delete_or_duplicate_events(corpus_input, corpus_output, anomaly_indices_output_path, instance_information_in,
                               instance_information_out, mode):
    if mode not in ["del", "dup", "ins"]:
        print("Allowed modes are del and dup. Exiting")
        return -1
    total_lines_file = open(corpus_input, 'r')
    total_lines = total_lines_file.readlines()
    total_lines_file.close()
    number_of_deletes = math.floor(len(total_lines) * overall_anomaly_ratio)
    instance_information = pickle.load(open(instance_information_in, 'rb'))

    instance_id_list = []

    # use instance information, to read every instance_id block in one list
    for instance in instance_information:
        instance_block = []
        for i in range(instance[0], instance[1] + 1):
            instance_block.append(total_lines[i])
        instance_id_list.append(instance_block)

    # select instance_ic blocks in which we will delete
    instance_id_indices_selected_for_altering = random.sample(range(0, len(instance_id_list)), number_of_deletes)
    indices_inside_blocks_to_alter = {}

    # delete lines and keep track of them
    for instance_id_block_to_alter in instance_id_indices_selected_for_altering:
        index_to_alter = random.randint(0, len(instance_id_list[instance_id_block_to_alter]) - 1)

        if mode == "ins":
            indices_inside_blocks_to_alter.update({instance_id_block_to_alter: index_to_alter})
            instance_id_list[instance_id_block_to_alter][index_to_alter:index_to_alter] = [utah_new_random_line]
        if mode == "del":
            indices_inside_blocks_to_alter.update({instance_id_block_to_alter: index_to_alter})
            del instance_id_list[instance_id_block_to_alter][index_to_alter]
        elif mode == "dup":
            # given, we want to duplicate line i, and line i+1 ... i+j are already exactly the same line,
            # we first greedily find the last duplicate in this possible row of duplicates, and insert it after the
            # last one of these duplicates
            line = instance_id_list[instance_id_block_to_alter][index_to_alter]
            next_index = index_to_alter
            while next_index + 1 < len(instance_id_list[instance_id_block_to_alter]) and \
                    instance_id_list[instance_id_block_to_alter][next_index + 1] == line:
                next_index += 1
            instance_id_list[instance_id_block_to_alter][next_index + 1:next_index + 1] = [line]
            indices_inside_blocks_to_alter.update({instance_id_block_to_alter: next_index + 1})

    # - re-write instance_id blocks to file
    # - overwrite instance information with updated (begin, end) intervals
    # - save anomaly indices
    new_instance_information = []
    output_file = open(corpus_output, 'w')
    anomaly_indices_file = open(anomaly_indices_output_path, 'w')
    anomaly_indices = []
    instance_block_end_line = 0
    for instance_id_block_index, instance_id_block in enumerate(instance_id_list):
        for line in instance_id_block:
            output_file.write(line)
        instance_block_begin_line = instance_block_end_line
        instance_block_end_line += len(instance_id_block)
        new_instance_information.append(tuple((instance_block_begin_line, instance_block_end_line - 1)))
        if instance_id_block_index in instance_id_indices_selected_for_altering:
            index_of_altered_line_inside_block = indices_inside_blocks_to_alter.get(instance_id_block_index)
            overall_index_of_deleted_line = instance_block_begin_line + index_of_altered_line_inside_block
            anomaly_indices.append(overall_index_of_deleted_line)
    pickle.dump(new_instance_information, open(instance_information_out, 'wb'))

    anomaly_indices.sort()
    for index in anomaly_indices:
        anomaly_indices_file.write(str(index) + "\n")
    return anomaly_indices


def insert_words(corpus_input, corpus_output, anomaly_indices_output_path, instance_information_in,
                 instance_information_out, number_of_words_to_be_added=1):
    copyfile(instance_information_in, instance_information_out)
    total_lines_file = open(corpus_input, 'r')
    total_lines = total_lines_file.readlines()
    total_lines_file.close()

    number_of_lines = len(total_lines)
    number_of_anomaly_lines_to_be_manipulated = math.floor(number_of_lines * overall_anomaly_ratio)
    line_indices_to_be_altered = random.sample(range(len(total_lines)), number_of_anomaly_lines_to_be_manipulated)

    lines_before_alter = []
    lines_after_alter = []

    for index in line_indices_to_be_altered:
        # select line for altering
        line = total_lines[index]
        lines_before_alter.append(line)
        line = line.split()
        for _ in range(0, number_of_words_to_be_added):
            line.insert(random.randrange(0, len(line)), random.choice(words_for_random_insert))
        # re-insert altered line
        line = " ".join(line) + "\n"
        lines_after_alter.append(line)
        total_lines[index] = line

    output_file = open(corpus_output, 'w')
    for line in total_lines:
        output_file.write(line)
    output_file.close()

    anomaly_indices_file = open(anomaly_indices_output_path, 'w')
    line_indices_to_be_altered.sort()
    for anomaly_index in line_indices_to_be_altered:
        anomaly_indices_file.write(str(anomaly_index) + "\n")
    anomaly_indices_file.close()

    return lines_before_alter, lines_after_alter, line_indices_to_be_altered


def replace_words(corpus_input, corpus_output, anomaly_indices_output_path, instance_information_in,
                  instance_information_out, number_of_words_to_be_replaced=1):
    copyfile(instance_information_in, instance_information_out)
    total_lines_file = open(corpus_input, 'r')
    total_lines = total_lines_file.readlines()
    total_lines_file.close()

    lines_before_alter = []
    lines_after_alter = []

    number_of_lines = len(total_lines)
    number_of_anomaly_lines_to_be_manipulated = math.floor(number_of_lines * overall_anomaly_ratio)
    line_indices_to_be_altered = random.sample(range(len(total_lines)), number_of_anomaly_lines_to_be_manipulated)
    for index in line_indices_to_be_altered:
        # select line for altering
        line = total_lines[index]
        lines_before_alter.append(line)
        line = line.split()
        indeces_to_be_replaced = random.sample(range(len(line)), number_of_words_to_be_replaced)
        for replace in indeces_to_be_replaced:
            line[replace] = random.choice(words_for_random_replace)
        # re-insert altered line
        line = " ".join(line) + "\n"
        lines_after_alter.append(line)
        total_lines[index] = line

    output_file = open(corpus_output, 'w')
    for line in total_lines:
        output_file.write(line)
    output_file.close()

    anomaly_indices_file = open(anomaly_indices_output_path, 'w')
    line_indices_to_be_altered.sort()
    for anomaly_index in line_indices_to_be_altered:
        anomaly_indices_file.write(str(anomaly_index) + "\n")
    anomaly_indices_file.close()

    return lines_before_alter, lines_after_alter, line_indices_to_be_altered


def remove_words(corpus_input, corpus_output, anomaly_indices_output_path, instance_information_in,
                 instance_information_out, number_of_words_to_be_removed=1):
    copyfile(instance_information_in, instance_information_out)
    total_lines_file = open(corpus_input, 'r')
    total_lines = total_lines_file.readlines()
    total_lines_file.close()

    lines_before_alter = []
    lines_after_alter = []

    number_of_lines = len(total_lines)
    # number_of_lines = sum([len(instance_id_dict[x]) for x in instance_id_dict])
    number_of_anomaly_lines_to_be_manipulated = math.floor(number_of_lines * overall_anomaly_ratio)
    line_indices_to_be_altered = random.sample(range(len(total_lines)), number_of_anomaly_lines_to_be_manipulated)
    for index in line_indices_to_be_altered:
        # select line for altering
        line = total_lines[index]
        lines_before_alter.append(line)
        line = line.split()
        removed_words = []
        for _ in range(0, number_of_words_to_be_removed):
            random_index = random.randrange(0, len(line))
            removed_words.append(line[random_index])
            del line[random_index]
        # re-insert altered line
        line = " ".join(line) + "\n"
        lines_after_alter.append(line)
        total_lines[index] = line

    output_file = open(corpus_output, 'w')
    for line in total_lines:
        output_file.write(line)
    output_file.close()

    anomaly_indices_file = open(anomaly_indices_output_path, 'w')
    line_indices_to_be_altered.sort()
    for anomaly_index in line_indices_to_be_altered:
        anomaly_indices_file.write(str(anomaly_index) + "\n")
    anomaly_indices_file.close()

    return lines_before_alter, lines_after_alter, line_indices_to_be_altered


def reverse_order(corpus_input, corpus_output, instance_information_in, instance_information_out, anomaly_indices_output_path):
    copyfile(instance_information_in, instance_information_out)
    with open(corpus_input, 'r') as f:
        total_lines = f.readlines()
    instance_information = pickle.load(open(instance_information_in, 'rb'))

    instance_id_list = []
    for instance in instance_information:
        instance_block = []
        for i in range(instance[0], instance[1] + 1):
            instance_block.append(total_lines[i])
        instance_id_list.append(instance_block)

    flipped_instanced_id_list = []
    for instance_block in instance_id_list:
        flipped_instanced_id_list.append(np.flip(instance_block, 0))

    with open(corpus_output, 'w') as f:
        for block in flipped_instanced_id_list:
            for line in block:
                f.write(str(line))

    anomaly_indices = np.arange(len(total_lines))
    with open(anomaly_indices_output_path, 'w') as f:
        for i in anomaly_indices:
            f.write(str(i) + "\n")

    return anomaly_indices


def shuffle(corpus_input, corpus_output, instance_information_in, instance_information_out, anomaly_indices_output_path,
            shuffles_per_instance=1):
    copyfile(instance_information_in, instance_information_out)
    shuffle_distances = [-6, -5, -4, -3, -2, 2, 3, 4, 5, 6]
    corpus = open(corpus_input, 'r').readlines()
    assert corpus
    instance_information = pickle.load(open(instance_information_in, 'rb'))
    assert instance_information

    total_number_of_lines = len(corpus)
    number_of_anomaly_lines_to_be_manipulated = math.floor(total_number_of_lines * overall_anomaly_ratio)
    interval_indeces_with_more_than_four_lines = []
    for interval_index, interval in enumerate(instance_information):
        if interval[1] - interval[0] > 4:
            interval_indeces_with_more_than_four_lines.append(interval_index)
    # TODO: wenn man 18k nimmt, dann gibt es weniger Intervalle, als number_of_anomaly_lines_to_be_manipulated,
    #   man müsste dann mehrere verschiebungen pro Intervall machen, vielleicht später
    if number_of_anomaly_lines_to_be_manipulated > len(interval_indeces_with_more_than_four_lines):
        intervals_to_be_altered = interval_indeces_with_more_than_four_lines
    else:
        intervals_to_be_altered = random.sample(interval_indeces_with_more_than_four_lines,
                                                number_of_anomaly_lines_to_be_manipulated)

    os.makedirs(os.path.dirname(corpus_output), exist_ok=True)
    output_file = open(corpus_output, 'w')
    anomaly_indices_output_file = open(anomaly_indices_output_path, 'w')
    anomaly_indices = []
    for interval_index, interval in enumerate(instance_information):
        shuffle_indeces = False
        # check if we're currently looking at an inst_id that's supposed to receive anomalies, and inject them
        if interval_index in intervals_to_be_altered:
            shuffle_indeces = True
            index_to_be_altered = random.choice(list(range(interval[0], interval[1])))
            shuffle_distance = random.choice(shuffle_distances)
            shuffle_index = index_to_be_altered + shuffle_distance
            # check if we're still inside the interval
            if shuffle_index < interval[0] or shuffle_index > interval[1]:
                shuffle_index = index_to_be_altered - shuffle_distance

            line_to_be_altered = corpus[index_to_be_altered]
            del corpus[index_to_be_altered]
            corpus[shuffle_index:shuffle_index] = [line_to_be_altered]
        this_list = corpus[interval[0]:interval[1] + 1]
        # for now, only put in shuffle_index as anomaly
        # TODO: check, if all indices that were shuffled should be marked as anomalies


        for l in this_list:
            output_file.write(l)
        if shuffle_indeces:
            assert (shuffle_index is not None)
            if shuffle_index < index_to_be_altered:
                anomaly_indices.append(shuffle_index)
            elif shuffle_index > index_to_be_altered:
                anomaly_indices.append(index_to_be_altered)
            else:
                raise Exception ("This shouldn't have happened. Have a look at the logic here.")
        shuffle_index = None

    anomaly_indices.sort()
    for index in anomaly_indices:
        anomaly_indices_output_file.write(str(index) + "\n")

    anomaly_indices_output_file.close()
    output_file.close()

    return anomaly_indices


def no_anomaly(corpus_input, corpus_output, instance_information_in, instance_information_out, anomaly_indices_output_path):
    copyfile(instance_information_in, instance_information_out)
    copyfile(corpus_input, corpus_output)
    return []