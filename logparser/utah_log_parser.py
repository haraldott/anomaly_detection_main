import re
import os
import pickle
import random
import math
from numpy import mean
from collections import defaultdict

my_random_line_sasho = 'AW6jHxiZRnRNbAqZnom1,flog-2019.11.25,1.0,fluentd,wally117,1f7d6d735d5d4513ae1f7a5ac6888c45,default,2019-11-25 16:12:29.581,2019-11-25T16:12:29.581000000+01:00,INFO,6.0,"[instance: {}] My personal randomly injected line.",f09ddef028834df19337502ece1490c5,nova-compute,0a5bfbed-f4f7-4a42-9333-2d99adb16cdd,nova.compute.claims,openstack.nova,default,-,,,,,,,,,,,'
randomly_injected_line_utah = "nova-compute.log.2017-05-14_21:27:09 2017-05-14 19:39:22.577 2931 INFO nova.compute.manager [req-3ea4052c-895d-4b64-9e2d-04d64c4d94ab - - - - -] [instance: {}] My personal randomly injected line.\n"
randomly_injected_line_utah_2 = "nova-compute.log.2017-05-14_21:27:09 2017-05-14 19:39:22.577 2931 INFO nova.compute.manager [req-3ea4052c-895d-4b64-9e2d-04d64c4d94ab - - - - -] [instance: {}] Openstack connection failure. Unable to establish connection.\n"
randomly_injected_line_utah_new = "nova-compute.log.2017-05-14_21:27:09 2017-05-14 19:39:22.577 2931 INFO nova.compute.manager [req-3ea4052c-895d-4b64-9e2d-04d64c4d94ab - - - - -] [instance: {}] Deleting instance files /var/lib/nova/instances/a6c5e900-d575-4447-a815-3e156c84aa90_del now.\n"

number_of_instances_to_inject_anomalies_in = 20
max_number_of_anomalies_per_instance = 4
number_of_swaps_per_instance = 4
overall_anomaly_ratio = 0.02
words_for_random_insert = ["time <*>", "for", "when", "during <*>", "deleted", "random", "bullshit", "this", "after",
                           "brain", "cell", "whatever"]
max_number_of_words_to_be_altered = 1
ratio_of_words_to_be_altered_per_line = 0.15


############# BEFORE DRAIN #############
def instance_id_sort(
        logfile_path='/Users/haraldott/Development/thesis/detector/data/openstack/utah/raw/137k',
        output_path='/Users/haraldott/Development/thesis/detector/data/openstack/utah/raw/sorted_per_request/137k_spr',
        instance_information_path='/Users/haraldott/Development/thesis/detector/data/openstack/utah/raw/sorted_per_request_pickle/137k_spr.pickle'):
    """
    Takes a raw Openstack log file (logfile_path) as input, sorts per instance_id, removes all lines without instance_id,
    outputs the sorted file to output_path, and saves instance information (i.e. instance_id log line block from i to j)
    to instance_information_path
    :param logfile_path: raw Openstack log file
    :param output_path:
    :param instance_information_path:
    :return: void
    """
    instance_id_dict = __sort_per_instance_id(logfile_path)
    __create_instance_intervals(instance_id_dict, output_path, instance_information_path)


def parse_sort_and_inject_random_lines(logfile_path='../data/openstack/utah/raw/openstack_18k_anomalies',
                                       output_path='../data/openstack/utah/raw/sorted_per_request/openstack_18k_random_lines_anomalies_new',
                                       instance_information_path='../data/openstack/utah/raw/sorted_per_request_pickle/openstack_18k_random_lines_anomalies_new.pickle',
                                       anomaly_indices_output_path='../data/openstack/utah/raw/sorted_per_request/anomaly_indices_18k_random_lines_new.txt'):
    instance_id_dict = __sort_per_instance_id(logfile_path)
    instance_id_dict, line_numbers_containing_anomalies_per_instance = __inject_random_line(instance_id_dict)
    __create_instance_intervals_with_anomalies(instance_id_dict,
                                               output_path,
                                               instance_information_path,
                                               line_numbers_containing_anomalies_per_instance,
                                               anomaly_indices_output_path)


def parse_sort_and_swap_lines(logfile_path='../data/openstack/utah/raw/openstack_18k_anomalies',
                              output_path='../data/openstack/utah/raw/sorted_per_request/openstack_18k_swapped_anomalies',
                              instance_information_path='../data/openstack/utah/raw/sorted_per_request_pickle/openstack_18k_swapped_anomalies.pickle',
                              anomaly_indices_output_path='../data/openstack/utah/raw/sorted_per_request/anomaly_indices_18k_swapped.txt'):
    instance_id_dict = __sort_per_instance_id(logfile_path)
    instance_id_dict, line_numbers_swapped_per_instance = __swap_log_lines(instance_id_dict)
    __create_instance_intervals_with_anomalies(instance_id_dict, output_path, instance_information_path,
                                               line_numbers_swapped_per_instance, anomaly_indices_output_path)

############# AFTER DRAIN #############
def shuffle_log_sequences(
        corpus_input_file_path='/Users/haraldott/Development/thesis/detector/data/openstack/utah/parsed/18k_spr_corpus',
        output_file_path='/Users/haraldott/Development/thesis/detector/data/openstack/utah/parsed/anomalies_injected/18k_spr_shuffled',
        instance_information_path='/Users/haraldott/Development/thesis/detector/data/openstack/utah/raw/sorted_per_request_pickle/18k_spr.pickle',
        anomaly_indices_output_path='/Users/haraldott/Development/thesis/detector/data/openstack/utah/parsed/anomaly_indeces/18k_spr_shuffled_indices.txt'):
    shuffle(corpus_input_file_path, output_file_path, instance_information_path, anomaly_indices_output_path)




########################################################################################################################
########################################################################################################################
#                                           HELPER FUNCTIONS
########################################################################################################################
########################################################################################################################

def remove_event(corpus_input_file_path, output_path, instance_information_path,
                 anomaly_indices_output_path):
    delete_numbers = [1, 2]
    corpus = open(corpus_input_file_path, 'r').readlines()
    assert corpus
    instance_information = pickle.load(open(instance_information_path, 'rb'))
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

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    output_file = open(output_path, 'w')
    anomaly_indices_output_file = open(anomaly_indices_output_path, 'w')

    for interval_index, interval in enumerate(instance_information):
        shuffle_indeces = False
        # check if we're currently looking at an inst_id that's supposed to receive anomalies, and inject them
        if interval_index in intervals_to_be_altered:
            shuffle_indeces = True
            index_to_be_altered = random.choice(list(range(interval[0], interval[1])))
            no_of_deletes = random.choice(delete_numbers)
            shuffle_index = index_to_be_altered + no_of_deletes
            # check if we're still inside the interval
            if shuffle_index < interval[0] or shuffle_index > interval[1]:
                shuffle_index = index_to_be_altered - no_of_deletes

            # if corpus[index_to_be_altered] == corpus[shuffle_index]:
            #     print("Zeileninhalt: {}, index_to_be_altered: {}, shuffle_index: {}".format(corpus[index_to_be_altered], index_to_be_altered, shuffle_index))
            line_to_be_altered = corpus[index_to_be_altered]
            del corpus[index_to_be_altered]
            corpus[shuffle_index:shuffle_index] = [line_to_be_altered]
        this_list = corpus[interval[0]:interval[1] + 1]
        # for now, only put in shuffle_index as anomaly
        # TODO: check, if all indices that were shuffled should be marked as anomalies

        for l in this_list:
            output_file.write(l)
        if shuffle_indeces:
            assert (index_to_be_altered is not None)
            anomaly_indices_output_file.write(str(index_to_be_altered) + "\n")
        shuffle_index = None
    anomaly_indices_output_file.close()
    output_file.close()


def shuffle(corpus_input_file_path, output_path, instance_information_path,
            anomaly_indices_output_path):
    shuffle_distances = [-6, -5, -4, -3, -2, 2, 3, 4, 5, 6]
    corpus = open(corpus_input_file_path, 'r').readlines()
    assert corpus
    instance_information = pickle.load(open(instance_information_path, 'rb'))
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

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    output_file = open(output_path, 'w')
    anomaly_indices_output_file = open(anomaly_indices_output_path, 'w')
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
                anomaly_indices_output_file.write(str(shuffle_index) + "\n")
            elif shuffle_index > index_to_be_altered:
                anomaly_indices_output_file.write(str(index_to_be_altered) + "\n")
            else:
                raise Exception ("This shouldn't have happened. Have a look at the logic here.")
        shuffle_index = None
    anomaly_indices_output_file.close()
    output_file.close()


def __sort_per_instance_id_without_dict(log_lines_path):
    logfile_lines = open(log_lines_path, 'r').readlines()
    number_of_no_id = 0
    total_lines = []
    for line in logfile_lines:
        m = re.search('\[instance:\s([^\]]+)', line)
        if m is not None:
            total_lines.append(line)
        else:
            number_of_no_id += 1
    return total_lines


def __sort_per_instance_id(log_lines_path):
    logfile_lines = open(log_lines_path, 'r').readlines()
    number_of_no_id = 0
    instance_id_dict = {}
    for line in logfile_lines:
        m = re.search('\[instance:\s([^\]]+)', line)
        if m is not None:
            instance_id = m.group(1)
            if instance_id_dict.get(instance_id) is None:
                instance_id_dict[instance_id] = [line]
            else:
                instance_id_dict[instance_id].append(line)
        else:
            number_of_no_id += 1
    return instance_id_dict


def __inject_random_line(instance_id_dict):
    instance_ids_selected_for_anomaly_injection = random.sample(list(instance_id_dict.keys()),
                                                                number_of_instances_to_inject_anomalies_in)
    line_numbers_containing_anomalies_per_instance = defaultdict(list)
    for instance_id in instance_ids_selected_for_anomaly_injection:
        this_anomaly_line = randomly_injected_line_utah.format(instance_id)
        for _ in range(0, max_number_of_anomalies_per_instance):
            line_number_of_anomaly = random.randrange(len(instance_id_dict[instance_id]) + 1)
            instance_id_dict[instance_id].insert(line_number_of_anomaly, this_anomaly_line)
            # line_numbers_containing_anomalies_per_instance[instance_id].append(line_number_of_anomaly)
        for i, line in enumerate(instance_id_dict[instance_id]):
            if this_anomaly_line in line:
                line_numbers_containing_anomalies_per_instance[instance_id].append(i)

    return instance_id_dict, line_numbers_containing_anomalies_per_instance


def __create_instance_intervals_with_anomalies(instance_id_dict,
                                               output_path,
                                               instance_information_path,
                                               line_numbers_containing_anomalies_per_instance,
                                               anomaly_indices_output_path):
    anomaly_indices = []
    linecounter = 0
    instance_information = []
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    output_file = open(output_path, 'w')
    for inst_id in instance_id_dict:
        this_list = instance_id_dict.get(inst_id)
        for line in this_list:
            output_file.write(line)
        instance_block_begin_line = linecounter
        linecounter += len(this_list)
        if inst_id in line_numbers_containing_anomalies_per_instance:
            for index in line_numbers_containing_anomalies_per_instance[inst_id]:
                anomaly_indices.append(instance_block_begin_line + index)
        instance_information.append(tuple((instance_block_begin_line, linecounter - 1)))

    os.makedirs(os.path.dirname(instance_information_path), exist_ok=True)
    pickle.dump(instance_information, open(instance_information_path, 'wb'))
    anomaly_indices_output_file = open(anomaly_indices_output_path, 'w')
    for val in anomaly_indices:
        anomaly_indices_output_file.write(str(val) + "\n")
    anomaly_indices_output_file.close()


def __create_instance_intervals(instance_id_dict, output_path, instance_information_path):
    linecounter = 0
    instance_information = []
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    output_file = open(output_path, 'w')
    for inst_id in instance_id_dict:
        this_list = instance_id_dict.get(inst_id)
        if len(this_list) > 4:
            for line in this_list:
                output_file.write(line)
            instance_block_begin_line = linecounter
            linecounter += len(this_list)
            instance_information.append(tuple((instance_block_begin_line, linecounter - 1)))
    output_file.close()

    os.makedirs(os.path.dirname(instance_information_path), exist_ok=True)
    pickle.dump(instance_information, open(instance_information_path, 'wb'))


def __swap_log_lines(instance_id_dict):
    instance_ids_selected_for_swapping = random.sample(list(instance_id_dict.keys()),
                                                       number_of_instances_to_inject_anomalies_in)
    line_numbers_swapped_per_instance = defaultdict(list)
    for instance_id in instance_ids_selected_for_swapping:
        indices_to_swap = random.sample(range(len(instance_id_dict[instance_id]) - 1), number_of_swaps_per_instance)
        indices_swapped = indices_to_swap.copy()
        for ind in indices_to_swap:
            indices_swapped.append(ind + 1)
            tmp_line = instance_id_dict[instance_id][ind]
            instance_id_dict[instance_id][ind] = instance_id_dict[instance_id][ind + 1]
            instance_id_dict[instance_id][ind + 1] = tmp_line
        indices_swapped = list(set(indices_swapped))
        [line_numbers_swapped_per_instance[instance_id].append(index) for index in indices_swapped]
    return instance_id_dict, line_numbers_swapped_per_instance


def __shuffle_events():
    instance_id_dict = __sort_per_instance_id('../data/openstack/utah/raw/openstack_137k_normal')
    number_of_anomalies_per_instance_range = [2, 3]
    average_number_of_changes_per_instance = 2.5
    number_of_lines = 0
    for id in instance_id_dict:
        number_of_lines += len(instance_id_dict[id])
    number_of_anomaly_lines_to_be_manipulated = math.floor(number_of_lines * overall_anomaly_ratio)
    instance_ids_to_be_altered = random.sample(instance_id_dict.keys(), math.floor(number_of_anomaly_lines_to_be_manipulated / average_number_of_changes_per_instance))
    for instance_id in instance_ids_to_be_altered:
        number_of_changes = random.choice(number_of_anomalies_per_instance_range)
        instance_id_dict[instance_id]
    print("hello")


########################################################################################################################
########################################################################################################################
#                                           STANDALONE FUNCTIONS
########################################################################################################################
########################################################################################################################


def insert_words(input_file_path, output_file_path, anomaly_indices_output_path, number_of_words_to_be_added=1):
    total_lines_file = open(input_file_path, 'r')
    total_lines = total_lines_file.readlines()
    total_lines_file.close()

    number_of_lines = len(total_lines)
    number_of_anomaly_lines_to_be_manipulated = math.floor(number_of_lines * overall_anomaly_ratio)
    line_indices_to_be_altered = random.sample(range(len(total_lines)), number_of_anomaly_lines_to_be_manipulated)

    for index in line_indices_to_be_altered:
        # select line for altering
        line = total_lines[index]
        line = line.split()
        for _ in range(0, number_of_words_to_be_added):
            line.insert(random.randrange(0, len(line)), random.choice(words_for_random_insert))

        # re-insert altered line
        line = " ".join(line) + "\n"
        total_lines[index] = line

    output_file = open(output_file_path, 'w')
    for line in total_lines:
        output_file.write(line)
    output_file.close()

    anomaly_indices_file = open(anomaly_indices_output_path, 'w')
    for anomaly_index in line_indices_to_be_altered:
        anomaly_indices_file.write(str(anomaly_index) + "\n")
    anomaly_indices_file.close()


def remove_words(input_file_path, output_file_path, anomaly_indices_output_path, number_of_words_to_be_removed=1):
    total_lines_file = open(input_file_path, 'r')
    total_lines = total_lines_file.readlines()
    total_lines_file.close()

    number_of_lines = len(total_lines)
    # number_of_lines = sum([len(instance_id_dict[x]) for x in instance_id_dict])
    number_of_anomaly_lines_to_be_manipulated = math.floor(number_of_lines * overall_anomaly_ratio)
    line_indices_to_be_altered = random.sample(range(len(total_lines)), number_of_anomaly_lines_to_be_manipulated)
    for index in line_indices_to_be_altered:
        # select line for altering
        line = total_lines[index]
        line = line.split()
        removed_words = []
        for _ in range(0, number_of_words_to_be_removed):
            random_index = random.randrange(0, len(line))
            removed_words.append(line[random_index])
            del line[random_index]

        # re-insert altered line
        line = " ".join(line) + "\n"
        total_lines[index] = line

    output_file = open(output_file_path, 'w')
    for line in total_lines:
        output_file.write(line)
    output_file.close()

    anomaly_indices_file = open(anomaly_indices_output_path, 'w')
    for anomaly_index in line_indices_to_be_altered:
        anomaly_indices_file.write(str(anomaly_index) + "\n")
    anomaly_indices_file.close()

def insert_and_remove_words(input_file_path, output_file_path, anomaly_indices_output_path):
    total_lines_file = open(input_file_path, 'r')
    total_lines = total_lines_file.readlines()
    total_lines_file.close()

    number_of_lines = len(total_lines)
    # number_of_lines = sum([len(instance_id_dict[x]) for x in instance_id_dict])
    number_of_anomaly_lines_to_be_manipulated = math.floor(number_of_lines * overall_anomaly_ratio)
    line_indices_to_be_altered = random.sample(range(len(total_lines)), number_of_anomaly_lines_to_be_manipulated)
    fns = ["remove", "insert"]
    for index in line_indices_to_be_altered:
        # select line for altering
        line = total_lines[index]
        line = line.split()
        removed_words = []
        number_of_words_to_be_altered = max(math.ceil(len(line) * ratio_of_words_to_be_altered_per_line), 1)
        choice = random.choice(fns)
        for _ in range(number_of_words_to_be_altered):
            if choice == "insert":
                line.insert(random.randrange(0, len(total_lines[index])), random.choice(words_for_random_insert))
            if choice == "remove":
                # make sure that we don't remove <*>, but an actual word
                next = True
                i = 0
                while i in range(len(line)) and next:
                    random_index = random.randrange(0, len(line))
                    # we found a word that is not "<*>", so we can stop our search and continue
                    if line[random_index] != "<*>":
                        removed_words.append(line[random_index])
                        del line[random_index]
                        next = False
                    if next and i == len(line) - 1:
                        print("Warning: During search for a word != \"<*>\","
                              "all the words were \"<*>\", so we remove the last occurrence")
                        removed_words.append(line[random_index])
                        try:
                            del line[random_index]
                        except IndexError:
                            print("tried to delete index {} in line {}".format(random_index, index))
                    i = + 1

        # re-insert altered line
        line = " ".join(line) + "\n"
        total_lines[index] = line

    output_file = open(output_file_path, 'w')
    for line in total_lines:
        output_file.write(line)
    output_file.close()

    anomaly_indices_file = open(anomaly_indices_output_path, 'w')
    for anomaly_index in line_indices_to_be_altered:
        anomaly_indices_file.write(str(anomaly_index) + "\n")
    anomaly_indices_file.close()


    # this will alter indices file
def delete_or_duplicate_events(input_file_path, output_file_path, anomaly_indices_output_path, instance_information_path_in, instance_information_path_out, mode):
    print(mode)
    if mode not in ["del", "dup"]:
        print("Allowed modes are del and dup. Exiting")
        return -1
    total_lines_file = open(input_file_path, 'r')
    total_lines = total_lines_file.readlines()
    total_lines_file.close()
    number_of_deletes = math.floor(len(total_lines) * overall_anomaly_ratio)
    instance_information = pickle.load(open(instance_information_path_in, 'rb'))

    instance_id_list = []

    # use instance information, to read every instance_id block in one list
    for instance in instance_information:
        instance_block = []
        for i in range(instance[0], instance[1]+1):
            instance_block.append(total_lines[i])
        instance_id_list.append(instance_block)

    # select instance_ic blocks in which we will delete
    instance_id_indices_selected_for_altering = random.sample(range(0, len(instance_id_list)), number_of_deletes)
    indices_inside_blocks_to_alter = {}

    # delete lines and keep track of them
    for instance_id_block_to_alter in instance_id_indices_selected_for_altering:
        print(instance_id_block_to_alter)
        index_to_alter = random.randint(0, len(instance_id_list[instance_id_block_to_alter])-1)

        if mode == "del":
            indices_inside_blocks_to_alter.update({instance_id_block_to_alter: index_to_alter})
            del instance_id_list[instance_id_block_to_alter][index_to_alter]
        elif mode == "dup":
            # given, we want to duplicate line i, and line i+1 ... i+j are already exactly the same line,
            # we first greedily find the last duplicate in this possible row of duplicates, and insert it after the
            # last one of these duplicates
            line = instance_id_list[instance_id_block_to_alter][index_to_alter]
            next_index = index_to_alter
            while next_index+1 < len(instance_id_list[instance_id_block_to_alter]) and instance_id_list[instance_id_block_to_alter][next_index+1] == line:
                next_index += 1
            instance_id_list[instance_id_block_to_alter][next_index+1:next_index+1] = [line]
            indices_inside_blocks_to_alter.update({instance_id_block_to_alter: next_index+1})

    # - re-write instance_id blocks to file
    # - overwrite instance information with updated (begin, end) intervals
    # - save anomaly indices
    new_instance_information = []
    output_file = open(output_file_path, 'w')
    anomaly_indices = open(anomaly_indices_output_path, 'w')
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
            anomaly_indices.write(str(overall_index_of_deleted_line) + "\n")
    pickle.dump(new_instance_information, open(instance_information_path_out, 'wb'))

if __name__ == '__main__':
    # parse_sort_and_inject_random_lines(logfile_path='/Users/haraldott/Development/thesis/detector/data/openstack/utah/raw/18k',
    #                                    output_path='/Users/haraldott/Development/thesis/detector/data/openstack/utah/parsed/anomalies_injected/18k_random_lines',
    #                                    instance_information_path='/Users/haraldott/Development/thesis/detector/data/openstack/utah/raw/sorted_per_request_pickle/anomalies/18k_spr_random_lines.pickle',
    #                                    anomaly_indices_output_path='/Users/haraldott/Development/thesis/detector/data/openstack/utah/parsed/anomaly_indeces/18k_spr_random_lines_indeces.txt')
    # instance_id_sort()
    insert_words("/Users/haraldott/Development/thesis/detector/data/openstack/utah/parsed/18k_spr_corpus",
                 "/Users/haraldott/Development/thesis/detector/data/openstack/utah/parsed/anomalies_injected/18k_spr_injected_1_words",
                 "/Users/haraldott/Development/thesis/detector/data/openstack/utah/parsed/anomaly_indeces/18k_spr_injected_1_words.txt",
                 1)
    # insert_words("/Users/haraldott/Development/thesis/detector/data/openstack/utah/parsed/18k_spr_corpus",
    #              "/Users/haraldott/Development/thesis/detector/data/openstack/utah/parsed/anomalies_injected/18k_spr_6_words_injected",
    #              "/Users/haraldott/Development/thesis/detector/data/openstack/utah/parsed/anomaly_indeces/18k_spr_6_words_injected.txt",
    #              6)
    # delete_or_duplicate_events(
    #     input_file_path = "/Users/haraldott/Development/thesis/detector/data/openstack/sasho/parsed/logs_aggregated_normal_only_spr_corpus",
    #     output_file_path = "/Users/haraldott/Development/thesis/detector/data/openstack/sasho/parsed/anomalies_injected/logs_aggregated_normal_only_spr_corpus_deleted_lines",
    #     anomaly_indices_output_path = "/Users/haraldott/Development/thesis/detector/data/openstack/sasho/parsed/anomaly_indeces/logs_aggregated_normal_only_spr_corpus_deleted_lines.txt",
    #     instance_information_path_in = "/Users/haraldott/Development/thesis/detector/data/openstack/sasho/raw/sorted_per_request_pickle/logs_aggregated_normal_only_spr.pickle",
    #     instance_information_path_out = "/Users/haraldott/Development/thesis/detector/data/openstack/sasho/raw/sorted_per_request_pickle/anomalies/logs_aggregated_normal_only_spr_deleted_lines.pickle",
    #     mode = "del")
