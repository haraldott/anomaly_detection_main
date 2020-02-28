import re
import os
import pickle
import random
import math
from collections import defaultdict

randomly_injected_line_utah = "nova-compute.log.2017-05-14_21:27:09 2017-05-14 19:39:22.577 2931 INFO nova.compute.manager [req-3ea4052c-895d-4b64-9e2d-04d64c4d94ab - - - - -] [instance: {}] My personal randomly injected line.\n"
number_of_instances_to_inject_anomalies_in = 6
max_number_of_anomalies_per_instance = 3
number_of_swaps_per_instance = 4
anomaly_ratio = 0.02
words_for_random_insert = ["time <*>", "for", "when", "during <*>", "deleted", "random", "bullshit", "this", "after"]
max_number_of_words_to_be_altered = 1
ratio_of_words_to_be_altered_per_line = 0.15


def parse_and_sort(logfile_path='../data/openstack/sasho/raw/logs_aggregated_normal_only.csv',
                   output_path='../data/openstack/sasho/raw/sorted_per_request/logs_aggregated_normal_only.csv',
                   instance_information_path='../data/openstack/sasho/raw/sorted_per_request_pickle/logs_aggregated_normal_only.pickle'):
    instance_id_dict = __sort_per_instance_id(logfile_path)
    __create_instance_intervals(instance_id_dict, output_path, instance_information_path)


def parse_sort_and_inject_anomalies(logfile_path='../data/openstack/utah/raw/openstack_18k_anomalies',
                                    output_path='../data/openstack/utah/raw/sorted_per_request/openstack_18k_self_injected_anomalies',
                                    instance_information_path='../data/openstack/utah/raw/sorted_per_request_pickle/openstack_18k_self_injected_anomalies.pickle',
                                    anomaly_indices_output_path='../data/openstack/utah/raw/sorted_per_request/anomaly_indices_18k_self_injected.txt'):
    instance_id_dict = __sort_per_instance_id(logfile_path)
    instance_id_dict, line_numbers_containing_anomalies_per_instance = __inject_anomalies(instance_id_dict)
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
    instance_id_dict = defaultdict
    for line in logfile_lines:
        m = re.search('\[instance:\s([^\]]+)', line)
        if m is not None:
            instance_id = m.group(1)
            instance_id_dict[instance_id].append(line)
        else:
            number_of_no_id += 1
    return instance_id_dict


def __inject_anomalies(instance_id_dict):
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
        for line in this_list:
            output_file.write(line)
        instance_block_begin_line = linecounter
        linecounter += len(this_list)
        instance_information.append(tuple((instance_block_begin_line, linecounter - 1)))

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


def change_words(input_file_path, output_file_path, anomaly_indices_output_path):
    total_lines_file = open(input_file_path, 'r')
    total_lines = total_lines_file.readlines()
    total_lines_file.close()

    number_of_lines = len(total_lines)
    # number_of_lines = sum([len(instance_id_dict[x]) for x in instance_id_dict])
    number_of_anomaly_lines_to_be_manipulated = math.floor(number_of_lines * anomaly_ratio)
    line_indices_to_be_altered = random.sample(range(len(total_lines)), number_of_anomaly_lines_to_be_manipulated)
    fns = ["remove", "insert"]
    for index in line_indices_to_be_altered:
        # select line for altering
        line = total_lines[index]
        line = line.split()
        number_of_words_to_be_altered = max(math.ceil(len(line) * ratio_of_words_to_be_altered_per_line), 1)
        choice = random.choice(fns)
        for _ in range(number_of_words_to_be_altered):
            if choice == "insert":
                line.insert(random.randrange(0, len(total_lines[index])), random.choice(words_for_random_insert))
            if choice == "remove":
                del line[random.randrange(0, len(line))]
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


if __name__ == '__main__':
    change_words(input_file_path="../data/openstack/utah/parsed/openstack_18k_anomalies_sorted_per_request_corpus",
                 output_file_path="/Users/haraldott/Downloads/anomalies.txt",
                 anomaly_indices_output_path="/Users/haraldott/Downloads/anomaly_indices.txt")
