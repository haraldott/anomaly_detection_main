import re
import os
import pickle
import random
from collections import defaultdict

randomly_injected_line_utah = "nova-compute.log.2017-05-14_21:27:09 2017-05-14 19:39:22.577 2931 INFO nova.compute.manager [req-3ea4052c-895d-4b64-9e2d-04d64c4d94ab - - - - -] [instance: {}] My personal randomly injected line.\n"
number_of_instances_to_inject_anomalies_in = 6
max_number_of_anomalies_per_instance = 3


def parse_and_sort(logfile_path='../data/openstack/sasho/raw/logs_aggregated_normal_only.csv',
                   output_path='../data/openstack/sasho/raw/sorted_per_request/logs_aggregated_normal_only.csv',
                   instance_information_path='../data/openstack/sasho/raw/sorted_per_request_pickle/logs_aggregated_normal_only.pickle'):
    instance_id_dict = sort_per_instance_id(logfile_path)
    create_instance_intervals(instance_id_dict, output_path, instance_information_path)


def parse_sort_and_inject_anomalies(logfile_path='../data/openstack/utah/raw/openstack_18k_anomalies',
                                    output_path='../data/openstack/utah/raw/sorted_per_request/openstack_18k_self_injected_anomalies',
                                    instance_information_path='../data/openstack/utah/raw/sorted_per_request_pickle/openstack_18k_self_injected_anomalies.pickle',
                                    anomaly_indices_output_path='../data/openstack/utah/raw/sorted_per_request/anomaly_indices_18k_self_injected.txt'):
    instance_id_dict = sort_per_instance_id(logfile_path)
    instance_id_dict, line_numbers_containing_anomalies_per_instance = inject_anomalies(instance_id_dict)
    create_instance_intervals_with_anomalies(instance_id_dict,
                                             output_path,
                                             instance_information_path,
                                             line_numbers_containing_anomalies_per_instance,
                                             anomaly_indices_output_path)


def sort_per_instance_id(log_lines_path):
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


def inject_anomalies(instance_id_dict):
    instance_ids_selected_for_anomaly_injection = random.sample(list(instance_id_dict.keys()),
                                                                number_of_instances_to_inject_anomalies_in)
    line_numbers_containing_anomalies_per_instance = defaultdict(list)
    for instance_id in instance_ids_selected_for_anomaly_injection:
        for _ in range(0, max_number_of_anomalies_per_instance):
            line_number_of_anomaly = random.randrange(len(instance_id_dict[instance_id]) + 1)
            instance_id_dict[instance_id].insert(line_number_of_anomaly, randomly_injected_line_utah.format(instance_id))
            line_numbers_containing_anomalies_per_instance[instance_id].append(line_number_of_anomaly)

    return instance_id_dict, line_numbers_containing_anomalies_per_instance


def create_instance_intervals_with_anomalies(instance_id_dict,
                                             output_path,
                                             instance_information_path,
                                             line_numbers_containing_anomalies_per_instance,
                                             anomaly_indices_output_path):
    anomaly_indices = []
    linecounter = -1
    instance_information = []
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    output_file = open(output_path, 'w')
    for inst_id in instance_id_dict:
        this_list = instance_id_dict.get(inst_id)
        for line in this_list:
            output_file.write(line)
        begin_line = linecounter + 1
        linecounter += len(this_list)
        if inst_id in line_numbers_containing_anomalies_per_instance:
            for index in line_numbers_containing_anomalies_per_instance[inst_id]:
                anomaly_indices.append(begin_line + index)
        instance_information.append(tuple((begin_line, linecounter)))

    os.makedirs(os.path.dirname(instance_information_path), exist_ok=True)
    pickle.dump(instance_information, open(instance_information_path, 'wb'))
    anomaly_indices_output_file = open(anomaly_indices_output_path, 'w')
    for val in anomaly_indices:
        anomaly_indices_output_file.write(str(val+1)+"\n")
    anomaly_indices_output_file.close()


def create_instance_intervals(instance_id_dict, output_path, instance_information_path):
    linecounter = -1
    instance_information = []
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    output_file = open(output_path, 'w')
    for inst_id in instance_id_dict:
        this_list = instance_id_dict.get(inst_id)
        for line in this_list:
            output_file.write(line)
        begin_line = linecounter + 1
        linecounter += len(this_list)
        instance_information.append(tuple((begin_line, linecounter)))

    os.makedirs(os.path.dirname(instance_information_path), exist_ok=True)
    pickle.dump(instance_information, open(instance_information_path, 'wb'))


# if __name__ == '__main__':
#     parse_utah_logs()
