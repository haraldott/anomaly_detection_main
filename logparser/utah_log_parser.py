import re
import os
import pickle


def parse_utah_logs(logfile_path='../data/openstack/utah/raw/openstack_18k_anomalies',
                    output_path='../data/openstack/utah/raw/sorted_per_request/'
                                'openstack_18k_anomalies_sorted_per_request',
                    instance_information_path='../data/openstack/utah/raw/sorted_per_request/'
                                              'openstack_18k_anomalies_information.pickle'):
    logfile_lines = open(logfile_path, 'r').readlines()
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

    pickle.dump(instance_information, open(instance_information_path, 'wb'))
