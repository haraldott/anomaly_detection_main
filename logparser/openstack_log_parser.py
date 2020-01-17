import pandas as pd
from datetime import datetime
import json
import operator
import csv
import math

# def filter_timestamps(inpt='../data/openstack/sasho/raw/sequential_data/logs/logs_aggregated_sequential.csv',
#                       output='../data/openstack/sasho/raw/sequential_data/logs/logs_aggregated_sequential_filtered_new.csv'):
inpt = '../data/openstack/sasho/raw/concurrent data/logs/logs_aggregated_concurrent.csv'
output = '../data/openstack/sasho/raw/concurrent data/logs/logs_aggregated_concurrent_timefiltered.csv'

logfile = pd.read_csv(inpt)

datestr = '%Y-%m-%d %H:%M:%S.%f'
new_datestr = '%Y-%m-%dT%H:%M:%S.%f'

# sequential
# begin = datetime.strptime('2019-11-19 18:38:39.000', datestr)
# end = datetime.strptime('2019-11-20 02:30:00.000', datestr)

# concurrent
begin = datetime.strptime('2019-11-25 16:12:13.000', datestr)
end = datetime.strptime('2019-11-25 20:45:00.000', datestr)


# def filter_time_and_shorten_payload(l, idx, payload):
#     global logfile
#     current = datetime.strptime(l[0:-9], new_datestr)
#     print(idx)
#     if current < begin or current > end:
#         try:
#             logfile = logfile.drop(index=idx)
#         except KeyError:
#             pass
#     elif len(payload) > 100:
#         logfile.iloc[idx, 'Payload'] = payload[0:100]
#
#
# logfile.apply(lambda row: filter_time_and_shorten_payload(row["@timestamp"], row.name, row["Payload"]), axis=1)

for i, el in logfile.iterrows():
    print("{} {}".format(el["_id"], i))
    current = datetime.strptime(el["@timestamp"][0:-9], new_datestr)

    if current < begin or current > end:
        try:
            logfile = logfile.drop(index=i)
        except KeyError:
            print("Keyerror in line {}".format(i))
            continue
    else:
        payload = None
        try:
            payload = el["Payload"]
        except KeyError:
            payload = None
            pass
        if payload is not None:
            print("delete")
            if len(str(el["Payload"])) > 100:
                logfile.at[i, 'Payload'] = payload[0:100]

logfile.to_csv(output, index=False)


def sort_logfile(
        inpt='../data/openstack/sasho/raw/sequential_data/logs/logs_aggregated_sequential_filtered.csv',
        output='../data/openstack/sasho/raw/sequential_data/logs/logs_aggregated_sequential_filtered_sorted.csv'):
    lfile = csv.reader(open(inpt), delimiter=",")

    header = next(lfile)
    datestr = '%Y-%m-%dT%H:%M:%S.%f'
    sorted_list = sorted(lfile, key=lambda row: datetime.strptime(row[8][0:-9], datestr))
    with open(output, 'w') as file:
        for el in header:
            file.write(el + ',')
        file.write('\n')
        for line in sorted_list:
            for el in line:
                file.write(el + ',')
            file.write('\n')


def filter_anomalies(inpt='../data/openstack/sasho/raw/sequential_data/logs/logs_aggregated_concurrent_timefiltered_payload_sorted',
                     anomalies_timestamps='data/openstack/sasho/raw/anomalies_concurrent/output_boot.csv',
                     output_anomalies='../data/openstack/sasho/raw/sequential_data/logs/logs_aggregated_anomalies_only.csv',
                     output_normal='../data/openstack/sasho/raw/sequential_data/logs/logs_aggregated_normal_only.csv',
                     output_anomalies_indices='../data/openstack/sasho/raw/sequential_data/logs/anomaly_indices.txt'):
    logfile = pd.read_csv(inpt)
    anomaly_timestamps_file = pd.read_csv(anomalies_timestamps)

    datestr_for_anomaly_file = '%Y-%m-%d %H:%M:%S.%f'
    datestr_for_logs = '%Y-%m-%dT%H:%M:%S.%f'

    anomaly_indices = []

    anomaly_intervals = []
    for line in anomaly_timestamps_file.iterrows():
        anomaly_intervals.append(tuple((line["startTime"], line["endTime"])))

    anomaly_part_len = math.floor(len(logfile.index)/4)
    normal_part_len = len(logfile.index) - anomaly_part_len

    normal_slice = logfile[0:normal_part_len]
    anomaly_slice = logfile[normal_part_len::]

    for i, row in normal_slice.iterrows():
        current = datetime.strptime(row["@timestamp"], datestr_for_logs)
        for interval_start, interval_end in anomaly_intervals:
            if interval_start < current < interval_end:
                normal_slice.drop(index=i)

    for i, row in anomaly_slice.iterrows():
        current = datetime.strptime(row["@timestamp"], datestr_for_logs)
        for interval_start, interval_end in anomaly_intervals:
            if interval_start < current < interval_end:
                anomaly_indices.append(i)

    anomaly_slice.to_csv(output_anomalies)
    normal_slice.to_csv(output_normal)

    anomaly_indices_file = open(output_anomalies_indices, 'w+')
    for t in anomaly_indices:
        anomaly_indices_file.write(t + "\n")



# def filter_anomalies(input_csv='../data/openstack/sasho/raw/sequential_data/logs/logs_aggregated_sequential.csv',
#                      anomalies_json='../data/openstack/sasho/raw/new_sequential_data/traces/Reports/j_boot_delete_report.json',
#                      output='../data/openstack/sasho/raw/sequential_data/logs/logs_aggregated_sequential_filtered.csv'):
# anomalies_csv_file = pd.read_csv('../data/openstack/sasho/raw/sequential_data/logs/logs_aggregated_sequential.csv')
# json_file = open('../data/openstack/sasho/raw/new_sequential_data/traces/Reports/j_boot_delete_report.json', 'r')
# json_data = json.load(json_file)


# inpt = '../data/openstack/sasho/raw/concurrent data/logs/logs_aggregated_concurrent_2.csv'
# output = '../data/openstack/sasho/raw/concurrent data/logs/logs_aggregated_concurrent_timefiltered.csv'
#
# logfile = csv.DictReader(inpt, delimeter=',')
#
# datestr = '%Y-%m-%d %H:%M:%S.%f'
# new_datestr = '%Y-%m-%dT%H:%M:%S.%f'
#
# # sequential
# # begin = datetime.strptime('2019-11-19 18:38:39.000', datestr)
# # end = datetime.strptime('2019-11-20 02:30:00.000', datestr)
#
# # concurrent
# begin = datetime.strptime('2019-11-25 16:12:13.000', datestr)
# end = datetime.strptime('2019-11-25 20:45:00.000', datestr)
#
# for i, row in enumerate(logfile):
#     if i == 0:
#         continue
#     current = datetime.strptime(row['@timestamp'][0:-9], new_datestr)
#     if current < begin or current > end:
#         del row[i]
#     elif len(row["Payload"]) > 100:
#         row["Payload"] =
#         logfile.iloc[idx, 'Payload'] = payload[0:100]
#
#
# def filter_time_and_shorten_payload(l, idx, payload):
#     global logfile
#     current = datetime.strptime(l[0:-9], new_datestr)
#     if current < begin or current > end:
#         logfile = logfile.drop(index=idx)
#     if len(payload) > 100:
#         logfile.iloc[idx, 'Payload'] = payload[0:100]
#
#
# logfile.apply(lambda row: filter_time_and_shorten_payload(row["@timestamp"], row.name, row["Payload"]), axis=1)
#
# logfile.to_csv(output, index=False)