import math

import pandas as pd


def filter_timestamps(inpt='../data/openstack/sasho/raw/concurrent data/logs/logs_aggregated_concurrent.csv',
                      output='../data/openstack/sasho/raw/concurrent data/logs/logs_aggregated_concurrent_filtered.csv'):
    logfile = pd.read_csv(inpt)

    datestr = '%Y-%m-%d %H:%M:%S.%f'
    new_datestr = '%Y-%m-%dT%H:%M:%S.%f'

    # sequential
    # begin = datetime.strptime('2019-11-19 18:38:39.000', datestr)
    # end = datetime.strptime('2019-11-20 02:30:00.000', datestr)

    # concurrent
    begin = pd.to_datetime('2019-11-25 16:12:13.000000', format=datestr)
    end = pd.to_datetime('2019-11-25 20:45:00.000000', format=datestr)

    def filter_time_and_shorten_payload(l, idx, payload, df):
        if l.endswith('+01:00'):
            l = l[:-6]
        print("idx: {}, l: {}".format(idx, l))
        current = pd.to_datetime(l, format=new_datestr)
        if current < begin or current > end:
            try:
                df.drop(index=idx, inplace=True)
            except KeyError:
                pass
        elif len(str(payload)) > 100:
            df.loc[idx, 'Payload'] = payload[0:100]

    logfile.apply(lambda row: filter_time_and_shorten_payload(row["@timestamp"], row.name, row["Payload"], logfile), axis=1)
    logfile.to_csv(output, index=False)

def sort_logfile(
        inpt='../data/openstack/sasho/raw/concurrent data/logs/logs_aggregated_concurrent_filtered.csv',
        output='../data/openstack/sasho/raw/concurrent data/logs/logs_aggregated_concurrent_filtered_sorted.csv'):
    lfile = pd.read_csv(inpt)
    indices = pd.to_datetime(lfile['@timestamp'].str[:-6], format='%Y-%m-%dT%H:%M:%S.%f').sort_values().index
    lfile = lfile.loc[indices]

    # changes format
    # lfile['@timestamp'] = pd.to_datetime(lfile['@timestamp'].str[:-6], format='%Y-%m-%dT%H:%M:%S.%f')
    # lfile.sort_values(by='@timestamp', inplace=True)
    # changes format

    lfile.to_csv(output, index=False)


def filter_anomalies(
        inpt='../data/openstack/sasho/raw/concurrent data/logs/logs_aggregated_concurrent_filtered_sorted.csv',
        anomalies_timestamps='../data/openstack/sasho/raw/concurrent data/anomalies_concurrent/output_boot.csv',
        output_anomalies='../data/openstack/sasho/raw/concurrent data/logs/logs_aggregated_anomalies_only.csv',
        output_normal='../data/openstack/sasho/raw/concurrent data/logs/logs_aggregated_normal_only.csv',
        output_anomalies_indices='../data/openstack/sasho/raw/concurrent data/logs/anomaly_indices.txt'):


    logfile = pd.read_csv(inpt)
    anomaly_timestamps_file = pd.read_csv(anomalies_timestamps)
    anomaly_timestamps_file['startTime'] = pd.to_datetime(anomaly_timestamps_file['startTime'])\
        .dt.tz_localize('UTC').dt.tz_convert('UTC')
    anomaly_timestamps_file['endTime'] = pd.to_datetime(anomaly_timestamps_file['endTime'])\
        .dt.tz_localize('UTC').dt.tz_convert('UTC')

    datestr_for_anomaly_file = '%Y-%m-%d %H:%M:%S.%f'
    datestr_for_logs = '%Y-%m-%dT%H:%M:%S.%f'

    anomaly_indices = []

    anomaly_intervals = []
    for line in anomaly_timestamps_file.iterrows():
        start = pd.to_datetime(line[1]["startTime"], format=datestr_for_anomaly_file)
        end = pd.to_datetime(line[1]["endTime"], format=datestr_for_anomaly_file)
        tup = (start, end)
        anomaly_intervals.append(tup)

    anomaly_part_len = math.floor(len(logfile.index) / 4)
    normal_part_len = len(logfile.index) - anomaly_part_len

    normal_slice = logfile[0:normal_part_len]
    anomaly_slice = logfile[normal_part_len::]

    for i, row in normal_slice.iterrows():
        current = pd.to_datetime(row["@timestamp"], format=datestr_for_logs)
        for interval_start, interval_end in anomaly_intervals:
            if interval_start < current < interval_end:
                normal_slice.drop(index=i)

    for i, row in anomaly_slice.iterrows():
        current = pd.to_datetime(row["@timestamp"], format=datestr_for_logs)
        for interval_start, interval_end in anomaly_intervals:
            if interval_start < current < interval_end:
                anomaly_indices.append(i)

    anomaly_slice.to_csv(output_anomalies)
    normal_slice.to_csv(output_normal)

    anomaly_indices_file = open(output_anomalies_indices, 'w+')
    for t in anomaly_indices:
        anomaly_indices_file.write(str(t) + "\n")
    anomaly_indices_file.close()
