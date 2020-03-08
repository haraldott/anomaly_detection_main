#!/usr/bin/env python
import sys

sys.path.append('../../')
from logparser import Drain


def execute(directory='/Users/haraldott/Development/thesis/detector/data/openstack/sasho/raw/sorted_per_request',
            file='logs_aggregated_normal_only_spr.csv',
            output='/Users/haraldott/Development/thesis/detector/data/openstack/utah/parsed/',
            logtype='OpenStackSasho',
            st=0.2,
            depth=2,
            full_output=False):
    settings = {
        'HDFS': {
            'log_format': '<Date> <Time> <Pid> <Level> <Component>: <Content>',
            'regex': [r'blk_-?\d+', r'(\d+\.){3}\d+(:\d+)?'],
            'st': 0.5,
            'depth': 4
        },
        'OpenStack': {
            'log_format': '<Logrecord> <Date> <Time> <Pid> <Level> <Component> \[<ADDR>\] <Content>',
            'regex': [r'((\d+\.){3}\d+,?)+',
                      r'/.+?\s',
                      r'\d+',
                      r'\[.*?\]',
                      r'\[.*\]',
                      r'\[.*\] \[.*\]',
                      r'(/|)([0-9]+\.){3}[0-9]+(:[0-9]+|)(:|)',
                      r'(?<=[^A-Za-z0-9])(\-?\+?\d+)(?=[^A-Za-z0-9])|[0-9]+$',
                      r'\(\/.*\)'],
            'st': 0.2,
            'depth': 2
        },
        'OpenStackSasho': {
            'log_format': '<_id>,<_index>,<_score>,<_type>,<Hostname>,<user_id>,<project_domain>,<Timestamp>,<timestamp>,<log_level>,<Pid>,<Content>,<tenant_id>,<programname>,<request_id>,<python_module>,<Logger>,<user_domain>,<domain_id>,<http_status>,<http_method>,<http_version>,<http_url>,<chunk>,<next_retry_seconds>,<error>,<retry_time>,<message>,<chunk_id>,<worker>',
            'regex': [r'((\d+\.){3}\d+,?)+',
                      r'/.+?\s',
                      r'\d+',
                      r'\[.*?\]',
                      r'\[.*\]',
                      r'\[.*\] \[.*\]',
                      r'(/|)([0-9]+\.){3}[0-9]+(:[0-9]+|)(:|)',
                      r'(?<=[^A-Za-z0-9])(\-?\+?\d+)(?=[^A-Za-z0-9])|[0-9]+$',
                      r'\(\/.*\)'],
            'st': 0.2,
            'depth': 2
        },
    }

    try:
        log_format = settings[logtype]["log_format"]
        regex = settings[logtype]["regex"]
        # depth = settings[log_type]["depth"]
        # st = settings[log_type]["st"]
    except ValueError:
        print("log format does not exist")
        raise

    parser = Drain.LogParser(log_format, indir=directory, outdir=output, depth=depth, st=st, rex=regex)
    parser.parse(file, full_output)


if __name__ == "__main__":
    execute()
