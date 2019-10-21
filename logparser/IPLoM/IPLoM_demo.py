#!/usr/bin/env python

# python IPLoM_demo.py -dir  -file  -logtype

import sys
sys.path.append('../../')
from logparser import IPLoM
import argparse

settings = {
    'HDFS': {
        'log_format': '<Date> <Time> <Pid> <Level> <Component>: <Content>',
        'CT': 0.35,
        'lowerBound': 0.25,
        'regex': [r'blk_-?\d+', r'(\d+\.){3}\d+(:\d+)?']
        },
    'OpenStack': {
        'log_format': '<Logrecord> <Date> <Time> <Pid> <Level> <Component> \[<ADDR>\] <Content>',
        'CT': 0.9,
        'lowerBound': 0.25,
        'regex': [r'((\d+\.){3}\d+,?)+',
                  r'/.+?\s',
                  r'\d+',
                  r'\[.*?\]',
                  r'\[.*\]',
                  r'\[.*\] \[.*\]',
                  r'(/|)([0-9]+\.){3}[0-9]+(:[0-9]+|)(:|)',
                  r'(?<=[^A-Za-z0-9])(\-?\+?\d+)(?=[^A-Za-z0-9])|[0-9]+$',
                  r'\(\/.*\)'],
    },
}

parser = argparse.ArgumentParser()
parser.add_argument('-dir', default='/Users/haraldott/Development/thesis/logparser/logs/OpenStack/', type=str)
parser.add_argument('-file', default='openstack_val_anomalies', type=str)
parser.add_argument('-logtype', type=str, required=True)
args = parser.parse_args()

input_dir = args.dir  # The input directory of log file
log_file = args.file  # The input log file name

output_dir = 'IPLoM_result/'  # The output directory of parsing results
try:
    log_format = settings[args.logtype]["log_format"]
    regex = settings[args.logtype]["regex"]
    lower_bound = settings[args.logtype]["lower_bound"]
    CT = settings[args.logtype]["CT"]
except ValueError:
    print("log format does not exist")
    raise

parser = IPLoM.LogParser(log_format=log_format, indir=input_dir, outdir=output_dir, rex=regex,
                         CT=CT, lowerBound=lower_bound)
parser.parse(log_file)
