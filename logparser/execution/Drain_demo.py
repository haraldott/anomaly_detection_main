#!/usr/bin/env python
import sys

# python Drain_demo.py -dir  -file  -logtype

sys.path.append('../../')
from logparser.implementations import Drain
import argparse

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
        'st': 0.5,
        'depth': 5
        },
}

parser = argparse.ArgumentParser()
parser.add_argument('-dir', default='/Users/haraldott/Development/thesis/logparser/logs/OpenStack/', type=str)
parser.add_argument('-file', default='openstack_val_anomalies', type=str)
parser.add_argument('-logtype', type=str, required=True)
args = parser.parse_args()

input_dir = args.dir  # The input directory of log file
log_file = args.file  # The input log file name

output_dir = 'Drain_result/'  # The output directory of parsing results
try:
    log_format = settings[args.logtype]["log_format"]
    tau = settings[args.logtype]["tau"]
    regex = settings[args.logtype]["regex"]
    depth = settings[args.logtype]["depth"]
    st = settings[args.logtype]["st"]
except ValueError:
    print("log format does not exist")
    raise

parser = Drain.LogParser(log_format, indir=input_dir, outdir=output_dir, depth=depth, st=st, rex=regex)
parser.parse(log_file)
