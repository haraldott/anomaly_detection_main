#!/usr/bin/env python
import sys

# python Drain_demo.py -dir  -file  -logtype

sys.path.append('../../')
from logparser import Drain
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
parser.add_argument('-dir', default='/Users/haraldott/Development/thesis/anomaly_detection_main/data/openstack', type=str)
parser.add_argument('-file', default='openstack_137k_normal', type=str)
parser.add_argument('-logtype', default='OpenStack', type=str)
parser.add_argument('-st', type=float)
parser.add_argument('-depth', type=int)
args = parser.parse_args()

input_dir = args.dir  # The input directory of log file
log_file = args.file  # The input log file name
log_type = args.logtype
st = args.st
depth = args.depth

output_dir = 'Drain_result/st_' + str(st) + ' depth_' + str(depth)  # The output directory of parsing results
try:
    log_format = settings[log_type]["log_format"]
    regex = settings[log_type]["regex"]
    #depth = settings[log_type]["depth"]
    #st = settings[log_type]["st"]
except ValueError:
    print("log format does not exist")
    raise

parser = Drain.LogParser(log_format, indir=input_dir, outdir=output_dir, depth=depth, st=st, rex=regex)
parser.parse(log_file)
