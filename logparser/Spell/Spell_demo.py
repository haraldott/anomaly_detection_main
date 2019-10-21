#!/usr/bin/env python

# python Spell_demo.py -dir  -file  -logtype

import sys
sys.path.append('../../')
sys.path.append('../')
from logparser.Spell import Spell
import argparse

settings = {
    'HDFS': {
        'log_format': '<Date> <Time> <Pid> <Level> <Component>: <Content>',
        'tau': 0.7,
        'regex': [r'blk_-?\d+', r'(\d+\.){3}\d+(:\d+)?']
        },
    'OpenStack': {
        'log_format': '<Logrecord> <Date> <Time> <Pid> <Level> <Component> \[<ADDR>\] <Content>',
        'tau': 0.9,
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
parser.add_argument('-file', default='openstack_val_normal_n2', type=str)
parser.add_argument('-logtype', type=str, required=True)
args = parser.parse_args()

input_dir = args.dir  # The input directory of log file
log_file = args.file  # The input log file name
output_dir = 'Spell_result/'  # The output directory of parsing results
try:
    log_format = settings[args.logtype]["log_format"]
    tau = settings[args.logtype]["tau"]
    regex = settings[args.logtype]["regex"]
except ValueError:
    print("log format does not exist")
    raise
parser = Spell.LogParser(indir=input_dir, outdir=output_dir, log_format=log_format, tau=tau, rex=regex)
parser.parse(log_file)