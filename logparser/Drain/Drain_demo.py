#!/usr/bin/env python
import sys

sys.path.append('../../')
from logparser import Drain


def execute(dir='../../data/openstack/utah/raw/',
            file='openstack_18k_anomalies',
            output='../../data/openstack/utah/parsed/',
            logtype='OpenStack',
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
    }

    try:
        log_format = settings[logtype]["log_format"]
        regex = settings[logtype]["regex"]
        # depth = settings[log_type]["depth"]
        # st = settings[log_type]["st"]
    except ValueError:
        print("log format does not exist")
        raise

    parser = Drain.LogParser(log_format, indir=dir, outdir=output, depth=depth, st=st, rex=regex)
    parser.parse(file, full_output)
