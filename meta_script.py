import argparse
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('-inputdir', type=str, default='data/openstack/utah/raw/')
parser.add_argument('-inputfile', type=str, default='openstack_52k_normal')
parser.add_argument('-outputdir', type=str, default='data/openstack/utah/parsed/')
parser.add_argument('-logtype', default='OpenStack', type=str)
args = parser.parse_args()

subprocess.call(['python', 'logparser/Drain/Drain_demo.py',
                 '-dir', args.inputdir, '-file', args.inputfile, '-output', args.outputdir])


