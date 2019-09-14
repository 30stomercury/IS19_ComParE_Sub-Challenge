#!/usr/bin/python
# Tested with python 2.7

import os
import pandas as pd
from glob import glob

# Modify openSMILE paths HERE:
SMILEpath = '../../../../opensmile-2.3.0/bin/linux_x64_standalone_static/SMILExtract'
#SMILEconf = '../../../../opensmile-2.3.0/config/ComParE_2016.conf'
SMILEconf = '../../../../opensmile-2.3.0/config/gemaps/eGeMAPSv01a.conf'

# Task name
task_name = os.getcwd().split('/')[-2]  # 'ComParE2019_XXX'

# Paths
audio_folder    = '../wav_aug_vol/'
labels_file     = '../lab/labels.csv'
#features_folder = '../features/'
features_folder = '../features_aug_vol/'

# Define partition names (according to audio files)
if 'aug' in audio_folder:
    partitions = ['train']
else:
    partitions = ['train','devel','test']

# Load file list
if 'aug' in audio_folder:
    instances = glob('../wav_aug_vol/*.wav')
    instances = [i.split('/')[-1] for i in instances]
else:
    instances = pd.read_csv(labels_file)['file_name']

# Iterate through partitions and extract features
for part in partitions:
    instances_part = [i for i in instances if part in i]
    #output_file      = features_folder + task_name + '.ComParE.'      + part + '.csv'
    #output_file_lld  = features_folder + task_name + '.ComParE-LLD.'  + part + '.csv'
    output_file      = features_folder + task_name + '.eGemaps.'      + part + '.csv'
    output_file_lld  = features_folder + task_name + '.eGemaps-LLD.'  + part + '.csv'

    if os.path.exists(output_file):
        os.remove(output_file)
    if os.path.exists(output_file_lld):
        os.remove(output_file_lld)
    # Extract openSMILE features for the whole partition (standard ComParE and LLD-only)
    for inst in instances_part:
        os.system(SMILEpath + ' -C ' + SMILEconf + ' -I ' + audio_folder + inst + \
        ' -instname ' + inst + ' -csvoutput '+ output_file + ' -timestampcsv 0 -lldcsvoutput ' + output_file_lld + ' -appendcsvlld 1')

