# reference: https://github.com/swshon/dialectID_e2e/blob/master/scripts/augmentation_by_speed_vol.py

import sox
import sys
import numpy as np
import pandas as pd

def creat_vol_augmentation(filelist, source_folder, target_folder, vol_list):
    aug_generator = sox.Transformer()
    for volume in vol_list:
        aug_generator.vol(volume)
        for index,files in enumerate(filelist):
            save_filename = TARGET_FOLDER+files.split('.')[0]+'_'+str(volume)+'.'+files.split('.')[1] 
            print(save_filename)
            aug_generator.build(source_folder+files, save_filename)

def creat_speed_augmentation(filelist, source_folder, target_folder, speed_list):
    aug_generator = sox.Transformer()
    for speed in speed_list:
        aug_generator.speed(speed)
        for index,files in enumerate(filelist):
            save_filename = TARGET_FOLDER+files.split('.')[0]+'_'+str(speed)+'.'+files.split('.')[1] 
            print(save_filename)
            aug_generator.build(source_folder+files, save_filename)
        

def creat_speed_vol_augmentation(filelist, target_folder, speed, volume):
    aug_generator = sox.Transformer()
    aug_generator.vol(volume)
    aug_generator.speed(speed)
    for index,files in enumerate(filelist):
        save_filename = TARGET_FOLDER+filename.split('.')[0].split('/')[1] + '/'+ files.split('/')[-2]+'/s'+str(speed)+'_v'+str(volume)+'_'+files.split('/')[-1]
        aug_generator.build(files,save_filename)

# Paths
audio_folder    = '../wav/'
labels_file     = '../lab/labels.csv'
features_folder = '../features/'

# Define partition names (according to audio files)
partitions = ['train','devel','test']

# Load file list
instances = pd.read_csv(labels_file)['file_name']
filelist = [ins for ins in instances if 'train' in ins]


SPEED_LIST = [0.9, 1.0, 1.1]
VOL_LIST = [0.125, 1.0, 2.0]
TARGET_FOLDER = '../wav_aug_vol/'


creat_vol_augmentation(filelist, audio_folder, TARGET_FOLDER, VOL_LIST)
#creat_speed_vol_augmentation(filename, TARGET_FOLDER, SPEED_LIST, VOL_LIST)
#creat_speed_augmentation(filelist, audio_folder, TARGET_FOLDER, SPEED_LIST)
