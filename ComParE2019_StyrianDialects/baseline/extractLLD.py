import pandas as pd 
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler


task_name = 'ComParE2019_StyrianDialects'

# Configuration
feature_set = 'eGemaps-LLD'  # For all available options, see the dictionary feat_conf

# Mapping each available feature set to tuple (number of features, offset/index of first feature, separator, header option)
feat_conf = {'ComParE-LLD':      (130+1, 0, ';', 'infer'),
             'eGemaps-LLD':      (24+1,  0, ';', 'infer')}

# Augmentation
aug = True

num_feat = feat_conf[feature_set][0]
ind_off  = feat_conf[feature_set][1]
sep      = feat_conf[feature_set][2]
header   = feat_conf[feature_set][3]

# Path of the features and labels
features_path = '../features/'
features_aug = '../features_aug_vol/'
label_file    = '../lab/labels.csv'

# Load features and labels
X_train = pd.read_csv(features_aug + task_name + '.' + feature_set + '.train.csv', sep=sep, header=header, usecols=range(ind_off,num_feat+ind_off)).values
X_devel = pd.read_csv(features_path + task_name + '.' + feature_set + '.devel.csv', sep=sep, header=header, usecols=range(ind_off,num_feat+ind_off)).values
X_test  = pd.read_csv(features_path + task_name + '.' + feature_set + '.test.csv',  sep=sep, header=header, usecols=range(ind_off,num_feat+ind_off)).values

print('number of train data:', X_train.shape[0])

X_train_map = X_train[:,0]
X_train = np.array(X_train[:,2:], dtype=float)
X_devel_map = X_devel[:,0]
X_devel = np.array(X_devel[:,2:], dtype=float)
X_test_map = X_test[:,0]
X_test = np.array(X_test[:,2:], dtype=float)

# Feature normalisation
scaler  = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_devel = scaler.transform(X_devel)
X_test  = scaler.transform(X_test)
print(X_train[0].shape)

# Define labels
df_labels = pd.read_csv(label_file)
if aug:
    dict_label = {}
    for index, label in enumerate(df_labels['file_name']):
        dict_label[label] = df_labels['label'].values[index]
        #print(df_labels['label'].values[index])
    y_train = [dict_label[i[1:1+10]+'.wav'] for i in X_train_map]
    y_devel = df_labels['label'][df_labels['file_name'].str.startswith('devel')].values
else:
    y_train = df_labels['label'][df_labels['file_name'].str.startswith('train')].values
    y_devel = df_labels['label'][df_labels['file_name'].str.startswith('devel')].values

# convert into dict
def feature2dic(X, X_map, filename, aug=False):
    # input:
    #    X: all frames, 
    #    X_map: series of corresponding audio files
    #    filename: name of audio files 
    # output: 
    #   dictionary, key: file name, value: feauture
    dict_ = {}
    for i in filename:
        print(i)
        if aug:
            dict_[i] = X[np.where(X_map==i)]
        else:
            dict_[i] = X[np.where(X_map=="'{}'".format(i))]
    return dict_

dict_train = feature2dic(X_train, X_train_map, np.unique(X_train_map), True)
#dict_train = feature2dic(X_train, X_train_map, df_labels['file_name'][df_labels['file_name'].str.startswith('train'))
dict_devel = feature2dic(X_devel, X_devel_map,  df_labels['file_name'][df_labels['file_name'].str.startswith('devel')].values) 
dict_test = feature2dic(X_test, X_test_map, df_labels['file_name'][df_labels['file_name'].str.startswith('test')].values)

# save features
if aug:
    joblib.dump(dict_train, '../features/{}_v2/X_train.minmax.voice.pkl'.format(feature_set))
else:
    joblib.dump(dict_train, '../features/{}/X_train.minmax.pkl'.format(feature_set))
joblib.dump(dict_devel, '../features/{}/X_devel.minmax.pkl'.format(feature_set))
joblib.dump(dict_test, '../features/{}/X_test.minmax.pkl'.format(feature_set))


