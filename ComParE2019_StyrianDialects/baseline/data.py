import joblib
import numpy as np
import pandas as pd
from sklearn import preprocessing
from collections import Counter

feature_set = 'eGemaps-LLD'
#feature_set = 'ComParE-LLD'
#feature_set = 'spec'

aug = True

# load features
if aug:
    X_train1 = joblib.load('../features/{}/X_train.minmax.speed.pkl'.format(feature_set))
    X_train2 = joblib.load('../features/{}/X_train.minmax.vol.pkl'.format(feature_set))
    for i in X_train1.items():
        if i[0] not in list(X_train2.keys()):
            X_train2[i[0]] = i[1]
    X_train_map = np.array(list(X_train2.keys()))
    X_train = np.array(list(X_train2.values()))
else:
    X_train = joblib.load('../features/{}/X_train.minmax.pkl'.format(feature_set))
    X_train_map = np.array(list(X_train.keys()))
    X_train = np.array(list(X_train.values()))

    
X_devel = joblib.load('../features/{}/X_devel.minmax.pkl'.format(feature_set))
X_test = joblib.load('../features/{}/X_test.minmax.pkl'.format(feature_set))

# Path of the labels
label_file    = '../lab/labels.csv'

# Load features and labels
df_labels = pd.read_csv(label_file)
X_devel = np.array([X_devel[i] for i in df_labels['file_name'][df_labels['file_name'].str.startswith('devel').values]])
X_test = np.array([X_test[i] for i in df_labels['file_name'][df_labels['file_name'].str.startswith('test').values]])

if aug:
    dict_label = {}
    for index, label in enumerate(df_labels['file_name']):
        dict_label[label] = df_labels['label'].values[index]
    y_train = [dict_label[i[1:1+10]+'.wav'] for i in X_train_map]
    y_train = np.array(y_train)
    y_devel = df_labels['label'][df_labels['file_name'].str.startswith('devel')].values
else:
    dict_label = {}
    for index, label in enumerate(df_labels['file_name']):
        dict_label[label] = df_labels['label'].values[index]
    y_train = [dict_label[i[:10]+'.wav'] for i in X_train_map]
    y_train = np.array(y_train)
    #y_train = df_labels['label'][df_labels['file_name'].str.startswith('train')].values
    y_devel = df_labels['label'][df_labels['file_name'].str.startswith('devel')].values

print(Counter(y_train))
y_train = y_train.reshape(len(y_train), 1)
y_devel = y_devel.reshape(len(y_devel), 1)

# convert into one-hot label
enc = preprocessing.OneHotEncoder(sparse=False)
y_train = enc.fit_transform(y_train)
y_devel = enc.transform(y_devel)

print('labels:', enc.inverse_transform([[1, 0, 0], [0, 1, 0], [0, 0, 1]])[:, 0])
print('Training data:', X_train.shape[0])

class BatchGenerator:
    def __init__(self, X, y, hparams, shuffle=True):
        self.hps = hparams
        D = self.hps.input_dim
        long_ = self.hps.seq_length
        batch_size = self.hps.BATCH_SIZE
        n = len(X)
        # shuffle
        if shuffle:
            p = np.random.permutation(n)
            X = X[p]
            y = y[p]
        self.batch_xs, self.batch_ys = [], []
        for i in range(n//batch_size+1):
            if i != (n//batch_size):
                batch_x = np.zeros((batch_size, long_, D))
                batch_y = np.zeros((batch_size, 3))
                for j in range(batch_size):
                    words = X[i*batch_size+j]
                    for k in range(len(words)):
                        batch_x[j][k] = words[k]
                        #print(k)
                    #for k in range(len(words)-1, long_):
                    #    batch_x[j][k] = np.zeros(D) # padding with 0
                    batch_y[j] = y[i*batch_size+j] # 1-hot vector
            else:
                batch_x = np.zeros((len(X) % batch_size, long_, D))
                batch_y = np.zeros((len(y) % batch_size, 3))
                for j in range((len(y) % batch_size)):
                    words = X[i*batch_size+j]
                    for k in range(len(words)):
                        batch_x[j][k] = words[k]

                    batch_y[j] = y[i*batch_size+j] # 1-hot vector
            self.batch_xs.append(batch_x)
            self.batch_ys.append(batch_y)

    def get(self, batch_id):
        return self.batch_xs[batch_id], self.batch_ys[batch_id]
