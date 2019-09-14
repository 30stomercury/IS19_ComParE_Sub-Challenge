#!/usr/bin/python
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn import svm
from sklearn.metrics import recall_score, confusion_matrix
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectPercentile
from sklearn.calibration import CalibratedClassifierCV
# Task
task_name = 'ComParE2019_StyrianDialects'
classes   = ['NorthernS','UrbanS','EasternS']

# Enter your team name HERE
team_name = 'baseline'

# Enter your submission number HERE
submission_index = 1

# Option
show_confusion = True   # Display confusion matrix on devel
feat_selection = False

# Configuration
feature_set = 'auDeep-50'  # For all available options, see the dictionary feat_conf
complexities = [1e-5,1e-4,1e-3,1e-2,1e-1,1e0]  # SVM complexities (linear kernel)


# Mapping each available feature set to tuple (number of features, offset/index of first feature, separator, header option)
feat_conf = {'ComParE':      (6373, 1, ';', 'infer'),
             'eGemaps':      (88,   1, ';', 'infer'),
             'BoAW-125':     ( 250, 1, ';',  None),
             'BoAW-250':     ( 500, 1, ';',  None),
             'BoAW-500':     (1000, 1, ';',  None),
             'BoAW-1000':    (2000, 1, ';',  None),
             'BoAW-2000':    (4000, 1, ';',  None),
             'auDeep-40':    (1024, 2, ',', 'infer'),
              'auDeep-50':    (1024, 2, ',', 'infer'),
             'auDeep-60':    (1024, 2, ',', 'infer'),
             'auDeep-70':    (1024, 2, ',', 'infer'),
             'auDeep-fused': (4096, 2, ',', 'infer')}
num_feat = feat_conf[feature_set][0]
ind_off  = feat_conf[feature_set][1]
sep      = feat_conf[feature_set][2]
header   = feat_conf[feature_set][3]

# Path of the features and labels
features_path = '../features/'
label_file    = '../lab/labels.csv'

# Start
print('\nRunning ' + task_name + ' ' + feature_set + ' baseline ... (this might take a while) \n')

# Load features and labels
X_train = pd.read_csv(features_path + task_name + '.' + feature_set + '.train.csv', sep=sep, header=header, usecols=range(ind_off,num_feat+ind_off), dtype=np.float32).values
X_devel = pd.read_csv(features_path + task_name + '.' + feature_set + '.devel.csv', sep=sep, header=header, usecols=range(ind_off,num_feat+ind_off), dtype=np.float32).values
X_test  = pd.read_csv(features_path + task_name + '.' + feature_set + '.test.csv',  sep=sep, header=header, usecols=range(ind_off,num_feat+ind_off), dtype=np.float32).values
print(X_train.shape)
df_labels = pd.read_csv(label_file)
y_train = df_labels['label'][df_labels['file_name'].str.startswith('train')].values
y_devel = df_labels['label'][df_labels['file_name'].str.startswith('devel')].values

# (No concatenation of training and development sets for final training - development set much smaller than training set)

# Feature normalisation
scaler  = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_devel = scaler.transform(X_devel)
X_test  = scaler.transform(X_test)

# Train SVM model with different complexities and evaluate
uar_scores = []
for comp in complexities:
    print('\nComplexity {0:.6f}'.format(comp))
    clf = svm.LinearSVC(C=comp, random_state=0)
    #clf = svm.SVC(C=comp, kernel='linear', random_state=0)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_devel)
    uar_scores.append( recall_score(y_devel, y_pred, labels=classes, average='macro') )
    print('UAR on Devel {0:.1f}'.format(uar_scores[-1]*100))
    if show_confusion:
        print('Confusion matrix (Devel):')
        print(classes)
        print(confusion_matrix(y_devel, y_pred, labels=classes))

# Train SVM model on the whole training data with optimum complexity and get predictions on test data
optimum_complexity = complexities[np.argmax(uar_scores)]
print('\nOptimum complexity: {0:.6f}, maximum UAR on Devel {1:.1f}\n'.format(optimum_complexity, np.max(uar_scores)*100))

if not feat_selection:
    linear_svc = svm.LinearSVC(C=optimum_complexity, random_state=0)
    clf = CalibratedClassifierCV(linear_svc, cv=5)
    #clf = svm.LinearSVC(C=optimum_complexity, random_state=0)
    clf.fit(X_train, y_train)
    #y_margins = clf.decision_function(X_devel)
    '''
    y_prob = (y_margins - y_margins.min()) / (y_margins.max() - y_margins.min())
    y_prob = 1./(1 + np.exp(-y_margins))
    '''
    y_prob_devel = clf.predict_proba(X_devel)
    y_prob_test = clf.predict_proba(X_test)
    y_pred = clf.predict(X_test)
    np.save('./predictions/SD_devel_svm_baseline.npy', y_prob_devel)
    np.save('./predictions/SD_test_svm_baseline.npy', y_prob_test)
else:
    uar = []
    percentile = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    for p in percentile:
        selection = SelectPercentile(f_classif, percentile=p)
        feat_selected = selection.fit_transform(X_train, y_train)
        feat_devel = selection.transform(X_devel)
        print('\nComplexity {0:.6f}'.format(optimum_complexity))
        #clf = svm.LinearSVC(C=optimum_complexity, random_state=0)
        clf = svm.SVC(C=optimum_complexity, kernel='linear', random_state=0)
        clf.fit(feat_selected, y_train)
        y_pred = clf.predict(feat_devel)
        uar.append( recall_score(y_devel, y_pred, labels=classes, average='macro') )
        print('UAR on Devel {0:.1f}'.format(uar[-1]*100))
        if show_confusion:
            print('Confusion matrix (Devel):')
            print(classes)
            print(confusion_matrix(y_devel, y_pred, labels=classes))
    optimum_percentile = percentile[np.argmax(uar)]
    print('\nOptimum percentile: {0:.6f}, maximum UAR on Devel {1:.1f}\n'.format(optimum_percentile, np.max(uar)*100))

    selection = SelectPercentile(f_classif, percentile=optimum_percentile)
    feat_selected = selection.fit_transform(X_train, y_train)
    feat_devel = selection.transform(X_devel)
    feat_test = selection.transform(X_test)
    clf = svm.LinearSVC(C=optimum_complexity, random_state=0)
    #clf = svm.SVC(C=optimum_complexity, kernel='linear', random_state=0)
    clf.fit(feat_selected, y_train)
    y_pred = clf.predict(feat_test)

"""
# Write out predictions to csv file (official submission format)
pred_file_name = task_name + '.test.' + team_name + '_' + str(submission_index) + '.csv'
df = pd.DataFrame(data={'file_name': df_labels['file_name'][df_labels['file_name'].str.startswith('test')].values,
                        'prediction': y_pred.flatten()},
                  columns=['file_name','prediction'])
df.to_csv(pred_file_name, index=False)
print('U:', len(y_pred[y_pred=='UrbanS']))
print('N:', len(y_pred[y_pred=='NorthernS']))
print('E:', len(y_pred[y_pred=='EasternS']))
"""
print('Done.\n')
