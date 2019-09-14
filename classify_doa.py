# Train and test a classifier based on the within- and between- parcel rsfc across UO_TDS, OHSU_TSS, and OHSU_MINA studies
# T Cheng | 9/10/2019

import csv
import pyreadr
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut, train_test_split, GridSearchCV, cross_val_score
loo = LeaveOneOut()
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.datasets import base

# from sklearn.pipeline import Pipeline

### LOAD DATA ###

# Directory and file names
working_dir = '/Users/theresacheng/projects/doa_classification/'
subject_list_path = working_dir + 'data/inc_subj_full_ids.csv'
beh_path = working_dir + 'data/beh_data.RDS'
comm_conns_path = working_dir + 'data/comm_cors.csv'

# Load and clean subject id list, generate study and full_subject_ids vectors
with open(subject_list_path, newline='') as subject_id_file:
    reader=csv.reader(subject_id_file)
    full_subject_ids_raw = list(reader)

study=[]
full_subject_ids=[]

for a, b in full_subject_ids_raw[1:len(full_subject_ids_raw)]:
    study.append(a)
    full_subject_ids.append(b)

# Load behavioral data
beh = pyreadr.read_r(beh_path)[None]

# Load within and between community connectivity rsfc MRI data
comm_conns_file = open(comm_conns_path, "r+")
comm_conns = list(csv.reader(comm_conns_file, delimiter='\t'))

# Create dataframe for adversity classification
target_names = 'maltreatment'
feature_names = list(comm_conns[0])
data = list(comm_conns[1:len(comm_conns)])

# exclude subject ids from the data
target = np.asarray(beh.maltreatment)

classify_adv_df = base.Bunch(target_names=target_names,
                             feature_names=feature_names,
                             target=target,
                             data=data)

# Train and test classifier

# Create splits with leave-one-out cross validation
loo.get_n_splits(classify_adv_df.data)

for row in data[:2]:
    row.split(",")
    print(row)
    temp = row[:-1]
    print(temp)

for train_index, test_index in loo.split(classify_adv_df.data):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = classify_adv_df.data[train_index], classify_adv_df.data[test_index]
    y_train, y_test = classify_adv_df.target[train_index], classify_adv_df.target[test_index]
    print(X_train, X_test, y_train, y_test)

# Need to do this next: Create dataframe for exposure type analysis
# Note that fewer subjects are used here -- those with unspecified forms of maltreatment are removed

# Specify the hyperparameter space, test code here:
# parameters = {'SVM__C':[1, 10, 100],
#               'SVM__gamma':[0.1, 0.01]}
#
# c_space = np.logspace(-5, 8, 15)
# param_grid = {'C': c_space, 'penalty': ['l1', 'l2']}
# temp = GridSearchCV(svc, param_grid, cv = 5)

# fit models
svc = LinearSVC()

svc.fit(X_train, y_train)