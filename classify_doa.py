# Train and test a classifier based on the within- and between- parcel rsfc across UO_TDS, OHSU_TSS, and OHSU_MINA studies
# T Cheng | 9/10/2019

import csv
import pyreadr
import numpy as np
import statistics as stats

from sklearn.model_selection import LeaveOneOut, train_test_split, GridSearchCV, cross_val_score
from sklearn.utils import Bunch

loo = LeaveOneOut()
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.datasets import base

import matplotlib.pyplot as plt
import pandas as pd

# from sklearn.pipeline import Pipeline

run_permutation = False

### LOAD DATA ###

# Directory and file names
working_dir = '/Users/theresacheng/projects/doa_classification/'
subject_list_path = working_dir + 'data/inc_subj_full_ids.csv'
beh_path = working_dir + 'data/beh_data.RDS'
comm_conns_path = working_dir + 'data/comm_cors.csv'

# Load and clean subject id list, generate study and full_subject_ids vectors
with open(subject_list_path, newline='') as subject_id_file:
    reader = csv.reader(subject_id_file)
    full_subject_ids_raw = list(reader)

study = []
full_subject_ids = []

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
raw_data = list(comm_conns[1:len(comm_conns)])

data = []
for row in raw_data:
    row_as_list_subj_id = row[0].split(",")  # there are 92 items per row
    row_as_list = row_as_list_subj_id[0:90]
    row_as_list = list(map(float, row_as_list))
    data.append(row_as_list)
data = np.asarray(data)

# exclude subject ids from the data
target = np.asarray(beh.maltreatment)

classify_adv_df = base.Bunch(target_names=target_names,
                             feature_names=feature_names,
                             target=target,
                             data=data)

# Train and test adversity classifier

# Specify the hyperparameter space, test code here:
# parameters = {'SVM__C':[1, 10, 100],
#               'SVM__gamma':[0.1, 0.01]}
#
# c_space = np.logspace(-5, 8, 15)
# param_grid = {'C': c_space, 'penalty': ['l1', 'l2']}
# temp = GridSearchCV(svc, param_grid, cv = 5)

# Create splits with leave-one-out cross validation
loo.get_n_splits(classify_adv_df.data)

# fit models
svc = LinearSVC()
y_pred_all = []
y_test_all = []

for train_index, test_index in loo.split(classify_adv_df.data):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = classify_adv_df.data[train_index], classify_adv_df.data[test_index]
    y_train, y_test = classify_adv_df.target[train_index], classify_adv_df.target[test_index]
    svc.fit(X_train, y_train)
    y_pred_all.append(svc.predict(X_test))
    y_test_all.append(y_test)

adv_tn, adv_fp, adv_fn, adv_tp = confusion_matrix(y_test_all, y_pred_all).ravel()
pred_adv_accuracy = (adv_tn + adv_tp)/(adv_tn + adv_tp + adv_fn + adv_fp)

# Conduct control analyses
all_adv_accuracies = []

# re-run classifier 1000x with shuffled target labels and save accuracy values
if run_permutation == True:
    for i in range(1000):
        idx = np.random.permutation(len(classify_adv_df.target))
        y = classify_adv_df.target[idx]

        shuffled_adv_df = classify_adv_df
        shuffled_adv_df.target = y

        y_pred_all = []
        y_test_all = []

        for train_index, test_index in loo.split(shuffled_adv_df.data):
            print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = shuffled_adv_df.data[train_index], shuffled_adv_df.data[test_index]
            y_train, y_test = shuffled_adv_df.target[train_index], shuffled_adv_df.target[test_index]
            svc.fit(X_train, y_train)
            y_pred_all.append(svc.predict(X_test))
            y_test_all.append(y_test)

        adv_tn, adv_fp, adv_fn, adv_tp = confusion_matrix(y_test_all, y_pred_all).ravel()
        pred_adv_accuracy = (adv_tn + adv_tp)/(adv_tn + adv_tp + adv_fn + adv_fp)

        all_adv_accuracies.append(pred_adv_accuracy)

    # plot a distribution of accuracy values
    plt.hist(all_adv_accuracies, bins='auto')
    plt.show()

### Exposure-specific analyses ###
# Create df for exposure type analysis

# generate indices based on exposure status
# note that these indices intentionally exclude participants with adversity but who don't endorse a specific type
abuse_only_idx = np.where(np.logical_and(np.logical_and(beh.abuse.astype(bool) == True, beh.neglect.astype(bool) == False), beh.maltreatment.astype(bool) == True))
abuse_only_idx = np.asarray(abuse_only_idx)[0]

neglect_only_idx = np.where(np.logical_and(np.logical_and(beh.abuse.astype(bool) == False, beh.neglect.astype(bool) == True), beh.maltreatment.astype(bool) == True))
neglect_only_idx = np.asarray(neglect_only_idx)[0]

both_idx = np.where(np.logical_and(np.logical_and(beh.abuse.astype(bool) == True, beh.neglect.astype(bool) == True), beh.maltreatment.astype(bool) == True))
both_idx = np.asarray(both_idx)[0]

none_idx = np.where(np.logical_and(np.logical_and(beh.abuse.astype(bool) == False, beh.neglect.astype(bool) == False), beh.maltreatment.astype(bool) == False))
none_idx = np.asarray(none_idx)[0]

# convert pandas dataframe into numpy array, and then delete and reconstitute
np_beh_values = beh.values

# Train and test abuse classifier with loo
## abuse vs. none
abuse_none_beh_values = np_beh_values[np.sort(np.concatenate([abuse_only_idx, none_idx]))]
abuse_none_beh = pd.DataFrame(abuse_none_beh_values, columns = beh.columns)
classify_abuse_df = base.Bunch(target_names='abuse',
                               feature_names=feature_names,
                               target=np.asarray(abuse_none_beh.abuse.astype(int)),
                               data=data[np.sort(np.concatenate([abuse_only_idx, none_idx]))])

y_pred_abuse = []
y_test_abuse = []

loo.get_n_splits(classify_abuse_df.data)

for train_index, test_index in loo.split(classify_abuse_df.data):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = classify_abuse_df.data[train_index], classify_abuse_df.data[test_index]
    y_train, y_test = classify_abuse_df.target[train_index], classify_abuse_df.target[test_index]
    svc.fit(X_train, y_train)
    y_pred_abuse.append(svc.predict(X_test))
    y_test_abuse.append(y_test)

abuse_tn, abuse_fp, abuse_fn, abuse_tp = confusion_matrix(y_test_abuse, y_pred_abuse).ravel()
pred_abuse_accuracy = (abuse_tn + abuse_tp)/(abuse_tn + abuse_tp + abuse_fn + abuse_fp)


# Train and test neglect classifier with loo
## neglect vs. none
neglect_none_beh_values = np_beh_values[np.sort(np.concatenate([neglect_only_idx, none_idx]))]
neglect_none_beh = pd.DataFrame(neglect_none_beh_values, columns = beh.columns)
classify_neglect_df = base.Bunch(target_names='neglect',
                               feature_names=feature_names,
                               target=np.asarray(neglect_none_beh.neglect.astype(int)),
                               data=data[np.sort(np.concatenate([neglect_only_idx, none_idx]))])

y_pred_neglect = []
y_test_neglect = []

loo.get_n_splits(classify_neglect_df.data)

for train_index, test_index in loo.split(classify_neglect_df.data):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = classify_neglect_df.data[train_index], classify_neglect_df.data[test_index]
    y_train, y_test = classify_neglect_df.target[train_index], classify_neglect_df.target[test_index]
    svc.fit(X_train, y_train)
    y_pred_neglect.append(svc.predict(X_test))
    y_test_neglect.append(y_test)

neglect_tn, neglect_fp, neglect_fn, neglect_tp = confusion_matrix(y_test_neglect, y_pred_neglect).ravel()
pred_neglect_accuracy = (neglect_tn + neglect_tp)/(neglect_tn + neglect_tp + neglect_fn + neglect_fp)

# Train and test both classifier with loo
## both vs. none
both_none_beh_values = np_beh_values[np.sort(np.concatenate([both_idx, none_idx]))]
both_none_beh = pd.DataFrame(both_none_beh_values, columns = beh.columns)
classify_both_df = base.Bunch(target_names='both',
                               feature_names=feature_names,
                               target=np.asarray(both_none_beh.polyvictimization.astype(int)),
                               data=data[np.sort(np.concatenate([both_idx, none_idx]))])

y_pred_both = []
y_test_both = []

loo.get_n_splits(classify_both_df.data)

for train_index, test_index in loo.split(classify_both_df.data):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = classify_both_df.data[train_index], classify_both_df.data[test_index]
    y_train, y_test = classify_both_df.target[train_index], classify_both_df.target[test_index]
    svc.fit(X_train, y_train)
    y_pred_both.append(svc.predict(X_test))
    y_test_both.append(y_test)

both_tn, both_fp, both_fn, both_tp = confusion_matrix(y_test_both, y_pred_both).ravel()
pred_both_accuracy = (both_tn + both_tp)/(both_tn + both_tp + both_fn + both_fp)

## CROSS-CATEGORY

## TRAINING: ABUSE

# Train on abuse, test on neglect

y_pred_abuse2neglect = []
y_test_abuse2neglect = []

X_train, X_test = classify_abuse_df.data, classify_neglect_df.data
y_train, y_test = classify_abuse_df.target, classify_neglect_df.target
svc.fit(X_train, y_train)
y_pred_abuse2neglect.append(svc.predict(X_test))
y_test_abuse2neglect.append(y_test)

abuse2neglect_tn, abuse2neglect_fp, abuse2neglect_fn, abuse2neglect_tp = confusion_matrix(y_test_abuse2neglect[0], y_pred_abuse2neglect[0]).ravel()
pred_abuse2neglect_accuracy = (abuse2neglect_tn + abuse2neglect_tp)/(abuse2neglect_tn + abuse2neglect_tp + abuse2neglect_fn + abuse2neglect_fp)

# Train on abuse, test on both
y_pred_abuse2both = []
y_test_abuse2both = []

X_train, X_test = classify_abuse_df.data, classify_both_df.data
y_train, y_test = classify_abuse_df.target, classify_both_df.target
svc.fit(X_train, y_train)
y_pred_abuse2both.append(svc.predict(X_test))
y_test_abuse2both.append(y_test)

abuse2both_tn, abuse2both_fp, abuse2both_fn, abuse2both_tp = confusion_matrix(y_test_abuse2both[0], y_pred_abuse2both[0]).ravel()
pred_abuse2both_accuracy = (abuse2both_tn + abuse2both_tp)/(abuse2both_tn + abuse2both_tp + abuse2both_fn + abuse2both_fp)

## TRAINING: NEGLECT

# Train on neglect, test on abuse
y_pred_neglect2abuse = []
y_test_neglect2abuse = []

X_train, X_test = classify_neglect_df.data, classify_abuse_df.data
y_train, y_test = classify_neglect_df.target, classify_abuse_df.target
svc.fit(X_train, y_train)
y_pred_neglect2abuse.append(svc.predict(X_test))
y_test_neglect2abuse.append(y_test)

# same, none were predicted to have neglect
neglect2abuse_tn, neglect2abuse_fp, neglect2abuse_fn, neglect2abuse_tp = confusion_matrix(y_test_neglect2abuse[0], y_pred_neglect2abuse[0]).ravel()
pred_neglect2abuse_accuracy = (neglect2abuse_tn + neglect2abuse_tp)/(neglect2abuse_tn + neglect2abuse_tp + neglect2abuse_fn + neglect2abuse_fp)

# Train on neglect, test on both
y_pred_neglect2both = []
y_test_neglect2both = []

X_train, X_test = classify_neglect_df.data, classify_both_df.data
y_train, y_test = classify_neglect_df.target, classify_both_df.target
svc.fit(X_train, y_train)
y_pred_neglect2both.append(svc.predict(X_test))
y_test_neglect2both.append(y_test)

# none predicted to have neglect
neglect2both_tn, neglect2both_fp, neglect2both_fn, neglect2both_tp = confusion_matrix(y_test_neglect2both[0], y_pred_neglect2both[0]).ravel()
pred_neglect2both_accuracy = (neglect2both_tn + neglect2both_tp)/(neglect2both_tn + neglect2both_tp + neglect2both_fn + neglect2both_fp)

## TRAINING: BOTH

# Train on both, test on abuse
y_pred_both2abuse = []
y_test_both2abuse = []

X_train, X_test = classify_both_df.data, classify_abuse_df.data
y_train, y_test = classify_both_df.target, classify_abuse_df.target
svc.fit(X_train, y_train)
y_pred_both2abuse.append(svc.predict(X_test))
y_test_both2abuse.append(y_test)

both2abuse_tn, both2abuse_fp, both2abuse_fn, both2abuse_tp = confusion_matrix(y_test_both2abuse[0], y_pred_both2abuse[0]).ravel()
pred_both2abuse_accuracy = (both2abuse_tn + both2abuse_tp)/(both2abuse_tn + both2abuse_tp + both2abuse_fn + both2abuse_fp)

# Train on both, test on neglect
y_pred_both2neglect = []
y_test_both2neglect = []

X_train, X_test = classify_neglect_df.data, classify_both_df.data
y_train, y_test = classify_neglect_df.target, classify_both_df.target
svc.fit(X_train, y_train)
y_pred_both2neglect.append(svc.predict(X_test))
y_test_both2neglect.append(y_test)

both2neglect_tn, both2neglect_fp, both2neglect_fn, both2neglect_tp = confusion_matrix(y_test_both2neglect[0], y_pred_both2neglect[0]).ravel()
pred_both2neglect_accuracy = (both2neglect_tn + both2neglect_tp)/(both2neglect_tn + both2neglect_tp + both2neglect_fn + both2neglect_fp)

## FIND CONTROL VALUES

cross_test_names = ["pred_abuse2neglect_accuracy", "pred_abuse2both_accuracy", "pred_neglect2abuse_accuracy", "pred_neglect2both_accuracy", "pred_both2abuse_accuracy", "pred_both2neglect_accuracy"]
cross_test_accuracies = [pred_abuse2neglect_accuracy, pred_abuse2both_accuracy, pred_neglect2abuse_accuracy, pred_neglect2both_accuracy, pred_both2abuse_accuracy, pred_both2neglect_accuracy]

all_adv_accuracies = []
all_abuse_accuracies = []
all_neglect_accuracies = []
all_both_accuracies = []

# re-run classifier 1000x with shuffled target labels and save accuracy values (for abuse and neglect only)
if run_permutation == True:
    ### ADV
    for i in range(1000):
        idx = np.random.permutation(len(classify_adv_df.target))
        y = classify_adv_df.target[idx]

        shuffled_adv_df = classify_adv_df
        shuffled_adv_df.target = y

        y_pred_all = []
        y_test_all = []

        for train_index, test_index in loo.split(shuffled_adv_df.data):
            print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = shuffled_adv_df.data[train_index], shuffled_adv_df.data[test_index]
            y_train, y_test = shuffled_adv_df.target[train_index], shuffled_adv_df.target[test_index]
            svc.fit(X_train, y_train)
            y_pred_all.append(svc.predict(X_test))
            y_test_all.append(y_test)

        adv_tn, adv_fp, adv_fn, adv_tp = confusion_matrix(y_test_all, y_pred_all).ravel()
        pred_adv_accuracy = (adv_tn + adv_tp)/(adv_tn + adv_tp + adv_fn + adv_fp)

        all_adv_accuracies.append(pred_adv_accuracy)

    # plot a distribution of accuracy values
    plt.hist(all_adv_accuracies, bins='auto')
    plt.show()
    plt.savefig('adv_permutation_accuracies.png')

    ### ABUSE
    for i in range(10):
        idx = np.random.permutation(len(classify_abuse_df.target))
        y = classify_abuse_df.target[idx]

        shuffled_abuse_df = classify_abuse_df
        shuffled_abuse_df.target = y

        y_pred_abuse = []
        y_test_abuse = []


        for train_index, test_index in loo.split(shuffled_abuse_df.data):
            print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = shuffled_abuse_df.data[train_index], shuffled_abuse_df.data[test_index]
            y_train, y_test = shuffled_abuse_df.target[train_index], shuffled_abuse_df.target[test_index]
            svc.fit(X_train, y_train)
            y_pred_abuse.append(svc.predict(X_test))
            y_test_abuse.append(y_test)

        abuse_tn, abuse_fp, abuse_fn, abuse_tp = confusion_matrix(y_test_abuse, y_pred_abuse).ravel()
        pred_abuse_accuracy = (abuse_tn + abuse_tp) / (abuse_tn + abuse_tp + abuse_fn + abuse_fp)

        all_abuse_accuracies.append(pred_abuse_accuracy)

    # plot a distribution of accuracy values
    plt.hist(all_abuse_accuracies, bins='auto')
    plt.show()
    plt.savefig('abuse_permutation_accuracies.png')

    ### NEGLECT
    for i in range(1000):
        idx = np.random.permutation(len(classify_neglect_df.target))
        y = classify_neglect_df.target[idx]

        shuffled_neglect_df = classify_neglect_df
        shuffled_neglect_df.target = y

        y_pred_neglect = []
        y_test_neglect = []

        for train_index, test_index in loo.split(shuffled_neglect_df.data):
            print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = shuffled_neglect_df.data[train_index], shuffled_neglect_df.data[test_index]
            y_train, y_test = shuffled_neglect_df.target[train_index], shuffled_neglect_df.target[test_index]
            svc.fit(X_train, y_train)
            y_pred_neglect.append(svc.predict(X_test))
            y_test_neglect.append(y_test)

        neglect_tn, neglect_fp, neglect_fn, neglect_tp = confusion_matrix(y_test_neglect, y_pred_neglect).ravel()
        pred_neglect_accuracy = (neglect_tn + neglect_tp) / (neglect_tn + neglect_tp + neglect_fn + neglect_fp)

        all_neglect_accuracies.append(pred_neglect_accuracy)

    # plot a distribution of accuracy values
    plt.hist(all_neglect_accuracies, bins='auto')
    plt.show()
    plt.savefig('neglect_permutation_accuracies.png')

    ## BOTH
    for i in range(1000):
        idx = np.random.permutation(len(classify_both_df.target))
        y = classify_both_df.target[idx]

        shuffled_both_df = classify_both_df
        shuffled_both_df.target = y

        y_pred_both = []
        y_test_both = []

        for train_index, test_index in loo.split(shuffled_both_df.data):
            print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = shuffled_both_df.data[train_index], shuffled_both_df.data[test_index]
            y_train, y_test = shuffled_both_df.target[train_index], shuffled_both_df.target[test_index]
            svc.fit(X_train, y_train)
            y_pred_both.append(svc.predict(X_test))
            y_test_both.append(y_test)

        both_tn, both_fp, both_fn, both_tp = confusion_matrix(y_test_both, y_pred_both).ravel()
        pred_both_accuracy = (both_tn + both_tp) / (both_tn + both_tp + both_fn + both_fp)

        all_both_accuracies.append(pred_both_accuracy)

    # plot a distribution of accuracy values
    plt.hist(all_both_accuracies, bins='auto')
    plt.show()
    plt.savefig('both_permutation_accuracies.png')

# save median values
adv_chance = stats.median(all_adv_accuracies)
abuse_chance = stats.median(all_abuse_accuracies)
neglect_chance = stats.median(all_neglect_accuracies)
both_chance = stats.median(all_both_accuracies)

a = np.asarray([adv_chance, abuse_chance, neglect_chance, both_chance])
np.savetxt("chance_levels.csv", a, delimiter=",")

b = np.asarray([pred_adv_accuracy, pred_abuse_accuracy, pred_neglect_accuracy, pred_both_accuracy])
np.savetxt("loo_accuracies.csv", b, delimiter=",")

c = np.asarray([pred_abuse2neglect_accuracy, pred_abuse2both_accuracy, pred_neglect2abuse_accuracy, pred_neglect2both_accuracy, pred_both2abuse_accuracy, pred_both2neglect_accuracy])
np.savetxt("pairwise_accuracies.csv", c, delimiter=",")