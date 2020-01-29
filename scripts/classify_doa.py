# Train and test a classifier based on the within- and between- parcel rsfc across UO_TDS, OHSU_TSS, and OHSU_MINA studies
# T Cheng | 9/10/2019

import csv
import pyreadr
import numpy as np
#import statistics as stats
import scipy.stats as stats
from itertools import compress
from random import sample

from sklearn.model_selection import LeaveOneOut, StratifiedKFold, train_test_split, GridSearchCV, cross_val_score
from sklearn.utils import Bunch

loo = LeaveOneOut()
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.datasets import base
from sklearn.feature_selection import SelectPercentile, f_classif

import matplotlib.pyplot as plt
import pandas as pd

# define some list extraction functions
def extract(lst):
    return [item[0] for item in lst]


def extract_two(lst):
    return [item[1] for item in lst]


from sklearn.pipeline import make_pipeline
run_permutation = False # this takes ~ 1 hr
run_sampling = True

### LOAD DATA ###

# Directory and file names
working_dir = '/Users/theresacheng/projects/doa_classification/'
subject_list_path = working_dir + 'data/inc_subj_full_ids.csv'
beh_path = working_dir + 'data/beh_data.RDS'
comm_conns_path = working_dir + 'data/comm_cors_z_hs_feat.csv' # re-run with data Z-transformed within study 'data/comm_cors_z.csv'

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
feature_names = list(comm_conns[0])[0].split(",")[1:92] # make a list of the column names split by commas, remove subject_id
raw_data = list(comm_conns[1:len(comm_conns)])

data = []
for row in raw_data:
    row_as_list_subj_id = row[0].split(",")  # there are 92 items per row
    row_as_list = row_as_list_subj_id[1:92] # remove subject_id
    row_as_list = list(map(float, row_as_list))
    data.append(row_as_list)
data = np.asarray(data)

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
percent_of_features = 30
feat_sel = SelectPercentile(f_classif, percent_of_features)

y_pred_all = []
y_test_all = []
y_dec_func_all = []
n_features = round((percent_of_features/100)*len(classify_adv_df.data[0]))
#feat_sel_idxes = np.empty((len(classify_adv_df.data), n_features), dtype=np.int8) #(

for train_index, test_index in loo.split(classify_adv_df.data):
   # print("TRAIN:", train_index, "TEST:", test_index)

    # generate train and test feature and outcome sets
    X_train, X_test = classify_adv_df.data[train_index], classify_adv_df.data[test_index]
    y_train, y_test = classify_adv_df.target[train_index], classify_adv_df.target[test_index]

    # apply feature selection using the training group only
    feat_sel.fit(X_train, y_train) # fit ANOVA models, ID top 15
    X_train_new = feat_sel.transform(X_train) # pare down training set features
    X_test_new = feat_sel.transform(X_test)  # pare down test set features
    #feat_sel_idxes[test_index] = feat_sel.get_support(indices=True)

    # fit model with pared down training set
    svc.fit(X_train_new, y_train)
    y_dec_func_all.append(svc.decision_function(X_test_new))
    y_pred_all.append(svc.predict(X_test_new)) # predict outcomes based on new test set
    y_test_all.append(y_test)

adv_tn, adv_fp, adv_fn, adv_tp = confusion_matrix(y_test_all, y_pred_all).ravel()
pred_adv_accuracy = (adv_tn + adv_tp)/(adv_tn + adv_tp + adv_fn + adv_fp)
print('Adversity status accuracy:', round(pred_adv_accuracy, 2))

# compare classification confidence in the true adversity and no adversity groups
y_dec_func_adv = list(compress(y_dec_func_all, classify_adv_df.target))
y_dec_func_no_adv = list(compress(y_dec_func_all, np.logical_not(classify_adv_df.target)))

ttest_result = stats.ttest_ind(y_dec_func_adv, y_dec_func_no_adv)

# save subject_id, study, protocol, adv, abuse, neglect, class_label, class_evidence, mean_FD, mean_FD_censored
anon_ids = ["anon_" + str(num) for num in range(1,len(beh.study)+1)]

with open('output/adv_classifier_by_subj.csv', 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["subject_id", "study", "protocol", "adv", "abuse", "neglect", "adv_pred", "adv_dec_func", "mean_FD", "mean_FD_censored"])
    writer.writerows(zip(anon_ids, beh.study, beh.protocol, beh.maltreatment, beh.abuse, beh.neglect, y_pred_all, y_dec_func_all, beh.mean_FD, beh.mean_FD_censored))

# plt.hist(feat_sel_idxes, bins=91)

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

            # apply feature selection using the training group only
            feat_sel.fit(X_train, y_train)  # fit ANOVA models, ID top 15
            X_train_new = feat_sel.transform(X_train)  # pare down training set features
            X_test_new = feat_sel.transform(X_test)  # pare down test set features
            # feat_sel_idxes[test_index] = feat_sel.get_support(indices=True)

            svc.fit(X_train_new, y_train)
            y_pred_all.append(svc.predict(X_test_new))
            y_test_all.append(y_test)

        adv_tn, adv_fp, adv_fn, adv_tp = confusion_matrix(y_test_all, y_pred_all).ravel()
        pred_adv_accuracy = (adv_tn + adv_tp)/(adv_tn + adv_tp + adv_fn + adv_fp)

        all_adv_accuracies.append(pred_adv_accuracy)

    # plot a distribution of accuracy values
    plt.hist(all_adv_accuracies, bins='auto')
    plt.savefig('output/all_adv_perm_accuracies.png')

    plt.close()

    all_adv_accuracies_quartiles = [round(np.percentile(all_adv_accuracies, 25), 2),
                                    round(np.percentile(all_adv_accuracies, 50), 2),
                                    round(np.percentile(all_adv_accuracies, 75), 2),
                                    round(np.percentile(all_adv_accuracies, 95), 2)]

    print('Permuted adversity accuracy 1st, 2nd, 3rd quartile and 95%:', all_adv_accuracies_quartiles)

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

# CLASSIFY INDIVIDUALS WITH ABUSE
all_abuse_accuracies = []
all_abuse_tstats = []
all_abuse_hit = []
all_abuse_fa = []
avg_y_dec_func_abuse = []
avg_y_dec_func_none = []

if run_sampling == True:
    for i in range(1000):
        # take a random subset of the none_idx so that they match the number of individuals that have experienced abuse
        none_idx_subsample = sample(list(none_idx), len(abuse_only_idx))

        # Train and test abuse classifier with loo
        ## abuse vs. none
        abuse_none_beh_values = np_beh_values[np.sort(np.concatenate([abuse_only_idx, none_idx_subsample]))]
        abuse_none_beh = pd.DataFrame(abuse_none_beh_values, columns = beh.columns)
        classify_abuse_df = base.Bunch(target_names='abuse',
                                       feature_names=feature_names,
                                       target=np.asarray(abuse_none_beh.abuse.astype(int)),
                                       data=data[np.sort(np.concatenate([abuse_only_idx, none_idx_subsample]))])

        skf = StratifiedKFold(n_splits=int(len(classify_abuse_df.data) / 2),
                              shuffle=True)  # remove each participant 1x for a total of 33 folds, but shuffle so that one from each group is randomly selected
        skf.get_n_splits(classify_abuse_df.data, classify_abuse_df.target)

        y_pred_abuse = []
        y_test_abuse = []
        y_dec_func = []
        test_indices = []

        for train_index, test_index in skf.split(X=classify_abuse_df.data, y=classify_abuse_df.target):
            print("TRAIN:", train_index, "TEST:", test_index)

            # generate train and test feature and outcome sets
            X_train, X_test = classify_abuse_df.data[train_index], classify_abuse_df.data[test_index]
            y_train, y_test = classify_abuse_df.target[train_index], classify_abuse_df.target[test_index]

            # apply feature selection using the training group only
            feat_sel.fit(X_train, y_train)  # fit ANOVA models, ID top 15%
            X_train_new = feat_sel.transform(X_train)  # pare down training set features
            X_test_new = feat_sel.transform(X_test)  # pare down test set features
            #feat_sel_idxes[test_index] = feat_sel.get_support(indices=True)

            # fit model with pared down training set
            svc.fit(X_train_new, y_train)
            y_dec_func.append(svc.decision_function(X_test_new))
            y_pred_abuse.append(svc.predict(X_test_new))  # predict outcomes based on new test set
            y_test_abuse.append(y_test)
            test_indices.append(test_index)

        # need to unpack arrays of doubles by extracting and concatenating them
        y_dec_func_all = extract(y_dec_func) + extract_two(y_dec_func)
        y_test_abuse_all = extract(y_test_abuse) + extract_two(y_test_abuse)
        y_pred_abuse_all = extract(y_pred_abuse) + extract_two(y_pred_abuse)
        test_indices = extract(test_indices) + extract_two(test_indices)

        # compare classification confidence in the true adversity and no adversity groups
        y_dec_func_abuse = list(compress(y_dec_func_all, classify_abuse_df.target[test_indices]))
        y_dec_func_none = list(compress(y_dec_func_all, np.logical_not(classify_abuse_df.target[test_indices])))

        ttest_result = stats.ttest_ind(y_dec_func_abuse, y_dec_func_none)

        # ok but this part should work
        abuse_tn, abuse_fp, abuse_fn, abuse_tp = confusion_matrix(y_test_abuse_all, y_pred_abuse_all).ravel()
        half_total = len(classify_abuse_df.target)/2
        pred_abuse_accuracy = (abuse_tn + abuse_tp)/(abuse_tn + abuse_fp + abuse_fn + abuse_tp)

        all_abuse_accuracies.append(round(pred_abuse_accuracy,2))
        all_abuse_tstats.append(ttest_result[0])
        all_abuse_hit.append(round(abuse_tp/half_total, 2))
        all_abuse_fa.append(round(abuse_fp/half_total, 2))

        avg_y_dec_func_abuse.append(round(np.mean(y_dec_func_abuse),2))
        avg_y_dec_func_none.append(round(np.mean(y_dec_func_none),2))

    # plot a distribution of accuracy values
    plt.hist(all_abuse_accuracies, bins='auto')
    plt.savefig('output/abuse_accuracies.png')
    plt.close()

    plt.hist(all_abuse_tstats, bins='auto')
    plt.savefig('output/abuse_decfunc_tstats.png')
    plt.close()

print('Abuse status accuracy:', round(np.mean(all_abuse_accuracies), 2))

with open('output/abuse_classifier_perm.csv', 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["avg_decfunc_abuse", "avg_decfunc_none", "t_stat", "perc_hits", "per_fa", "accuracy"])
    writer.writerows(zip(avg_y_dec_func_abuse, avg_y_dec_func_none, all_abuse_tstats, all_abuse_hit, all_abuse_fa, all_abuse_accuracies))

# CLASSIFY INDIVIDUALS WITH NEGLECT
all_neglect_accuracies = []
all_neglect_tstats = []
all_neglect_hit = []
all_neglect_fa = []
avg_y_dec_func_neglect = []
avg_y_dec_func_none = []

if run_sampling == True:
    for i in range(1000):
        # take a random subset of the none_idx so that they match the number of individuals that have experienced neglect
        none_idx_subsample = sample(list(none_idx), len(neglect_only_idx))

        # Train and test neglect classifier with loo
        ## neglect vs. none
        neglect_none_beh_values = np_beh_values[np.sort(np.concatenate([neglect_only_idx, none_idx_subsample]))]
        neglect_none_beh = pd.DataFrame(neglect_none_beh_values, columns = beh.columns)
        classify_neglect_df = base.Bunch(target_names='neglect',
                                       feature_names=feature_names,
                                       target=np.asarray(neglect_none_beh.neglect.astype(int)),
                                       data=data[np.sort(np.concatenate([neglect_only_idx, none_idx_subsample]))])

        skf = StratifiedKFold(n_splits=int(len(classify_neglect_df.data) / 2),
                              shuffle=True)  # remove each participant 1x for a total of 33 folds, but shuffle so that one from each group is randomly selected
        skf.get_n_splits(classify_neglect_df.data, classify_neglect_df.target)

        y_pred_neglect = []
        y_test_neglect = []
        y_dec_func = []
        test_indices = []

        for train_index, test_index in skf.split(X=classify_neglect_df.data, y=classify_neglect_df.target):
            print("TRAIN:", train_index, "TEST:", test_index)

            # generate train and test feature and outcome sets
            X_train, X_test = classify_neglect_df.data[train_index], classify_neglect_df.data[test_index]
            y_train, y_test = classify_neglect_df.target[train_index], classify_neglect_df.target[test_index]

            # apply feature selection using the training group only
            feat_sel.fit(X_train, y_train)  # fit ANOVA models, ID top 15%
            X_train_new = feat_sel.transform(X_train)  # pare down training set features
            X_test_new = feat_sel.transform(X_test)  # pare down test set features
            #feat_sel_idxes[test_index] = feat_sel.get_support(indices=True)

            # fit model with pared down training set
            svc.fit(X_train_new, y_train)
            y_dec_func.append(svc.decision_function(X_test_new))
            y_pred_neglect.append(svc.predict(X_test_new))  # predict outcomes based on new test set
            y_test_neglect.append(y_test)
            test_indices.append(test_index)

        # need to unpack arrays of doubles by extracting and concatenating them
        y_dec_func_all = extract(y_dec_func) + extract_two(y_dec_func)
        y_test_neglect_all = extract(y_test_neglect) + extract_two(y_test_neglect)
        y_pred_neglect_all = extract(y_pred_neglect) + extract_two(y_pred_neglect)
        test_indices = extract(test_indices) + extract_two(test_indices)

        # compare classification confidence in the true adversity and no adversity groups
        y_dec_func_neglect = list(compress(y_dec_func_all, classify_neglect_df.target[test_indices]))
        y_dec_func_none = list(compress(y_dec_func_all, np.logical_not(classify_neglect_df.target[test_indices])))

        ttest_result = stats.ttest_ind(y_dec_func_neglect, y_dec_func_none)

        # ok but this part should work
        neglect_tn, neglect_fp, neglect_fn, neglect_tp = confusion_matrix(y_test_neglect_all, y_pred_neglect_all).ravel()
        half_total = len(classify_neglect_df.target)/2
        pred_neglect_accuracy = (neglect_tn + neglect_tp)/(neglect_tn + neglect_fp + neglect_fn + neglect_tp)

        all_neglect_accuracies.append(round(pred_neglect_accuracy,2))
        all_neglect_tstats.append(ttest_result[0])
        all_neglect_hit.append(round(neglect_tp/half_total, 2))
        all_neglect_fa.append(round(neglect_fp/half_total, 2))

        avg_y_dec_func_neglect.append(round(np.mean(y_dec_func_neglect),2))
        avg_y_dec_func_none.append(round(np.mean(y_dec_func_none),2))

    # plot a distribution of accuracy values
    plt.hist(all_neglect_accuracies, bins='auto')
    plt.savefig('output/neglect_accuracies.png')
    plt.close()

    plt.hist(all_neglect_tstats, bins='auto')
    plt.savefig('output/neglect_decfunc_tstats.png')
    plt.close()

print('Neglect status accuracy:', round(np.mean(all_neglect_accuracies), 2))

with open('output/neglect_classifier_perm.csv', 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["avg_decfunc_neglect", "avg_decfunc_none", "t_stat", "perc_hits", "per_fa", "accuracy"])
    writer.writerows(zip(avg_y_dec_func_neglect, avg_y_dec_func_none, all_neglect_tstats, all_neglect_hit, all_neglect_fa, all_neglect_accuracies))


# CLASSIFY INDIVIDUALS WITH BOTH
all_both_accuracies = []
all_both_tstats = []
all_both_hit = []
all_both_fa = []
avg_y_dec_func_both = []
avg_y_dec_func_none = []

if run_sampling == True:
    for i in range(1000):
        # take a random subset of the none_idx so that they match the number of individuals that have experienced both
        none_idx_subsample = sample(list(none_idx), len(both_only_idx))

        # Train and test both classifier with loo
        ## both vs. none
        both_none_beh_values = np_beh_values[np.sort(np.concatenate([both_only_idx, none_idx_subsample]))]
        both_none_beh = pd.DataFrame(both_none_beh_values, columns = beh.columns)
        classify_both_df = base.Bunch(target_names='both',
                                       feature_names=feature_names,
                                       target=np.asarray(both_none_beh.both.astype(int)),
                                       data=data[np.sort(np.concatenate([both_only_idx, none_idx_subsample]))])

        skf = StratifiedKFold(n_splits=int(len(classify_both_df.data) / 2),
                              shuffle=True)  # remove each participant 1x for a total of 33 folds, but shuffle so that one from each group is randomly selected
        skf.get_n_splits(classify_both_df.data, classify_both_df.target)

        y_pred_both = []
        y_test_both = []
        y_dec_func = []
        test_indices = []

        for train_index, test_index in skf.split(X=classify_both_df.data, y=classify_both_df.target):
            print("TRAIN:", train_index, "TEST:", test_index)

            # generate train and test feature and outcome sets
            X_train, X_test = classify_both_df.data[train_index], classify_both_df.data[test_index]
            y_train, y_test = classify_both_df.target[train_index], classify_both_df.target[test_index]

            # apply feature selection using the training group only
            feat_sel.fit(X_train, y_train)  # fit ANOVA models, ID top 15%
            X_train_new = feat_sel.transform(X_train)  # pare down training set features
            X_test_new = feat_sel.transform(X_test)  # pare down test set features
            #feat_sel_idxes[test_index] = feat_sel.get_support(indices=True)

            # fit model with pared down training set
            svc.fit(X_train_new, y_train)
            y_dec_func.append(svc.decision_function(X_test_new))
            y_pred_both.append(svc.predict(X_test_new))  # predict outcomes based on new test set
            y_test_both.append(y_test)
            test_indices.append(test_index)

        # need to unpack arrays of doubles by extracting and concatenating them
        y_dec_func_all = extract(y_dec_func) + extract_two(y_dec_func)
        y_test_both_all = extract(y_test_both) + extract_two(y_test_both)
        y_pred_both_all = extract(y_pred_both) + extract_two(y_pred_both)
        test_indices = extract(test_indices) + extract_two(test_indices)

        # compare classification confidence in the true adversity and no adversity groups
        y_dec_func_both = list(compress(y_dec_func_all, classify_both_df.target[test_indices]))
        y_dec_func_none = list(compress(y_dec_func_all, np.logical_not(classify_both_df.target[test_indices])))

        ttest_result = stats.ttest_ind(y_dec_func_both, y_dec_func_none)

        # ok but this part should work
        both_tn, both_fp, both_fn, both_tp = confusion_matrix(y_test_both_all, y_pred_both_all).ravel()
        half_total = len(classify_both_df.target)/2
        pred_both_accuracy = (both_tn + both_tp)/(both_tn + both_fp + both_fn + both_tp)

        all_both_accuracies.append(round(pred_both_accuracy,2))
        all_both_tstats.append(ttest_result[0])
        all_both_hit.append(round(both_tp/half_total, 2))
        all_both_fa.append(round(both_fp/half_total, 2))

        avg_y_dec_func_both.append(round(np.mean(y_dec_func_both),2))
        avg_y_dec_func_none.append(round(np.mean(y_dec_func_none),2))

    # plot a distribution of accuracy values
    plt.hist(all_both_accuracies, bins='auto')
    plt.savefig('output/both_accuracies.png')
    plt.close()

    plt.hist(all_both_tstats, bins='auto')
    plt.savefig('output/both_decfunc_tstats.png')
    plt.close()

print('Both status accuracy:', round(np.mean(all_both_accuracies), 2))

with open('output/both_classifier_perm.csv', 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["avg_decfunc_both", "avg_decfunc_none", "t_stat", "perc_hits", "per_fa", "accuracy"])
    writer.writerows(zip(avg_y_dec_func_both, avg_y_dec_func_none, all_both_tstats, all_both_hit, all_both_fa, all_both_accuracies))

## CROSS-CATEGORY - NOTE: haven't implemented the random sampling, since there doesn't seem to be a point if it can't predict itself

## TRAINING: ABUSE

# Train on abuse, test on neglect
#
# y_pred_abuse2neglect = []
# y_test_abuse2neglect = []
#
# X_train, X_test = classify_abuse_df.data, classify_neglect_df.data
# y_train, y_test = classify_abuse_df.target, classify_neglect_df.target
# svc_selected.fit(X_train, y_train)
# y_pred_abuse2neglect.append(svc.predict(X_test))
# y_test_abuse2neglect.append(y_test)
#
# abuse2neglect_tn, abuse2neglect_fp, abuse2neglect_fn, abuse2neglect_tp = confusion_matrix(y_test_abuse2neglect[0], y_pred_abuse2neglect[0]).ravel()
# pred_abuse2neglect_accuracy = (abuse2neglect_tn + abuse2neglect_tp)/(abuse2neglect_tn + abuse2neglect_tp + abuse2neglect_fn + abuse2neglect_fp)
#
# # Train on abuse, test on both
# y_pred_abuse2both = []
# y_test_abuse2both = []
#
# X_train, X_test = classify_abuse_df.data, classify_both_df.data
# y_train, y_test = classify_abuse_df.target, classify_both_df.target
# svc_selected.fit(X_train, y_train)
# y_pred_abuse2both.append(svc.predict(X_test))
# y_test_abuse2both.append(y_test)
#
# abuse2both_tn, abuse2both_fp, abuse2both_fn, abuse2both_tp = confusion_matrix(y_test_abuse2both[0], y_pred_abuse2both[0]).ravel()
# pred_abuse2both_accuracy = (abuse2both_tn + abuse2both_tp)/(abuse2both_tn + abuse2both_tp + abuse2both_fn + abuse2both_fp)
#
# ## TRAINING: NEGLECT
#
# # Train on neglect, test on abuse
# y_pred_neglect2abuse = []
# y_test_neglect2abuse = []
#
# X_train, X_test = classify_neglect_df.data, classify_abuse_df.data
# y_train, y_test = classify_neglect_df.target, classify_abuse_df.target
# svc_selected.fit(X_train, y_train)
# y_pred_neglect2abuse.append(svc.predict(X_test))
# y_test_neglect2abuse.append(y_test)
#
# # same, none were predicted to have neglect
# neglect2abuse_tn, neglect2abuse_fp, neglect2abuse_fn, neglect2abuse_tp = confusion_matrix(y_test_neglect2abuse[0], y_pred_neglect2abuse[0]).ravel()
# pred_neglect2abuse_accuracy = (neglect2abuse_tn + neglect2abuse_tp)/(neglect2abuse_tn + neglect2abuse_tp + neglect2abuse_fn + neglect2abuse_fp)
#
# # Train on neglect, test on both
# y_pred_neglect2both = []
# y_test_neglect2both = []
#
# X_train, X_test = classify_neglect_df.data, classify_both_df.data
# y_train, y_test = classify_neglect_df.target, classify_both_df.target
# svc_selected.fit(X_train, y_train)
# y_pred_neglect2both.append(svc.predict(X_test))
# y_test_neglect2both.append(y_test)
#
# # none predicted to have neglect
# neglect2both_tn, neglect2both_fp, neglect2both_fn, neglect2both_tp = confusion_matrix(y_test_neglect2both[0], y_pred_neglect2both[0]).ravel()
# pred_neglect2both_accuracy = (neglect2both_tn + neglect2both_tp)/(neglect2both_tn + neglect2both_tp + neglect2both_fn + neglect2both_fp)
#
# ## TRAINING: BOTH
#
# # Train on both, test on abuse
# y_pred_both2abuse = []
# y_test_both2abuse = []
#
# X_train, X_test = classify_both_df.data, classify_abuse_df.data
# y_train, y_test = classify_both_df.target, classify_abuse_df.target
# svc_selected.fit(X_train, y_train)
# y_pred_both2abuse.append(svc.predict(X_test))
# y_test_both2abuse.append(y_test)
#
# both2abuse_tn, both2abuse_fp, both2abuse_fn, both2abuse_tp = confusion_matrix(y_test_both2abuse[0], y_pred_both2abuse[0]).ravel()
# pred_both2abuse_accuracy = (both2abuse_tn + both2abuse_tp)/(both2abuse_tn + both2abuse_tp + both2abuse_fn + both2abuse_fp)
#
# # Train on both, test on neglect
# y_pred_both2neglect = []
# y_test_both2neglect = []
#
# X_train, X_test = classify_neglect_df.data, classify_both_df.data
# y_train, y_test = classify_neglect_df.target, classify_both_df.target
# svc_selected.fit(X_train, y_train)
# y_pred_both2neglect.append(svc.predict(X_test))
# y_test_both2neglect.append(y_test)
#
# both2neglect_tn, both2neglect_fp, both2neglect_fn, both2neglect_tp = confusion_matrix(y_test_both2neglect[0], y_pred_both2neglect[0]).ravel()
# pred_both2neglect_accuracy = (both2neglect_tn + both2neglect_tp)/(both2neglect_tn + both2neglect_tp + both2neglect_fn + both2neglect_fp)
#
# ## FIND CONTROL VALUES
#
# cross_test_names = ["pred_abuse2neglect_accuracy", "pred_abuse2both_accuracy", "pred_neglect2abuse_accuracy", "pred_neglect2both_accuracy", "pred_both2abuse_accuracy", "pred_both2neglect_accuracy"]
# cross_test_accuracies = [pred_abuse2neglect_accuracy, pred_abuse2both_accuracy, pred_neglect2abuse_accuracy, pred_neglect2both_accuracy, pred_both2abuse_accuracy, pred_both2neglect_accuracy]
#
# all_adv_accuracies = []
# all_abuse_accuracies = []
# all_neglect_accuracies = []
# all_both_accuracies = []
#
# # re-run classifier 1000x with shuffled target labels and save accuracy values (for abuse and neglect only)
# if run_permutation == True:
#     ### ADV
#     for i in range(1000):
#         idx = np.random.permutation(len(classify_adv_df.target))
#         y = classify_adv_df.target[idx]
#
#         shuffled_adv_df = classify_adv_df
#         shuffled_adv_df.target = y
#
#         y_pred_all = []
#         y_test_all = []
#
#         for train_index, test_index in loo.split(shuffled_adv_df.data):
#             print("TRAIN:", train_index, "TEST:", test_index)
#             X_train, X_test = shuffled_adv_df.data[train_index], shuffled_adv_df.data[test_index]
#             y_train, y_test = shuffled_adv_df.target[train_index], shuffled_adv_df.target[test_index]
#             svc_selected.fit(X_train, y_train)
#             y_pred_all.append(svc.predict(X_test))
#             y_test_all.append(y_test)
#
#         adv_tn, adv_fp, adv_fn, adv_tp = confusion_matrix(y_test_all, y_pred_all).ravel()
#         pred_adv_accuracy = (adv_tn + adv_tp)/(adv_tn + adv_tp + adv_fn + adv_fp)
#
#         all_adv_accuracies.append(pred_adv_accuracy)
#
#     # plot a distribution of accuracy values
#     plt.hist(all_adv_accuracies, bins='auto')
#     plt.show()
#     plt.savefig('adv_permutation_accuracies.png')
#
#     ### ABUSE
#     for i in range(10):
#         idx = np.random.permutation(len(classify_abuse_df.target))
#         y = classify_abuse_df.target[idx]
#
#         shuffled_abuse_df = classify_abuse_df
#         shuffled_abuse_df.target = y
#
#         y_pred_abuse = []
#         y_test_abuse = []
#
#
#         for train_index, test_index in loo.split(shuffled_abuse_df.data):
#             print("TRAIN:", train_index, "TEST:", test_index)
#             X_train, X_test = shuffled_abuse_df.data[train_index], shuffled_abuse_df.data[test_index]
#             y_train, y_test = shuffled_abuse_df.target[train_index], shuffled_abuse_df.target[test_index]
#             svc_selected.fit(X_train, y_train)
#             y_pred_abuse.append(svc.predict(X_test))
#             y_test_abuse.append(y_test)
#
#         abuse_tn, abuse_fp, abuse_fn, abuse_tp = confusion_matrix(y_test_abuse, y_pred_abuse).ravel()
#         pred_abuse_accuracy = (abuse_tn + abuse_tp) / (abuse_tn + abuse_tp + abuse_fn + abuse_fp)
#
#         all_abuse_accuracies.append(pred_abuse_accuracy)
#
#     # plot a distribution of accuracy values
#     plt.hist(all_abuse_accuracies, bins='auto')
#     plt.show()
#     plt.savefig('abuse_permutation_accuracies.png')
#
#     ### NEGLECT
#     for i in range(1000):
#         idx = np.random.permutation(len(classify_neglect_df.target))
#         y = classify_neglect_df.target[idx]
#
#         shuffled_neglect_df = classify_neglect_df
#         shuffled_neglect_df.target = y
#
#         y_pred_neglect = []
#         y_test_neglect = []
#
#         for train_index, test_index in loo.split(shuffled_neglect_df.data):
#             print("TRAIN:", train_index, "TEST:", test_index)
#             X_train, X_test = shuffled_neglect_df.data[train_index], shuffled_neglect_df.data[test_index]
#             y_train, y_test = shuffled_neglect_df.target[train_index], shuffled_neglect_df.target[test_index]
#             svc_selected.fit(X_train, y_train)
#             y_pred_neglect.append(svc.predict(X_test))
#             y_test_neglect.append(y_test)
#
#         neglect_tn, neglect_fp, neglect_fn, neglect_tp = confusion_matrix(y_test_neglect, y_pred_neglect).ravel()
#         pred_neglect_accuracy = (neglect_tn + neglect_tp) / (neglect_tn + neglect_tp + neglect_fn + neglect_fp)
#
#         all_neglect_accuracies.append(pred_neglect_accuracy)
#
#     # plot a distribution of accuracy values
#     plt.hist(all_neglect_accuracies, bins='auto')
#     plt.show()
#     plt.savefig('neglect_permutation_accuracies.png')
#
#     ## BOTH
#     for i in range(1000):
#         idx = np.random.permutation(len(classify_both_df.target))
#         y = classify_both_df.target[idx]
#
#         shuffled_both_df = classify_both_df
#         shuffled_both_df.target = y
#
#         y_pred_both = []
#         y_test_both = []
#
#         for train_index, test_index in loo.split(shuffled_both_df.data):
#             print("TRAIN:", train_index, "TEST:", test_index)
#             X_train, X_test = shuffled_both_df.data[train_index], shuffled_both_df.data[test_index]
#             y_train, y_test = shuffled_both_df.target[train_index], shuffled_both_df.target[test_index]
#             svc_selected.fit(X_train, y_train)
#             y_pred_both.append(svc.predict(X_test))
#             y_test_both.append(y_test)
#
#         both_tn, both_fp, both_fn, both_tp = confusion_matrix(y_test_both, y_pred_both).ravel()
#         pred_both_accuracy = (both_tn + both_tp) / (both_tn + both_tp + both_fn + both_fp)
#
#         all_both_accuracies.append(pred_both_accuracy)
#
#     # plot a distribution of accuracy values
#     plt.hist(all_both_accuracies, bins='auto')
#     plt.show()
#     plt.savefig('both_permutation_accuracies.png')
#
# # save median values
# adv_chance = stats.median(all_adv_accuracies)
# abuse_chance = stats.median(all_abuse_accuracies)
# neglect_chance = stats.median(all_neglect_accuracies)
# both_chance = stats.median(all_both_accuracies)
#
# a = np.asarray([adv_chance, abuse_chance, neglect_chance, both_chance])
# np.savetxt("chance_levels.csv", a, delimiter=",")
#
# b = np.asarray([pred_adv_accuracy, pred_abuse_accuracy, pred_neglect_accuracy, pred_both_accuracy])
# np.savetxt("loo_accuracies.csv", b, delimiter=",")
#
# c = np.asarray([pred_abuse2neglect_accuracy, pred_abuse2both_accuracy, pred_neglect2abuse_accuracy, pred_neglect2both_accuracy, pred_both2abuse_accuracy, pred_both2neglect_accuracy])
# np.savetxt("pairwise_accuracies.csv", c, delimiter=",")