# Train and test a classifier based on the within- and between- parcel rsfc across UO_TDS, OHSU_TSS, and OHSU_MINA studies
# T Cheng | 9/10/2019

import csv
import pyreadr
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix
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
beh =  pyreadr.read_r(beh_path)[None]

# Load within and between community connectivity rsfc MRI data
comm_conns_file = open(comm_conns_path, "r+")
comm_conns = list(csv.reader(comm_conns_file, delimiter='\t'))

# Create dataframe for adversity classification
data = comm_conns[1:len(comm_conns)]
feature_names = list(comm_conns[0])
target_names = 'maltreatment'
target = list(beh.maltreatment)

classify_adv_df = sklearn.datasets.base.Bunch(target_names=target_names,
                                              feature_names=feature_names,
                                              target=target,
                                              data=data)

# Train and test classifier


# Create dataframe for exposure type analysis
# Note that fewer subjects are used here -- those with unspecified forms of maltreatment are removed


# temporarily to see how the bunch object is structured
# import sklearn.datasets
# wine = sklearn.datasets.load_wine()