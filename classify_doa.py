# Train and test a classifier based on the within- and between- parcel rsfc across UO_TDS, OHSU_TSS, and OHSU_MINA studies
# T Cheng | 9/10/2019

import pyreadr

### LOAD DATA ###

box_dir = '/Users/theresacheng/Box/projects/dim_of_adversity'
subject_list_file = 'subjects_across_samples/inc_subj_full_ids.csv'
# beh_data_file

parcellation_dir = '/Users/theresacheng/Box/projects/dim_of_adversity/data/gordon_pconns.RDS'


gordon_parcels = pyreadr.read_r(parcellation_dir)