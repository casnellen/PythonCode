#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 13:45:29 2019

@author: chelsi.snellen
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 12:19:44 2019

@author: chelsi.snellen
"""

# Colin Lee, 06/11/2018
# Python 3.6.1

# Standard lib
import os
import math

# Additional lib
import h5py
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from lib.preprocess.read import read_mat_multi_label
from lib.preprocess.datasets import form_multi_label_datasets
from lib.alg.sg_preprocess import log_mag_norm

### FILE ###
working_dir = os.getcwd()
print(working_dir)

path_mat_data = '/data/Multi-Label/20-Feb-2019'             # Path to input spectrograms
path_data_store = '/data/Multi-Label/Adapted_Multi_Class'      # Path to output directory for HDF5 files

# Compile list of emitters to exclude
low_pri_emits = ['1014', '1018']
cw_emits = ['1026', '1025', '1126', '1138']
too_few = ['1128', '1129', '1130','1009','1033','1034','1010','1026','1117']    # < 334 spectrograms
empty_record = ['1037', '5005']
double_emits = ['1021&1028', '1022&1029', '1023&1030', '1024&1031']
not_needed = ['1001','1005','1006','1008','1019','1020','1028','1032','1036','1038','1057','1062',
              '1063','1068','1111','1112','1113','1114','1125','1127','1131','1132','1134','1136','1137',
              '1139','1140','1147']

exclude_emits = too_few + empty_record + double_emits + low_pri_emits + cw_emits + not_needed
print(exclude_emits)

test_specs = np.array([['1007_1027_1039'],['1007_1027_1039_1043'],['1007_1027_1039_1043_1135'],['1007_1039_1043'],['1027_1043'],['1027_1135']])

DATA_NOTES = """Data:
Spectrogram: 
- Log(base10) of absolute DFT magnitude (heavy time/freq pooling)
Normalization: MinMax scaling, per-spectrogram
Excluded Emitters: """ + str(exclude_emits) + """
- Including all low PRI emitters to test null hypothesis
"""


# Read and preprocess .mat files into hdf5 files
"""path_sg, path_meta, path_hdf5 = read_mat_multi_label(path_mat_data, path_data_store, log_mag_norm, 
                                         exclude_emits=exclude_emits, 
                                         data_notes=DATA_NOTES,
                                         mat_key='pooledData')
"""
path_sg = "/data/Multi-Label/HDF5_Files/ALL/Mar1519-135955/spectrograms.h5"
path_meta = "/data/Multi-Label/HDF5_Files/ALL/Mar1519-135955/metadata.h5"
path_hdf5 = "/data/Multi-Label/Adapted_Multi_Class/Mar1819-135643"

# Form training/validation/testing datasets from HDF5 files
path_ds_sg, path_ds_meta = form_multi_label_datasets(path_sg, path_meta, path_hdf5+'/datasets',path_mat_data, 333, (203, 30, 100),test_specs,5)

hdf5_ds_sg = h5py.File(path_ds_sg, 'r')
hdf5_ds_meta = h5py.File(path_ds_meta, 'r')

ds_train_sg = hdf5_ds_sg['train_sg']
ds_train_emit = hdf5_ds_meta['train_emitter']
ds_train_cls = hdf5_ds_meta['train_class']
ds_train_file = hdf5_ds_meta['train_file']
ds_val_sg = hdf5_ds_sg['val_sg']
ds_val_emit = hdf5_ds_meta['val_emitter']
ds_val_cls = hdf5_ds_meta['val_class']
ds_val_file = hdf5_ds_meta['val_file']
ds_test_sg = hdf5_ds_sg['test_sg']
ds_test_emit = hdf5_ds_meta['test_emitter']
ds_test_cls = hdf5_ds_meta['test_class']
ds_test_file = hdf5_ds_meta['test_file']

print(ds_train_sg.shape)
print(ds_train_emit.shape)
print(ds_train_cls.shape)
print(ds_train_file.shape)
print(ds_val_sg.shape)
print(ds_val_emit.shape)
print(ds_val_cls.shape)
print(ds_val_file.shape)
print(ds_test_sg.shape)
print(ds_test_emit.shape)
print(ds_test_cls.shape)
print(ds_test_file.shape)

print("path_ds_sg = " + path_ds_sg)
print("path_ds_meta = " + path_ds_meta)
