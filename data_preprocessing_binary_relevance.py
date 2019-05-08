__author__ = 'clee'
# Python 3.6.1
"""
Changed: 02/12/2019 by C.Snellen
Added not needed emitters so that we can just have a sample of emitters needed for 
multi label classification by binary relevance
"""
# Standard lib
import os

# Additional lib
import h5py
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from lib.preprocess.read import read_mat_binary_relevance
from lib.preprocess.datasets import form_binary_relevance_dataset
from lib.alg.sg_preprocess import log_mag_norm

# File paths
working_dir = os.getcwd()
print(working_dir)

path_mat_data = '/data/Multi-Label/20-Feb-2019'             # Path to input spectrograms for 1007
path_data_store = '/data/Multi-Label/HDF5_Files/ALL/Mar1519-135955'
path_data_store1007 = '/data/Multi-Label/HDF5_Files/1007'      # Path to output directory for HDF5 files for 1007
path_data_store1027 = '/data/Multi-Label/HDF5_Files/1027'
path_data_store1039 = '/data/Multi-Label/HDF5_Files/1039'
path_data_store1043 = '/data/Multi-Label/HDF5_Files/1043'
path_data_store1135 = '/data/Multi-Label/HDF5_Files/1135'
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
# print(exclude_emits)

#Notes
DATA_NOTES = """Data:
Spectrogram: Log(base10) of absolute DFT magnitude (heavy time/freq pooling)
Normalization: MinMax scaling, per-spectrogram
Excluded Emitters: """ + str(exclude_emits) + """
- Including all low PRI emitters to test null hypothesis
"""
# Create postitive and negative sets for each emitter
positive1007 = np.array([['1007'],['1007_1043'],['1007_1027'],['1007_1039']])

positive1027 = np.array([['1027'],['1027_1135'],['1027_1039'],['1007_1027']])

positive1039 = np.array([['1039'],['1039_1135'],['1007_1039'],['1039_1043']])

positive1043 = np.array([['1043'],['1039_1043'],['1027_1043'],['1007_1043']])

positive1135 = np.array([['1135'],['1043_1135'],['1027_1135'],['1007_1135']])


negative1007 = np.array([['1043'],['1039'],['1027']])

negative1027 = np.array([['1135'],['1039'],['1007']])

negative1039 = np.array([['1043'],['1007'],['1135']])

negative1043 = np.array([['1039'],['1027'],['1007']])

negative1135 = np.array([['1043'],['1007'],['1027']])

print('[INFO]-- Creating datasets for emitters 1007, 1027, 1039, 1043, and 1135')
# Read and preprocess .mat files into hdf5 files
path_sg, path_meta, path_hdf5 = read_mat_binary_relevance(path_mat_data, path_data_store, log_mag_norm, data_notes=DATA_NOTES)

# Form training/validation/testing datasets from HDF5 files

path_ds_sg1007, path_ds_meta1007 = form_binary_relevance_dataset(path_sg, path_meta, path_data_store1007+'/datasets',
                                                                 84, (52,8,24), (68,12,32), positive1007,negative1007,shuffle=True)

print('[INFO] -- Dataset for 1007 sorted into train, val, and test sets')

path_ds_sg1027, path_ds_meta1027 = form_binary_relevance_dataset(path_sg, path_meta, path_data_store1027+'/datasets', 
                                                                 84, (52,8,24), (68,12,32), positive1027,negative1027,shuffle=True)

print('[INFO] -- Dataset for 1027 sorted into train, val, and test sets')

path_ds_sg1039, path_ds_meta1039 = form_binary_relevance_dataset(path_sg, path_meta, path_data_store1039+'/datasets', 
                                                                 84, (52,8,24), (68,12,32), positive1039,negative1039,shuffle=True)

print('[INFO] -- Dataset for 1039 sorted into train, val, and test sets')

path_ds_sg1043, path_ds_meta1043 = form_binary_relevance_dataset(path_sg, path_meta, path_data_store1043+'/datasets', 
                                                                 84, (52,8,24), (68,12,32), positive1043,negative1043,shuffle=True)

print('[INFO] -- Dataset for 1043 sorted into train, val, and test sets')

path_ds_sg1135, path_ds_meta1135 = form_binary_relevance_dataset(path_sg, path_meta, path_data_store1135+'/datasets', 
                                                                 84, (52,8,24), (68,12,32), positive1135,negative1135,shuffle=True)

print('[INFO] -- Dataset for 1135 sorted into train, val, and test sets')


"""
# Check Dataset Files
hdf5_ds_sg = h5py.File(path_ds_sg1007, 'r')
hdf5_ds_meta = h5py.File(path_ds_meta1007, 'r')

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

print("path_ds_sg1007 = " + path_ds_sg1007)
print("path_ds_meta1007 = " + path_ds_meta1007)

positive1007 = np.array([['1007'],['1007_1043'],['1007_1027'],['1007_1027_1043']])

positive1027 = np.array([['1027'],['1027_1135'],['1027_1039'],['1007_1027'],['1007_1027_1039'],['1027_1039_1135'],['1007_1027_1135'],['1007_1027_1039_1135']])

positive1039 = np.array([['1039'],['1039_1135'],['1007_1039'],['1039_1043'],['1039_1043_1135'],['1007_1039_1135'],['1007_1039_1043'],['1007_1039_1043_1135']])

positive1043 = np.array([['1043'],['1039_1043'],['1027_1043'],['1007_1043'],['1027_1039_1043'],['1007_1039_1043'],['1007_1027_1043'],['1007_1027_1039_1043']])

positive1135 = np.array([['1135'],['1043_1135'],['1027_1135'],['1007_1135'],['1027_1043_1135'],['1007_1027_1135'],['1007_1043_1135'],['1007_1027_1043_1135']])

negative1007 = np.array([['1027'],['1027_1043'],['1043']])

negative1027 = np.array([['1007'],['1007_1039'],['1007_1039_1135'],['1007_1135'],['1039'],['1039_1135'],['1135']])

negative1039 = np.array([['1007'],['1007_1043'],['1007_1043_1135'],['1007_1135'],['1043'],['1043_1135'],['1135']])

negative1043 = np.array([['1007'],['1007_1027'],['1007_1027_1039'],['1007_1039'],['1027'],['1027_1039'],['1039']])

negative1135 = np.array([['1007'],['1007_1027'],['1007_1027_1043'],['1007_1043'],['1027'],['1027_1043'],['1043']])
"""