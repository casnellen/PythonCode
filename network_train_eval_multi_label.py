__author__ = 'clee'
# Python 3.6.1
# Python 3.6.1
"""
Changed: 02/12/2019 by C.Snellen
changed the model to have a binary output so we can build an ensemble of binary 
classifiers for multi-label classification
"""
# Standard lib
import os
import sys
import datetime
import itertools

# Additional lib
import h5py
import tensorflow as tf
import numpy as np

# Keras
import keras
import keras.backend as K
from keras.layers import Input, Dense, Flatten, LSTM, GRU, Conv2D, MaxPooling2D, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.layers.core import Activation, Reshape
from keras.constraints import maxnorm
from keras.regularizers import l1

from lib.network.train_eval import train_eval_multi_label

print(sys.version)

# File paths
working_dir = os.getcwd()
print(working_dir)

path_data = '/data/Multi-Label/Adapted_Multi_Class/Mar1819-135643'       # Path to HDF5 file directory
path_datasets = path_data + '/datasets'                 # Path to dataset HDF5 file directory
path_results = working_dir + '/results/Adapted_Multi_Class'                  # Path to results output directory
fname_ds_sg = '/sg_datasets.h5'
fname_ds_meta = '/meta_datasets.h5'

# Notes
data_notes_file = open(path_data + '/_data_notes.txt', 'r')
data_notes = ''.join(data_notes_file.readlines())
data_notes_file.close()
EXPERIMENT_NOTES = data_notes + '\n\n' + """Model Notes:
Current standard convolutional LSTM model adapted to do multi label classification on emitters 1007, 1027, 1039, 1043, 1135
""" 


# Load Datasets
hdf5_ds_sg = h5py.File(path_datasets + fname_ds_sg, 'r')
hdf5_ds_meta = h5py.File(path_datasets + fname_ds_meta, 'r')

ds_train_sg = hdf5_ds_sg['train_sg']
ds_train_cls = hdf5_ds_meta['train_class']
ds_train_emit = hdf5_ds_meta['train_emitter']
ds_val_sg = hdf5_ds_sg['val_sg']
ds_val_cls = hdf5_ds_meta['val_class']
ds_test_sg = hdf5_ds_sg['test_sg']
ds_test_cls = hdf5_ds_meta['test_class']

train_set = (ds_train_sg, ds_train_cls)
val_set = (ds_val_sg, ds_val_cls)
test_set = (ds_test_sg, ds_test_cls)

shape_sg = ds_train_sg[0].shape
num_cls = 5
cls_names = ['1007','1027','1039','1043','1135']

# Build Model
with tf.device('/gpu:0'):

    # Convolutional Layers
    model_input = Input(shape_sg + (1,))
    x = Conv2D(32,
                (3,3),
                strides=(1,1),
                padding='same',
                kernel_initializer='glorot_uniform',
                kernel_regularizer=l1(0.0),
                kernel_constraint=maxnorm(4))(model_input)
    x = MaxPooling2D(pool_size=(3,3), strides=(5,5))(x)
    x = Activation(activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Conv2D(64,
                (3,3),
                strides=(1,1),
                padding='same',
                kernel_initializer='glorot_uniform',
                kernel_regularizer=l1(0.0),
                kernel_constraint=maxnorm(4))(x)
    x = MaxPooling2D(pool_size=(1,1), strides=(2,3))(x)
    x = Activation(activation='relu')(x)
    x = Dropout(0.5)(x)

    # Classifier
    x_shape = K.int_shape(x)
    print(x_shape)
    x = Reshape((int(x_shape[1]), int(x_shape[2] * x_shape[3])))(x)
    x_shape = K.int_shape(x)
    print(x_shape)
    lstm_out = LSTM(64, recurrent_dropout=0.5)(x)
    x = Activation(activation='relu')(lstm_out)
    classifier = Dense(num_cls, activation='sigmoid')(x)
    model = Model(model_input, classifier)
optimization = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(loss='categorical_crossentropy', optimizer=optimization, metrics=['accuracy'])


# Train model for TOTAL_EPOCHS, stopping to evaluate every INTERVAL_EPOCHS
print(model.summary())
count_epoch = 0
prev_hist = None
TOTAL_EPOCHS = 200
INTERVAL_EPOCHS = 50


rundate = datetime.datetime.now().strftime('%b%d%y-%H%M%S')

for i in range(0, TOTAL_EPOCHS, INTERVAL_EPOCHS):
    count_epoch, prev_hist = train_eval_multi_label(model, train_set, test_set, INTERVAL_EPOCHS, path_results, num_cls, cls_names, rundate,
                                        val_set=val_set,
                                        start_epoch=count_epoch,
                                        batch_size=4,
                                        experiment_notes=EXPERIMENT_NOTES,
                                        verbose=1,
                                        show=False,
                                        prev_hist=prev_hist)

