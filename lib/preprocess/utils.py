#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 10:35:44 2019

@author: chelsi.snellen
"""
from lib.alg.sg_preprocess import log_mag_norm
import scipy.io as sio
import numpy as np
from keras.preprocessing.image import img_to_array
import cv2
import sys


def GenRandNumList(total, number):
    randArray = np.random.rand(number,1)
    specArray = np.floor((randArray * total)+1)
    
    return specArray

mainArray = GenRandNumList(2187, 333)
    
    
def networkDataPreprocessing(data_path,
                     func_preprocess = log_mag_norm,
                     mat_key = 'pooledData'):
    # Read in the data and format it to fit into the network
    if data_path.split('.')[-1] == 'mat':    # Check if the file that is read is a MATLAB .mat file 
        data = log_mag_norm(sio.loadmat(data_path)[mat_key].T)   # Preprocess the .mat file 
        data = np.expand_dims(data,axis = 0)    # Expand the dimensions of the np array to tell the network how many samples there are
        data = np.expand_dims(data,axis = -1)   # Expand the dimensions of the np array to tell the network how many channels there are
    elif data_path.split('.')[-1] in ('jpg','jpeg','png'): # Check if the file that is read is an image file
        image = cv2.imread(data_path)   # Read in the image file
        image = cv2.resize(image, (96, 96)) # Resize the image to fit into the networks ****CHANGE IMAGE INPUT****
        image = image.astype("float") / 255.0   # Normalize the image
        image = img_to_array(image) # Conver the image into a Numpy array
        if len(image.shape) == 2:   # Check if the np array is in blacck and white (won't have a dimension for channels)
            data = np.expand_dims(image, axis=0)    # Expand the dimensions of the np array to tell the network how many samples there are
            data = np.expand_dims(data,axis = -1)   # Expand the dimensions of the np array to tell the network how many channels there are
        elif len(image.shape) == 3: # Check if the np array represents a color image (should have a channel dimension)
            data = np.expand_dims(image, axis=0)    # Expand the dimensions of the np array to tell the network how many samples there are
    
    return data

def progress(count, total, status=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
    sys.stdout.flush() # As suggested by Rom Ruben (see: http://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console/27871113#comment50529068_27871113)