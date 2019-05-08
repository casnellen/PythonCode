#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 16:05:23 2019

@author: chelsi.snellen
"""
import keras
from keras.models import load_model
from lib.alg.sg_preprocess import log_mag_norm


def binary_relevance(model_list,class_names, data):
    
    
    num_models = len(model_list)                                                #Find out how many models were given
    
    num_class_names = len(class_names)                                          # Get the number of class names given 
    
    # Check if the number of class labels and names are the same
    if num_class_names != num_models:
        raise ValueError('The number of class labels and models are not the same. Please provide a model for every label')   # Raise an exception if the two values are not the same
        
    # Create a zeros array that will hold the predictions from the networks
    predictions = []
    
    # Cycle through the trained models to get their predictions on the data
    for i in range (0,num_models):
        loaded_model = model_list[i]                                            # Get the current model in the list
        scores = loaded_model.predict(data)                                     # Get the prediction from the model
        if scores > 0.5:                                                        # Check to make sure that the model predicted the target_emitter for the model
            predictions.append(class_names[i])                                  # Add the class name to the predictions list 
    
    return predictions
    