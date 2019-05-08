#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 14:50:37 2019

@author: chelsi.snellen
"""

from lib.eval_bin_rel import evaluate_binary_relevance
import keras
from keras.models import load_model

data_path = "/data/Multi-Label/Test-Specs"                                      # Path to test data

model_paths = ['/home/chelsi.snellen/Documents/Code/Python/emitterid/results/1007/Mar1819-124414/models/epochs75-100.h5',
               '/home/chelsi.snellen/Documents/Code/Python/emitterid/results/1027/Mar2519-150607/models/epochs75-100.h5',
               '/home/chelsi.snellen/Documents/Code/Python/emitterid/results/1039/Mar2519-155953/models/epochs75-100.h5',
               '/home/chelsi.snellen/Documents/Code/Python/emitterid/results/1043/Mar2619-082728/models/epochs75-100.h5',
               '/home/chelsi.snellen/Documents/Code/Python/emitterid/results/1135/Mar2619-092130/models/epochs75-100.h5'] # Path to keras models
num_models = len(model_paths)                                                   # Get the number of models
models = []                                                                     # Initialize the model list
for i in range(0,num_models):                                                   # Cycle through the number of models
    current_model = model_paths[i]                                              # Create the current model path
    loaded_model = load_model(current_model)                                    # Load the current kears model
    models.append(loaded_model)                                                 # Append the keras model to the  models list
    
classNames = ['1007','1027','1039','1043','1135']                               # State the class names 

accuracy,scores = evaluate_binary_relevance(models,classNames,data_path)        # Evaulate the accuracy of the binary relevance models 

print("Overall accuracy of %d for binary network" %(accuracy))                  # Print the overall accuracy