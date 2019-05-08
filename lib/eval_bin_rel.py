#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 15:16:43 2019

@author: chelsi.snellen
"""
# Standard lib
import os

# Additional lib
import keras

# lib imports
from lib.preprocess.utils import networkDataPreprocessing
from lib.network.binary_relevance_network import binary_relevance


def evaluate_binary_relevance(model_list, class_names, test_folder):
    """
    Evaluates the overall accuracy of the binsry relevance model
    
    :param model_list       list of keras models        the binary keras models
    :param class_names      list of str                 the names of the class in relation to each model
    :param test_folder      str                         file path to the test data
    
    :return accuracy        int                         the overall accuracy of the model
    :return scores          list                        the scores for each test item
    """
    scores = []                                                                 # Initialize the output vector
    
    for root, subdirs, files in os.walk(test_folder):                           # Cycle through the test folder
        subdirs.sort()                                                          # Sort the sub directories
        files.sort()                                                            # Sort the files 
        for file in files:                                                      # Cycle through the files 
            path_file = os.path.join(root, file)                                # Create the file path for the current file
            if file.split('.')[-1] == 'mat':                                    # Check if the file is a .mat file
                data = networkDataPreprocessing(path_file)                      # Load and format the data from the file
                label = root.split('/')[-1]                                     # Get the true label from the file name (format: cls1_cls2_cls3)
                pre_prediction = binary_relevance(model_list,class_names,data)  # Get the networks prediction
                num_label = len(pre_prediction)                                 # Find the number of labels that were predicted
                for i in range(0,num_label):                                    # Cycle through the number of labels
                    if i == 0:                                                  # Check if it is the first label predicted 
                        prediction = pre_prediction[i]                          # Add the networks prediction                  
                    else:
                        prediction = prediction + '_' + pre_prediction[i]       # If not the first label then concatinate _prediction to label
                if label == prediction:
                    scores.append(1)                                            # If the true label and network prediction are the same then append a 1 to the scores
                else:
                    scores.append(0)                                            # If label and prediction are not the same then append a zero to the scores
                print("[INFO] -- y %s: y^ %s : Overall %d %% " %(label,
                      prediction,(sum(scores)/len(scores) * 100)))              # Print a running accuracy for each label predicted 
    accuracy =(sum(scores)/len(scores) * 100)                                   # Calculate the overall accuracy of the model
    return accuracy, scores