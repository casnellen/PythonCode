"""
Functions for building standard models
"""

__author__ = 'clee'
# Python 3.6.1

# Additional lib
import tensorflow as tf

# Keras
import keras
import keras.backend as K
from keras.layers import Input, Dense, Flatten, LSTM, GRU, Conv2D, MaxPooling2D, Dropout
from keras.models import Model
from keras.layers.core import Activation, Reshape
from keras.constraints import maxnorm
from keras.regularizers import l1
        
def may3118(shape_sg, num_cls):
    """
    Current standard 64 unit LSTM with two convolutional layers. No convolutional dropout

    :param shape_sg:
    :param num_cls:
    :return model:
    """
    with tf.device('/gpu:0'):
        # Convolutional Layers
        model_input = Input(shape_sg + (1,))
        x = Conv2D(32,
                   (3, 3),
                   strides=(1, 1),
                   padding='same',
                   kernel_initializer='glorot_uniform',
                   kernel_regularizer=l1(0.0),
                   kernel_constraint=maxnorm(4))(model_input)
        x = MaxPooling2D(pool_size=(3, 3), strides=(5, 5))(x)
        x = Activation(activation='relu')(x)
        x = Dropout(0.0)(x)
        x = Conv2D(64,
                   (3, 3),
                   strides=(1, 1),
                   padding='same',
                   kernel_initializer='glorot_uniform',
                   kernel_regularizer=l1(0.0),
                   kernel_constraint=maxnorm(4))(x)
        x = MaxPooling2D(pool_size=(1, 1), strides=(2, 3))(x)
        x = Activation(activation='relu')(x)
        x = Dropout(0.0)(x)

        # Classifier
        x_shape = K.int_shape(x)
        print(x_shape)
        x = Reshape((int(x_shape[1]), int(x_shape[2] * x_shape[3])))(x)
        x_shape = K.int_shape(x)
        print(x_shape)
        lstm_out = LSTM(64, recurrent_dropout=0.5)(x)
        x = Activation(activation='relu')(lstm_out)
        classifier = Dense(num_cls, activation='softmax')(x)
        model = Model(model_input, classifier)

    return model


def jun1318(shape_sg, num_cls):
    """
    Experimental 16 unit LSTM with two convolutional layers. High convolutional dropout

    :param shape_sg:
    :param num_cls:
    :return model:
    """
    with tf.device('/gpu:0'):
        # Convolutional Layers
        model_input = Input(shape_sg + (1,))
        x = Conv2D(32,
                   (3, 3),
                   strides=(1, 1),
                   padding='same',
                   kernel_initializer='glorot_uniform',
                   kernel_regularizer=l1(0.0),
                   kernel_constraint=maxnorm(4))(model_input)
        x = MaxPooling2D(pool_size=(3, 3), strides=(5, 5))(x)
        x = Activation(activation='relu')(x)
        x = Dropout(0.0)(x)
        x = Conv2D(64,
                   (3, 3),
                   strides=(1, 1),
                   padding='same',
                   kernel_initializer='glorot_uniform',
                   kernel_regularizer=l1(0.0),
                   kernel_constraint=maxnorm(4))(x)
        x = MaxPooling2D(pool_size=(1, 1), strides=(2, 3))(x)
        x = Activation(activation='relu')(x)
        x = Dropout(0.0)(x)

        # Classifier
        x_shape = K.int_shape(x)
        print(x_shape)
        x = Reshape((int(x_shape[1]), int(x_shape[2] * x_shape[3])))(x)
        x_shape = K.int_shape(x)
        print(x_shape)
        lstm_out = LSTM(64, recurrent_dropout=0.5)(x)
        x = Activation(activation='relu')(lstm_out)
        classifier = Dense(num_cls, activation='softmax')(x)
        model = Model(model_input, classifier)

    return model

def feb1519(shape_sg,num_cls):
    """
    Current binary classifer network used for multi-label classification
    
    :param shape_sg:
    :param num_cls:
    :return model:
    """
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
        
    return model