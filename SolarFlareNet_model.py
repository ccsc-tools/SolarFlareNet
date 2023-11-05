'''
 (c) Copyright 2023
 All rights reserved
 Programs written by Yasser Abduallah
 Department of Computer Science
 New Jersey Institute of Technology
 University Heights, Newark, NJ 07102, USA

 Permission to use, copy, modify, and distribute this
 software and its documentation for any purpose and without
 fee is hereby granted, provided that this copyright
 notice appears in all copies. Programmer(s) makes no
 representations about the suitability of this
 software for any purpose.  It is provided "as is" without
 express or implied warranty.

 @author: Yasser Abduallah
'''
from __future__ import print_function
import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

try:
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
except Exception as e:
    print('')
from pprint import pprint
import numpy as np
import os
from datetime import datetime, timedelta
import time
from time import sleep
import math
from math import sqrt
import random
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import tensorflow.compat.v2 as tf
tf.enable_v2_behavior()
from tensorflow.keras import layers,models
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers
import shutil
from utils import *
print_gpu_info = False

class SolarFlareNet:
    model = None
    model_name = None
    callbacks = None
    input = None
    def __init__(self,model_name='SolarFlareNet',early_stopping_patience=3):
        self.model_name = model_name
        callbacks = [EarlyStopping(monitor='loss', patience=early_stopping_patience,restore_best_weights=True)]
    if print_gpu_info:
        if tf.test.gpu_device_name() != '/device:GPU:0':
          print('WARNING: GPU device not found.')
        else:
            print('SUCCESS: Found GPU: {}'.format(tf.test.gpu_device_name()))
            physical_devices = tf.config.list_physical_devices('GPU')
            if len(physical_devices ) > 0:
                tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
            
    
    def build_base_model(self,
                    input_shape,
                    dropout=0.4,
                    b=4,
                    nclass=2,
                    verbose=True):
                input = keras.Input(shape=input_shape)
                self.input = input 
                input = keras.layers.BatchNormalization()(input)
                model = layers.Conv1D(filters=32, kernel_size=1, activation="relu",
                                        name=self.model_name+"_conv",
                                        kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                                        bias_regularizer=regularizers.l2(1e-4),
                                        activity_regularizer=regularizers.l2(1e-5)                                      
                                      )(input)
                model = layers.Conv1D(filters=32, kernel_size=1, activation="relu",
                                        name=self.model_name+"_conv",
                                        kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                                        bias_regularizer=regularizers.l2(1e-4),
                                        activity_regularizer=regularizers.l2(1e-5)                                      
                                      )(input)                                      
                model = tf.keras.layers.LSTM(400,
                                            return_sequences=True,
                                            name=self.model_name+'_lstm',
                                            kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                                            bias_regularizer=regularizers.l2(1e-4),
                                            activity_regularizer=regularizers.l2(1e-5)                                              
                                             )(model)
                model = layers.Dropout(dropout)(model)
                model = keras.layers.BatchNormalization()(model)
                if b == 0:
                    if verbose:
                        log('Loading multi head attention only..')
                    model = (layers.MultiHeadAttention(key_dim=4, num_heads=4, dropout=0,name=self.model_name +'_mh'))(model,model)
                else:
                    if verbose:
                        log('Loading transformer encoder blocks..')
                    for i in range(b):
                        model = self.transformer_encoder(model,head_size=2,ff_dim=14,dropout=dropout, name_postffix='_enc' + str(i))
                model = layers.Dropout(dropout)(model)
                model = layers.Dense(200, activation='relu',
                                        kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                                        bias_regularizer=regularizers.l2(1e-4),
                                        activity_regularizer=regularizers.l2(1e-5)                                      
                                     )(model)
                model = layers.Dropout(dropout)(model)
                model = layers.Dense(500, activation='relu',
                                        kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                                        bias_regularizer=regularizers.l2(1e-4),
                                        activity_regularizer=regularizers.l2(1e-5)                                      
                                     )(model)                 
                model = (layers.Dropout(dropout))(model)
                model = layers.Dense(nclass,
                                     activation='softmax',
                                     activity_regularizer=regularizers.l2(1e-5)
                                     )(model)
                self.model = model 
                return model   
    
    def transformer_encoder(self, inputs,head_size=2,num_heads=2,ff_dim=4,dropout=0, name_postffix='_enc1'):
        x = layers.Dropout(dropout)(inputs)
        x = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads)(x, x)
        x = keras.layers.BatchNormalization(name='norm_' + name_postffix)(x)
        x = x + inputs
        x = layers.Conv1D(filters=ff_dim, kernel_size=inputs.shape[-2], activation="relu")(x)
        x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
        x = tf.keras.layers.LSTM(400,
                                     return_sequences=True,
                                     name=self.model_name+name_postffix + '_lstm')(x)        
        self.model = x 
        return x;
    def models(self):
        self.model = models.Model(self.input, self.model)
        
    def summary(self):
        self.model.summary()
    
    def compile(self,loss='categorical_crossentropy',metrics=['accuracy'], adam_lr=0.0001):
        self.model.compile(optimizer=tf.optimizers.Adam(learning_rate=adam_lr), 
                           loss=loss,
                           metrics=metrics)       
    def fit(self,
            X_train, 
            y_train,
            X_valid=None, 
            y_valid=None,
            epochs=100,
            verbose=2,
            batch_size=512,
            class_weight=None):
        validation_data = None 
        if X_valid and y_valid:
            validation_data =[X_valid, y_valid]
        self.model.fit(X_train, 
                       y_train, 
                       epochs=epochs, 
                       verbose=verbose, 
                       batch_size=batch_size,
                       callbacks=self.callbacks)
             
        
    def predict(self,X_test,batch_size=1024,verbose=0):
        predictions = self.model.predict(X_test,
                                         verbose=verbose,
                                         batch_size=batch_size)
        return np.squeeze(predictions) 
    
    def save_weights(self,
                     flare_class=None,
                     w_dir=None, verbose=True):
        if w_dir is None and flare_class is None or str(flare_class) == '':
            log('You must specify flare class to save the mode. flare_class is set to:', flare_class, verbose=True)
            exit()        
        if w_dir is None:
            weight_dir = 'models' + os.sep + 'SolarFlareNet' + os.sep +  str(flare_class)  
        else:
            weight_dir = w_dir  
            
        if os.path.exists(weight_dir):
            shutil.rmtree(weight_dir)
        os.makedirs(weight_dir)
        if verbose:
            log('Saving model weights to directory:', weight_dir,verbose=True)
        self.model.save_weights(weight_dir + os.sep + 'model_weights')
    
    def load_weights(self,
                     flare_class=None,
                     w_dir=None,
                     verbose=True):
        if w_dir is None and flare_class is None or str(flare_class) == '':
            log('You must specify flare class to load the mode. flare_class is set to:', flare_class, verbose=True)
            exit()
        if w_dir is None:
            weight_dir = 'models' + os.sep + 'SolarFlareNet' + os.sep + str(flare_class) 
        else:
            weight_dir = w_dir
        if verbose:
            log('Loading weights from model dir:', weight_dir, verbose=True)
        if not os.path.exists(weight_dir):
            print( 'Model weights directory:', weight_dir ,'does not exist, you may train a new model to create it for flare class:',flare_class)
            exit()
        if self.model == None :
            print('You must create model first before loading the weights. You may train a new model to create it for flare class:', flare_class)
            exit() 
        self.model.load_weights(weight_dir + os.sep + 'model_weights').expect_partial() 
    
    def load_model(self,input_shape,
                   flare_class,
                    w_dir=None,
                    verbose=True):
        self.build_base_model(input_shape, verbose=verbose)
        self.models()
        self.compile()
        self.load_weights(flare_class, w_dir=w_dir,verbose=verbose)
        
    def get_model(self):
        return self.model  