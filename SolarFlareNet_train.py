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


import warnings 
warnings.filterwarnings('ignore')
import sys 
from sklearn.utils import class_weight
from utils import * 
from SolarFlareNet_model import SolarFlareNet


def train(time_window, flare_class):
    log('Training is initiated for time window:', time_window, 'and flare class:', flare_class,verbose=True)   
    X_train, y_train = get_training_data(time_window, flare_class)
    y_train_tr = data_transform(y_train)
    epochs=20
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = SolarFlareNet()
    
    model.build_base_model(input_shape)
    model.models()
    model.compile()
    y_train_tr = y_train_tr.reshape(y_train_tr.shape[0],1,y_train_tr.shape[1])
    w_dir = 'models' +os.sep + str(time_window) + os.sep + str(flare_class) 
    model.fit(X_train, y_train_tr, epochs=epochs, verbose=2)
    model.save_weights(flare_class=None, w_dir=w_dir)
    
if __name__ == '__main__':
    time_window = str(sys.argv[1]).strip().upper()
    flare_class = str(sys.argv[2]).strip().upper()
    if not flare_class in supported_flare_class:
        print('Unsupported flass class:', sys.argv[1], ', it must be one of:',  ', '.join(supported_flare_class))
    train(time_window,flare_class)