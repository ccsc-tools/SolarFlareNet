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
import random 
from sklearn.utils import class_weight
from utils import * 
from SolarFlareNet_model import SolarFlareNet


def test(time_window, flare_class):
    log('Testing is initiated for time_window:', time_window, 'and flare class:', flare_class,verbose=True)
    X_test, y_test = get_testing_data(time_window, flare_class)
    y_test_tr = data_transform(y_test)
    y_true = y_test_tr[:]
    input_shape = (X_test.shape[1], X_test.shape[2])
    
    w_dir = 'models' +os.sep + str(time_window) + os.sep + str(flare_class)
    
    model = SolarFlareNet()
    model.load_model(input_shape,flare_class,w_dir=w_dir)
    
    y_test_tr = y_test_tr.reshape(y_test_tr.shape[0],1,y_test_tr.shape[1])
    log('Predicting test data set samples..',verbose=True)
    prediction = model.predict(X_test)
    save_result(flare_class,time_window, y_true, prediction, alg='SolarFlareNet')
    
if __name__ == '__main__':
    time_window = str(sys.argv[1]).strip().upper()
    flare_class = str(sys.argv[2]).strip().upper()
    if not flare_class in supported_flare_class:
        print('Unsupported flass class:', sys.argv[2], ', it must be one of:',  ', '.join(supported_flare_class))
        exit()
    if not str(time_window) in ['24','48','72']:
        print('Unsupported time window class:', sys.argv[1], ', it must be one of: [24, 48, 72]')
        exit()
    test(time_window, flare_class)