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
warnings.filterwarnings("ignore")
import pandas as pd 
import numpy as np
import sys 
from datetime import datetime
import platform
import os
import random
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

try:
    import tensorflow as tf    
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
except Exception as e:
    print('')
    

if tf.test.gpu_device_name() != '/device:GPU:0':
  print('WARNING: GPU device not found.')
else:
    print('SUCCESS: Found GPU: {}'.format(tf.test.gpu_device_name()))
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices ) > 0:        
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

print ('Python version:',platform.python_version())
tf_version = tf.__version__
print('Tensorflow bakcend version:',tf_version )

supported_flare_class = ['C','M','M5']
n_features = 14
start_feature = 5
mask_value = 0
series_len = 10
batch_size = 256
nclass = 2
noise_enabled=False
c_date = datetime.now()

d_type = ''
log_handler = None
format_logging = True 

def create_log_file(alg='SolarFlareNet',  d_type='flares', dir_name='logs'):
    os.makedirs(dir_name,  exist_ok=True)
    global log_handler
    try:
        log_file = dir_name + os.sep + 'run_' + str(alg) + '_' + str(c_date.month) + '-' + str(c_date.day) + '-' + str(c_date.year) + '_' +  str( d_type )+  '.log'
    except Exception as e:
        log_file = 'logs' + os.sep +'run_' + str(alg) + '_' + str(c_date.month) + '-' + str(c_date.day) + '-' + str(c_date.year) + '_' +  str( d_type )+ '.log'
    log_handler = open(log_file,'a')
    sys.stdout = Logger(log_handler)  
    print('')

class Logger(object):
    def __init__(self,logger):
        self.terminal = sys.stdout
        self.log = logger

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass  

def log(*message,verbose=False, end=' '):
    log_str = []
    if verbose:
        if format_logging:
            print('[' + str(datetime.now().replace(microsecond=0))  +'] ', end='')
        for m in message:
            print(m,end=end)

        print('')
    log_handler.flush()
    
    
def truncate_float(number, digits=4) -> float:
    try :
        if math.isnan(number):
            return 0.0
        stepper = 10.0 ** digits
        return math.trunc(stepper * number) / stepper
    except Exception as e:
        return number

def parse_time(time):
    time = str(time).strip()
    # print('time:', time)
    time = time.replace('A','0').replace('90','09').replace('91','01').replace('U','0').replace('//','00')
    if '.' in time :
        time = time[:time.index('.')]
    
    time = time.replace('T',' ').replace('Z',':00')
    s = time.split()
    s1=  s[1].split(':')
    if int(float(s1[1])) > 59:
        s1[1] = '59'
    if int(float(s1[2])) > 59:
        s1[2] = '59'
    time = s[0] + ' ' + ':'.join(s1)    
    return datetime.strptime(time, '%Y-%m-%d %H:%M:%S')

def load_data(datafile, flare_label, series_len, start_feature, n_features, mask_value, data =None):
    # print('Loading...', datafile, flare_label, series_len, start_feature, n_features, mask_value)
    if datafile is not None:
        log('loading data from file:', datafile,verbose=False)
    if data is not None:
        df = data 
    else:
        df = pd.read_csv(datafile)
        
    df_columns = list(df.columns) 
    df_values = df.values
    X = []
    y = []
    tmp = []
    for k in range(start_feature, start_feature + n_features):
        tmp.append(mask_value)
    for idx in range(0, len(df_values)):
        each_series_data = []
        row = df_values[idx]
        label = row[0][0]
        if label == 'p':
            continue
        # print(row, label)
        # if flare_label == 'C' and (label == 'P' or label == 'M'):
        #     label = 'C'
        # if flare_label == 'C' and label == 'B':
        #     label = 'N'
        has_zero_record = False
        # if at least one of the 25 physical feature values is missing, then discard it.
        if flare_label == 'C':
            if float(row[5]) == 0.0:
                has_zero_record = True
            if float(row[7]) == 0.0:
                has_zero_record = True
            for k in range(9, 13):
                if float(row[k]) == 0.0:
                    has_zero_record = True
                    break
            for k in range(14, 16):
                if float(row[k]) == 0.0:
                    has_zero_record = True
                    break
            if float(row[18]) == 0.0:
                has_zero_record = True

        if has_zero_record is False:
            cur_noaa_num = int(row[3])
            each_series_data.append(row[start_feature:start_feature + n_features].tolist())
            itr_idx = idx - 1
            while itr_idx >= 0 and len(each_series_data) < series_len:
                prev_row = df_values[itr_idx]
                prev_noaa_num = int(prev_row[3])
                if prev_noaa_num != cur_noaa_num:
                    break
                has_zero_record_tmp = False
                if flare_label == 'C':
                    if float(row[5]) == 0.0:
                        has_zero_record_tmp = True
                    if float(row[7]) == 0.0:
                        has_zero_record_tmp = True
                    for k in range(9, 13):
                        if float(row[k]) == 0.0:
                            has_zero_record_tmp = True
                            break
                    for k in range(14, 16):
                        if float(row[k]) == 0.0:
                            has_zero_record_tmp = True
                            break
                    if float(row[18]) == 0.0:
                        has_zero_record_tmp = True

                if len(each_series_data) < series_len and has_zero_record_tmp is True:
                    each_series_data.insert(0, tmp)

                if len(each_series_data) < series_len and has_zero_record_tmp is False:
                    each_series_data.insert(0, prev_row[start_feature:start_feature + n_features].tolist())
                itr_idx -= 1

            while len(each_series_data) > 0 and len(each_series_data) < series_len:
                each_series_data.insert(0, tmp)

            if len(each_series_data) > 0:
                c_ls = 'TOTUSJH,TOTUSJZ,USFLUX,TOTBSQ,R_VALUE,TOTPOT,SAVNCPP,AREA_ACR,ABSNJZH'.split(',')
                c_all = 'TOTUSJH,Cdec,TOTUSJZ,Chis1d,USFLUX,TOTBSQ,R_VALUE,TOTPOT,Chis,SAVNCPP,AREA_ACR,Edec,Xmax1d,ABSNJZH'.split(',')
                for s1 in range(len(each_series_data)):
                    s1v = each_series_data[s1]
                    s11 =[]
                    for s111 in c_ls:
                        s11.append(s1v[c_all.index(s111)])
                    each_series_data[s1] = s11
                X.append(np.array(each_series_data).reshape(series_len, len(c_ls)).tolist())                
                # X.append(np.array(each_series_data).reshape(series_len, n_features).tolist())
                y.append(label)
    X_arr = np.array(X)
    y_arr = np.array(y)

    # log('data shape:',X_arr.shape)
    return X_arr, y_arr,df
def data_transform(data):
    encoder = LabelEncoder()
    encoder.fit(data)
    encoded_Y = encoder.transform(data)
    converteddata = np_utils.to_categorical(encoded_Y)
    return converteddata


def get_class_num (c):
    if c.strip().upper() == 'N':
        return 0
    return 1

def gaussian_noise(x,mu,std):
    noise = np.random.normal(mu, std, size = np.array(x.values).shape)
    x_noisy = x + noise
    return x_noisy

def add_gaussian_noise(flare_class, 
                       X_train_data, 
                       y_train_data,
                       train_data_for_noise):
    log('Adding Gaussian Noise')
    mu=0.0
    noise_data= train_data_for_noise[train_data_for_noise.columns[start_feature:n_features]]
    std = 0.05 * np.std(noise_data)
    d_noise = gaussian_noise(noise_data, mu, std)
    train_data_for_noise[train_data_for_noise.columns[start_feature:n_features]]=d_noise
    
    X_train_data1, y_train_data1,train_data_for_noise1 = load_data(datafile=None,
                                           flare_label=flare_class, series_len=series_len,
                                           start_feature=start_feature, n_features=n_features,
                                           mask_value=mask_value,data=train_data_for_noise)
    
    X_train = X_train_data 
    y_train = y_train_data 


    n_index_r = [ i for i in range(len(y_train_data1)) if y_train_data1[i] != 'N']
    X=X_train_data.tolist()
    y=y_train_data.tolist()
    
    X1=[X_train_data1[i] for i in n_index_r]
    y1=[y_train_data1[i] for i in n_index_r]

    
    X.extend(X1) 
    y.extend(y1)

    
    y_train=[get_class_num(c) for c in y]
    
    X_train =np.array(X) 
    y_train =np.array(y_train)
    return X_train, y_train

def get_cross_validation_data_raw(time_window, flare_class):
    file_name = 'data' + os.sep + 'data_' + flare_class + '_' + time_window+'.csv'   
    
    data =  pd.read_csv(file_name)
    print('data columns:', data.columns)
    
    return data


def get_all_data(time_window, flare_class, noise_enabled=True):
    file_name = 'data' + os.sep + 'data_' + flare_class + '_' + time_window+'.csv'   
    
    return get_data(flare_class,file_name, noise_enabled=noise_enabled)

def get_training_data(time_window, flare_class):
    file_name = 'data' + os.sep + 'testing_data_' + flare_class + '_' + time_window+'.csv'   
    return  get_data(flare_class,file_name, noise_enabled=True)

def get_testing_data(time_window, flare_class):
    file_name = 'data' + os.sep + 'testing_data_' + flare_class + '_' + time_window+'.csv'   
    return  get_data(flare_class,file_name, noise_enabled=False)

def get_data(flare_class, datafile, noise_enabled=noise_enabled, verbose=True):
    
    X_train_data, y_train_data,train_data_for_noise = load_data(datafile=datafile,
                                           flare_label=flare_class, series_len=series_len,
                                           start_feature=start_feature, n_features=n_features,
                                           mask_value=mask_value)
    
    neg_train = [ t for t in y_train_data if t == 'N' ]
    if verbose:
        log(flare_class, '--> Training: Positive:', len(y_train_data) - len(neg_train) , 'Negative:', len(neg_train))
    if flare_class in ['M','M5'] and noise_enabled:
        X_train, y_train = add_gaussian_noise(flare_class, X_train_data,y_train_data,train_data_for_noise)
        neg_train = [ t for t in y_train if t == 0 ]
        if verbose:
            log(flare_class, '--> With Noise Training: Positive:', len(y_train) - len(neg_train) , 'Negative:', len(neg_train))
    else:
        y_train=[get_class_num(c) for c in y_train_data]
        X_train = X_train_data

    
    
    return X_train, y_train

def save_result(flare_class, time_window, y_true, y_pred,alg='SolarFlareNet', dir_name=None, file_name=None):
    y_pred_probs = [1-p[0] for p in y_pred ]
    def getClass(values):
        b = []
        for v in values:
            if v[0] == 1:
                b.append(0)
            else:
                b.append(1) 
        return b
    y_true = getClass(y_true)

    y_pred = np.argmax(y_pred, axis=1)
    if dir_name is None:
        dir_name = 'result' + os.sep +  alg

    os.makedirs(dir_name,  exist_ok=True) 
    if file_name is None:
        file_name = dir_name + os.sep + flare_class.strip().upper() + '_' + str(time_window)+ '.csv'
    log('Saving result to file:', file_name,verbose=True)
    h = open(file_name, 'w')
    h.write('FlareLabel,Prediction,PredictionProbability\n')
    matchings = []
    for i in range(len(y_true)):
        h.write(str(y_true[i]) + ',' + str(y_pred[i])  +',' + str(y_pred_probs[i])+'\n')
    
    h.flush()
    h.close()

    
create_log_file()