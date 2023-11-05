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

'''
This script runs all the test for flare class: C, M, M5, and time window: 24, 48, 72
'''

from SolarFlareNet_test import *
for time_window in [24,48,72]:
    for flare_class in ['C', 'M', 'M5']:
        test(str(time_window), flare_class)
        log('===========================================================\n\n',verbose=True)