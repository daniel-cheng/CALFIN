# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 18:06:26 2019

@author: Daniel
"""

import datetime
import time

o = 2
print('o:', o)
print('fromordinal(o):', datetime.date.fromordinal(o))
t = time.time()
print('t:', t)
wprint('fromtimestamp(t):', datetime.date.fromtimestamp(30000))