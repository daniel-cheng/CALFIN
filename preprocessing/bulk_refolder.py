# -*- coding: utf-8 -*-
"""
Created on Thu May 16 12:25:54 2019

@author: Daniel
"""

import os, glob, shutil
  
steps = ['1', '2']
steps = ['2']

if '1' in steps:
    source = r'D:/Daniel/Pictures/CALFIN/test_full/'
    dest = r'D:/Daniel/Pictures/CALFIN/test_full/'
    dry_run = 0
    
    # Function to rename multiple files 
    for source_path in glob.glob(source + '*.png'): 
        filename = source_path.split(os.path.sep)[-1]
        
        #Make domain folder if not existing already
        domain = filename.split('_')[0]
        domain_path = os.path.join(dest, domain)
        if not os.path.exists(domain_path):
            os.mkdir(domain_path)
        
        dest_path = os.path.join(domain_path, filename)
        print(dest_path)
        if dry_run != 0:
            shutil.move(source_path, dest_path)
        
if '2' in steps:
    source = r'D:/Daniel/Pictures/CALFIN/test_full/'
    dest = r'D:/Daniel/Pictures/CALFIN/test_all/'
    dry_run = 1
    
    # Function to rename multiple files 
    for domain_path in glob.glob(source + '*'):
        for filename in os.listdir(domain_path):
            source_path = os.path.join(domain_path, filename)
            dest_path = os.path.join(dest, filename)
#            print(source_path, dest_path)
            if dry_run != 0:
                shutil.copy(source_path, dest_path)