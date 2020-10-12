# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 22:38:07 2019

@author: Daniel
"""

import os, shutil, glob

root_source_path = r'../outputs/production_staging'
total = 0
total_bad = 0
domains = ['Nordenskiold', 'Steenstrup', 'Kjer']
for domain in domains:
    for file_path in glob.glob(os.path.join(root_source_path, 'all', domain + '*_closed*')):
        print(file_path)
        # break
        os.remove(file_path)
    for file_path in glob.glob(os.path.join(root_source_path, 'domain', domain, '*_closed*')):
        print(file_path)
        # break
        os.remove(file_path)
    for file_path in glob.glob(os.path.join(root_source_path, 'quality_assurance', domain, '*polygon*')):
        print(file_path)
        # break
        os.remove(file_path)
