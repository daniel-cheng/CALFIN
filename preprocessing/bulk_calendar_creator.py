# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 22:38:07 2019

@author: Daniel
"""
import os, datetime
from collections import defaultdict
import pandas as pd

source_path = r'D:\Daniel\Documents\Github\CALFIN Repo\preprocessing\calvingfrontmachine\landsat_raw'
domain_validation_calendar = defaultdict(lambda: dict((k, 0) for k in range(1972, datetime.datetime.now().year)))

def output_calendar_csv():
	"""Creates a csv file with the number of images per year per domain."""
	calendar_name = 'all.csv'
	calendar_path = os.path.join(r'D:\Daniel\Documents\Github\CALFIN Repo\preprocessing\calendars', calendar_name)
	pd.DataFrame.from_dict(data=domain_validation_calendar, orient='columns').to_csv(calendar_path, header=True)
	
if __name__ == '__main__':
	for domain in os.listdir(source_path):
		domain_path = os.path.join(source_path, domain)
		for file_name in os.listdir(domain_path):
			year = int(file_name.split('_')[3].split('-')[0])
			domain_validation_calendar[domain][year] += 1
	
	output_calendar_csv()
