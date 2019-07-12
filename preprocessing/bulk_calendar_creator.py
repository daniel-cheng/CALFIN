# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 22:38:07 2019

@author: Daniel
"""
import numpy as np
import os, shutil, glob
from skimage.io import imsave, imread
import csv

source_path = r'C:\Users\Daniel\AppData\Roaming\QGIS\QGIS3\profiles\default\python\plugins\calvingfrontmachine\landsat_raw'
dest_path = r'D:\Daniel\Documents\Github\CALFIN Repo\reprocessing\images_1024'
calendar_path = r'D:\Daniel\Documents\Github\CALFIN Repo\preprocessing\calendars'

domains = sorted(os.listdir(source_path))
domains_count = len(domains)


yearly_calendar_file = open(os.path.join(calendar_path, 'all.csv'), "w", newline='')
yearly_writer = csv.writer(yearly_calendar_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONE)
yearly_writer.writerow([''] + domains)
years = list(range(1972, 2020))
years_count = len(years)
yearly_counts = np.zeros((years_count, domains_count))

core_domains = sorted(['Jakobshavn', 'Upernavik-NE', 'Rink-Isbrae', 'Kangiata-Nunaata', 'Hayes', 'Kong-Oscar', 'Kangerlussuaq', 'Helheim'])
core_yearly_calendar_file = open(os.path.join(calendar_path, 'core.csv'), "w", newline='')
core_yearly_writer = csv.writer(core_yearly_calendar_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONE)
core_yearly_writer.writerow([''] + core_domains)
core_count = len(core_domains)
core_yearly_counts = np.zeros((years_count, core_count))
core_index = 0

for d in range(domains_count):
	domain = domains[d]
	monthly_counts = np.zeros((years_count, 12))
	monthly_domain_calendar_file = open(os.path.join(calendar_path, domain + '.csv'), "w", newline='')
	monthly_writer = csv.writer(monthly_domain_calendar_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONE)
	monthly_writer.writerow(['', 'January', 'Feburary', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'])

	for file_path in glob.glob(os.path.join(source_path, domain, '*B[0-9].png')):
		name = os.path.basename(file_path)
		name_parts = name.split('_')
		date = name_parts[3].split('-')
		year = int(date[0])
		month = int(date[1])
		
		monthly_counts[year - 1972, month - 1] += 1
		yearly_counts[year - 1972, d] += 1
		
	if domain in core_domains:
#		print(domain, core_index)
		core_yearly_counts[:, core_index] = yearly_counts[:, d]
		core_index += 1
		
	for year in years:
		monthly_writer.writerow([year] + list(monthly_counts[year - 1972,:].astype(np.uint8)))
	monthly_domain_calendar_file.close()

for year in years:
	yearly_writer.writerow([year] + list(yearly_counts[year - 1972,:].astype(np.uint8)))
yearly_calendar_file.close()
for year in years:
	core_yearly_writer.writerow([year] + list(core_yearly_counts[year - 1972,:].astype(np.uint8)))
core_yearly_calendar_file.close()