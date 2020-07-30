import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

import sys, os
sys.path.insert(1, '../training/keras-deeplab-v3-plus')
sys.path.insert(2, '../training')

from plotting import plot_histogram, plot_scatter

def output_calendar_csv():
	manual_qa_path = r'D:\Daniel\Documents\Github\CALFIN Repo\outputs\mask_extractor\quality_assurance'
	auto_qa_path = r'D:\Daniel\Documents\Github\CALFIN Repo\outputs\production\quality_assurance'
	auto_qa_bad_path = r'D:\Daniel\Documents\Github\CALFIN Repo\outputs\production\quality_assurance_bad'
	
	domain_validation_calendar = defaultdict(lambda: defaultdict(str))
	original_names = []
	scene_ids = []
	
	for domain in os.listdir(auto_qa_bad_path):
		domain_path = os.path.join(auto_qa_bad_path, domain)
		if os.path.isdir(domain_path):
			for file_name in os.listdir(domain_path):
				file_path = os.path.join(domain_path, file_name)
				file_name_parts = file_name.split('_')
				domain = file_name_parts[0]
				date = file_name_parts[3]
				domain_validation_calendar[domain][date] = "UNPICKED"
	
	for domain in os.listdir(auto_qa_path):
		domain_path = os.path.join(auto_qa_path, domain)
		if os.path.isdir(domain_path):
			for file_name in os.listdir(domain_path):
				file_path = os.path.join(domain_path, file_name)
				file_name_parts = file_name.split('_')
				domain = file_name_parts[0]
				date = file_name_parts[3]
				domain_validation_calendar[domain][date] = "AUTO-GOOD"
	
	for domain in os.listdir(manual_qa_path):
		domain_path = os.path.join(manual_qa_path, domain)
		if os.path.isdir(domain_path):
			for file_name in os.listdir(domain_path):
				file_path = os.path.join(domain_path, file_name)
				file_name_parts = file_name.split('_')
				domain = file_name_parts[0]
				date = file_name_parts[3]
				domain_validation_calendar[domain][date] = "MANUAL"
			
	"""Creates a csv file with the number of images per year per domain."""
	calendar_name = 'calendar_expanded_updated.csv'
	calendar_path = os.path.join(auto_qa_path, calendar_name)
	pd.DataFrame.from_dict(data=domain_validation_calendar, orient='columns').to_csv(calendar_path, header=True)
	return domain_validation_calendar

if __name__ == '__main__':
	domain_validation_calendar = output_calendar_csv()