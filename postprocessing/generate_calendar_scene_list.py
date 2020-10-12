import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

import sys, os, glob

from plotting import plot_histogram, plot_scatter

def landsat_sort(file_name):
	"""Sorting key function derives date from landsat file path."""
	return file_name.split('_')[3]


def concatenate_and_move_scene_lists():
	input_path = r'../downloader/scenes'
	output_path = r'../downloader/scenes'
	
	domains = defaultdict(int)
	
	for file_name in os.listdir(input_path):
		if '_' in file_name:
			domains[file_name.split('_')[0]] += 1
	for domain in domains.keys():
		domain_path = os.path.join(input_path, domain + '_*[0-9].txt')
		output_file_path = os.path.join(output_path, domain + '_scenes.txt')
		with open(output_file_path, 'w') as scene_file:
			lines = []
			for file_path in sorted(glob.glob(domain_path)):
				with open(file_path, 'r') as partial_scene_file:
					for line in reversed(partial_scene_file.readlines()):
						if len(line.strip()) > 0:
							lines.append(line)
			lines.sort(key = landsat_sort)
			for line in lines:
				scene_file.write(line.strip() + '/n')

def concatenate_all_scene_lists():
	input_path = r'../downloader/scenes'
	output_path = r'../downloader/scenes'
	
	domains = defaultdict(int)
	
	for file_name in os.listdir(input_path):
		if '_' in file_name:
			domains[file_name.split('_')[0]] += 1
	output_file_path = os.path.join(output_path, 'all_scenes.txt')
	with open(output_file_path, 'w') as all_scene_file:
		lines = []
		for domain in domains.keys():
			domain_file_path = os.path.join(input_path, domain + '_scenes.txt')
			with open(domain_file_path, 'r') as domain_scene_file:
				for line in domain_scene_file.readlines():
					if len(line.strip()) > 0:
						lines.append(line)
		lines.sort(key = landsat_sort)
		lines = list(dict.fromkeys(lines)) #remove duplicates
		for line in lines:
			all_scene_file.write(line.strip() + '/n')
if __name__ == '__main__':
#	concatenate_and_move_scene_lists()
	concatenate_all_scene_lists()