# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 18:06:26 2019

@author: Daniel
"""

import matplotlib.pyplot as plt

if __name__ == '__main__':
	
	domains = metrics['domain_mean_deviations_meters'].keys()
	x_pos = np.arange(len(domains))
	means_pixels = []
	means_meters = []
	error_pixels = []
	error_meters = []
	for domain in domains:
		sqrt_point_samples = len(metrics['domain_validation_distances_meters'][domain])
		sqrt_image_samples = len(metrics['domain_mean_deviations_meters'][domain])
		domain_mean_deviation_images_pixels = np.nanmean(metrics['domain_mean_deviations_pixels'][domain])
		domain_mean_deviation_images_meters = np.nanmean(metrics['domain_mean_deviations_meters'][domain])
		domain_std_deviation_points_pixels = np.nanstd(metrics['domain_validation_distances_pixels'][domain]) / sqrt_point_samples * 1.96
		domain_std_deviation_points_meters = np.nanstd(metrics['domain_validation_distances_meters'][domain]) / sqrt_point_samples * 1.96
		domain_std_deviation_images_pixels = np.nanstd(metrics['domain_mean_deviations_pixels'][domain]) / sqrt_image_samples * 1.96
		domain_std_deviation_images_meters = np.nanstd(metrics['domain_mean_deviations_meters'][domain]) / sqrt_image_samples * 1.96
		means_pixels.append(domain_mean_deviation_images_pixels)
		means_meters.append(domain_std_deviation_images_meters)
		error_pixels.append(domain_mean_deviation_images_pixels)
		error_meters.append(domain_std_deviation_images_pixels)
		

	# Build the plot
	fig, ax = plt.subplots()
	ax.bar(x_pos, means_pixels, yerr=error_pixels, align='center', alpha=0.5, ecolor='black', capsize=10)
	ax.set_ylabel('Mean Distance (pixels)')
	ax.set_xticks(x_pos)
	ax.set_xticklabels(domains, rotation='vertical')
	ax.set_title('Mean Distance per basin (pixels)')
	ax.yaxis.grid(True)
	plt.show()
		