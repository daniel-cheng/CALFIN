from scipy.interpolate import interp1d
import numpy as np

def redistribute_points(x_in, y_in, resolution):
	#redistribute_points Spreads points evenly on a line such that distance
	#between points on the parameteric line l(t) = (x(t), y(t)) is equal
	#relative to differences in t
	
	path_xy = np.stack((x_in, y_in));
	path_diff = np.diff(path_xy, axis=1)**2
	path_sum = np.sum(path_diff, axis=0)
	path_lengths = np.sqrt(path_sum)
	path_lengths = np.append(np.array([0]), path_lengths) #add the starting point
	path_cumalative = np.cumsum(path_lengths)
	
	final_step_locations = np.linspace(0, path_cumalative[-1], resolution)
	final_xy = interp1d(path_cumalative, path_xy)(final_step_locations)
	return final_xy[0], final_xy[1]
	
#Try measuring difference between x
#calcualte once for t, x and t, y, then combine breakpoints (error = max(error(x), error(y))