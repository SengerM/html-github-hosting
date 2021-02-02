# This is an automatic copy of the script that processed the data in this directory.
# The script original location was /home/alf/scripts_and_codes/repos/phd-scripts/spagetti/process_microscope_picture.py
# The timestamp for this processing is 20210201200301.
# The local variables in the script at the moment this copy was made were:
# directory: /home/alf/cernbox/measurements/spagetti/20210201000000_spaghetti_diode_in_microscope
# distance_range: (-inf, inf)
# datpath: /home/alf/cernbox/measurements/spagetti/20210201000000_spaghetti_diode_in_microscope/processed_by_ds9/S9_04A1_DC_50x51_21.dat
# scale: 459.98607
# -----------------------------------
from data_processing_bureaucrat.Bureaucrat import Bureaucrat
import numpy as np
from pathlib import Path
import myplotlib as mpl
from scipy.fft import fft, ifft

def spaghetti_profile(x, pitch=80e-6, duty_cycle=20/80, offset=0):
	prof = np.zeros(x.shape)
	prof[((x-offset)%pitch > pitch*duty_cycle)] = 1
	return prof

def gaussian(x, mu, sigma):
	return np.exp(-(x-mu)**2/2/sigma**2)

def intensity_profile(x, pitch=80e-6, duty_cycle=10/80, offset=0, beam_size=10e-6):
	if not isinstance(x, np.ndarray):
		raise TypeError(f'<x> must be a numpy array, received {x} of type {type(x)}.')
	extended_x = np.array(list(x-x.min()-x.max()-np.diff(x)[0]) + list(x-x.min()) + list(x-x.min()+x.max()+np.diff(x)[0])) + x.min()
	intensity = np.convolve(
		spaghetti_profile(extended_x, pitch, duty_cycle, offset),
		gaussian(extended_x, mu=extended_x.mean(), sigma=beam_size),
	)
	# ~ intensity = intensity[int(np.floor(len(intensity)/4)):int(np.ceil(len(intensity)*3/4))]
	intensity = intensity[int(np.floor(len(intensity)/2-len(x)/2)):int(np.ceil(len(intensity)/2+len(x)/2))]
	return intensity/intensity.max()

def script_core(directory, distance_range: tuple, datpath, scale):
	bureaucrat = Bureaucrat(
		directory,
		variables = locals(), # <-- Variables were registered at this point: {'directory': '/home/alf/cernbox/measurements/spagetti/20210201000000_spaghetti_diode_in_microscope', 'distance_range': (-inf, inf), 'datpath': '/home/alf/cernbox/measurements/spagetti/20210201000000_spaghetti_diode_in_microscope/processed_by_ds9/S9_04A1_DC_50x51_21.dat', 'scale': 459.98607}
	)
	data = np.genfromtxt(Path(datpath)).transpose()
	
	distances = data[0]/scale/1e3
	intensities = data[1]
	
	fig = mpl.manager.new(
		title = 'Intensity profile in the picture',
		subtitle = f'Measurement: {bureaucrat.measurement_name}',
		xlabel = 'Distance (m)',
	)
	fig.plot(
		distances,
		intensities/intensities.max(),
		marker = '.',
		label = 'Profile from the picture',
	)
	fig.plot(
		distances,
		spaghetti_profile(
			x = distances, 
			duty_cycle = 20/80,
			pitch = 80e-6,
			offset = 34e-6, 
		),
		label = 'Spaghetti "theoretical" profile',
	)
	mpl.manager.save_all(mkdir = bureaucrat.processed_data_dir_path)

if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser(description='Plot data from a linear scan in the TCT.')
	parser.add_argument(
		'--dir',
		metavar = 'path', 
		help = 'Path to the base directory of a measurement.',
		required = True,
		dest = 'directory',
		type = str,
	)
	parser.add_argument(
		'--dat',
		metavar = 'path', 
		help = 'Path to the ".dat" file containing the intensity profile along a line.',
		required = True,
		dest = 'datpath',
		type = str,
	)
	parser.add_argument(
		'--scale',
		metavar = 's', 
		help = 'Scale in px/mm',
		required = True,
		dest = 'scale',
		type = float,
	)
	parser.add_argument(
		'--start',
		metavar = 'x_min', 
		help = 'Data points with distance < x_min will not be used in the analysis.',
		required = False,
		dest = 'x_min',
		type = float,
		default = -float('inf'),
	)
	parser.add_argument(
		'--stop',
		metavar = 'x_max', 
		help = 'Data points with distance > x_max will not be used in the analysis.',
		required = False,
		dest = 'x_max',
		type = float,
		default = float('inf'),
	)
	args = parser.parse_args()
	script_core(
		args.directory,
		(args.x_min,args.x_max),
		args.datpath,
		args.scale,
	)
