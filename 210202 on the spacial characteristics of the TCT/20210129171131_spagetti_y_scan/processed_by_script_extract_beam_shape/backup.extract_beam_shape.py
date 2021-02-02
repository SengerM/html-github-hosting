# This is an automatic copy of the script that processed the data in this directory.
# The script original location was /home/alf/scripts_and_codes/repos/phd-scripts/spagetti/extract_beam_shape.py
# The timestamp for this processing is 20210202083547.
# The local variables in the script at the moment this copy was made were:
# directory: /home/alf/cernbox/measurements/spagetti/20210129171131_spagetti_y_scan
# distance_range: (0.000466, inf)
# -----------------------------------
from data_processing_bureaucrat.Bureaucrat import Bureaucrat
import numpy as np
from pathlib import Path
import myplotlib as mpl
import pandas
from scipy.fft import fft, ifft
import lmfit
from scipy.signal import find_peaks

COLLECTED_CHARGE_COLUMN = 'Collected charge (a.u.)'
PITCH = 92e-6
METAL_STRIP_WIDTH = PITCH*20/80#12e-6
BEAM_SIZE = 4.5e-6
OFFSET = 25e-6

def spaghetti_profile(x, pitch=80e-6, duty_cycle=20/80, offset=0):
	prof = np.zeros(x.shape) + 1e-3
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

def script_core(directory, distance_range: tuple):
	bureaucrat = Bureaucrat(
		directory,
		variables = locals(), # <-- Variables were registered at this point: {'directory': '/home/alf/cernbox/measurements/spagetti/20210129171131_spagetti_y_scan', 'distance_range': (0.000466, inf)}
	)
	
	data = pandas.read_csv(
		bureaucrat.processed_by_script_dir_path('linear_scan_many_triggers_per_point.py')/Path('measured_data.csv'),
		sep = '\t',
	)
	
	n_steps = sorted(list(set(data['n_step'])))
	for idx,n in enumerate(n_steps):
		if idx == 0:
			distance = []
			start_point = (list(data[data['n_step']==n]['x (m)'])[0], list(data[data['n_step']==n]['y (m)'])[0])
			average_collected_charge = []
		distance.append(
			np.linalg.norm(np.array((list(data[data['n_step']==n]['x (m)'])[0], list(data[data['n_step']==n]['y (m)'])[0])) - np.array(start_point))
		)
		average_collected_charge.append(
			np.mean(data[data['n_step']==n][COLLECTED_CHARGE_COLUMN])
		)
	distance = np.array(distance)
	average_collected_charge = np.array(average_collected_charge)
	normalized_collected_charge = average_collected_charge/max(average_collected_charge)*1.05
	
	use_indices = (distance_range[0] < distance) & (distance < distance_range[1])
	distance = distance[use_indices]
	average_collected_charge = average_collected_charge[use_indices]
	normalized_collected_charge = normalized_collected_charge[use_indices]
	
	# Perform fit ---------
	model = lmfit.Model(intensity_profile)
	params = model.make_params()
	params['pitch'].set(
		value = 80e-6, 
		vary = False,
		min = 50e-6,
		max = 110e-6,
	)
	params['duty_cycle'].set(
		value = 20/80, 
		vary = False,
		min = 0,
		max = 1,
	)
	params['offset'].set(
		value = 2.2380234663808414e-05,
		vary = False,
		min = 15e-6,
		max = 25e-6,
	)
	params['beam_size'].set(
		value = 5.5e-6,
		vary = False,
		min = 1e-6,
		max = 20e-6,
	)
	fit_results = model.fit(
		average_collected_charge, 
		params, 
		x = distance,
	)
	print(f'pitch = {fit_results.params["pitch"].value}')
	print(f'offset = {fit_results.params["offset"].value}')
	print(f'beam_size = {fit_results.params["beam_size"].value}')
	print(f'duty_cycle = {fit_results.params["duty_cycle"].value}')
	
	fig = mpl.manager.new(
		title = f'Scan vs spaghetti profile',
		subtitle = f'Measurement: {bureaucrat.measurement_name}',
		xlabel = 'Distance (m)',
	)
	fig.plot(
		distance,
		normalized_collected_charge,
		label = 'Nomalized collected charge',
		marker = '.',
	)
	fig.plot(
		distance,
		spaghetti_profile(
			distance,
			duty_cycle = fit_results.params["duty_cycle"].value,
			pitch = fit_results.params["pitch"].value,
			offset = fit_results.params["offset"].value, 
		),
		label = f'Spaghetti profile',
	)
	fig.plot(
		distance,
		intensity_profile(
			x = distance, 
			duty_cycle = fit_results.params["duty_cycle"].value,
			pitch = fit_results.params["pitch"].value,
			offset = fit_results.params["offset"].value, 
			beam_size = fit_results.params["beam_size"].value,
		),
		label = f'Fit (D={int(fit_results.params["pitch"].value*1e6)} µm, d={int(fit_results.params["pitch"].value*1e6*fit_results.params["duty_cycle"].value)} µm, o={int(fit_results.params["offset"].value*1e6)} µm, σ={fit_results.params["beam_size"].value*1e6:.2f} µm)',
	)
	
	# ~ # Deconvolution all peaks at the same time -------------------------
	# ~ extended_distance = np.array(list(distance) + list(distance + distance.max() - distance.min() + np.diff(distance)[0]))
	# ~ extended_distance = extended_distance[extended_distance < distance.max() + fit_results.params["pitch"].value]
	# ~ extended_distance = np.array(list(extended_distance) + [extended_distance.max()+np.diff(extended_distance)[0]])
	# ~ inverted_spaghetti_profile = 1/(spaghetti_profile(
			# ~ extended_distance,
			# ~ duty_cycle = fit_results.params["duty_cycle"].value,
			# ~ pitch = fit_results.params["pitch"].value,
		# ~ )
	# ~ )
	# ~ inverted_spaghetti_profile -= inverted_spaghetti_profile.min()
	# ~ inverted_spaghetti_profile /= inverted_spaghetti_profile.max()
	# ~ laser_profile_deconv = np.convolve(
		# ~ normalized_collected_charge, 
		# ~ inverted_spaghetti_profile,
		# ~ mode = 'valid',
	# ~ )
	# ~ laser_profile_deconv *= -1 # Don't understand why I have to multiply by -1
	# ~ laser_profile_deconv -= min(laser_profile_deconv)
	# ~ laser_profile_deconv /= max(laser_profile_deconv)
	
	# ~ fig = mpl.manager.new(
		# ~ title = 'Total deconvolution',
		# ~ subtitle = f'Measurement: {bureaucrat.measurement_name}',
		# ~ xlabel = 'Distance (m)',
	# ~ )
	# ~ fig.plot(
		# ~ distance,
		# ~ normalized_collected_charge,
		# ~ label = 'Normalized collected charge',
	# ~ )
	# ~ fig.plot(
		# ~ extended_distance,
		# ~ inverted_spaghetti_profile,
		# ~ label = 'Inverted spaghetti profile'
	# ~ )
	# ~ fig.plot(
		# ~ [i*np.diff(distance)[0] for i in range(len(laser_profile_deconv))],
		# ~ laser_profile_deconv,
		# ~ label = 'Deconvoluted laser profile',
		# ~ marker = '.',
	# ~ )
	# ~ fig.plot(
		# ~ [i*np.diff(distance)[0] for i in range(len(laser_profile_deconv)*2-1)],
		# ~ list(laser_profile_deconv) + list(laser_profile_deconv)[1:],
		# ~ label = 'Deconvoluted laser profile extended',
		# ~ marker = '.',
	# ~ )
	
	# Deconvolution of individual peaks --------------------------------
	extended_distance = distance[distance < distance.min() + fit_results.params["pitch"].value]
	convolute_with = spaghetti_profile(
		extended_distance,
		duty_cycle = fit_results.params["duty_cycle"].value,
		pitch = fit_results.params["pitch"].value,
		offset = fit_results.params["offset"].value, 
	)
	convolute_with **= -1
	convolute_with -= min(convolute_with)
	convolute_with /= max(convolute_with)
	laser_profile_deconv = np.convolve(
		normalized_collected_charge, 
		convolute_with,
		mode = 'valid',
	)
	laser_profile_deconv *= -1 # Don't understand why I have to multiply by -1
	laser_profile_deconv -= min(laser_profile_deconv)
	laser_profile_deconv /= max(laser_profile_deconv)
	peaks, _ = find_peaks(laser_profile_deconv)
	peaks = peaks[laser_profile_deconv[peaks]>.5]
	distance_for_laser_profile_deconv = distance[:-len(extended_distance)+1]
	
	laser_deconv_fit_results = {}
	x_vals_for_fitting_gaussian = {}
	y_vals_for_fitting_gaussian = {}
	for idx, peak in enumerate(peaks):
		if idx == 0 or idx == len(peaks)-1: continue
		model = lmfit.Model(gaussian)
		params = model.make_params()
		params['mu'].set(
			value = distance_for_laser_profile_deconv[peak], 
			vary = True,
			min = distance_for_laser_profile_deconv[peak] - fit_results.params["duty_cycle"].value*fit_results.params["pitch"].value,
			max = distance_for_laser_profile_deconv[peak] + fit_results.params["duty_cycle"].value*fit_results.params["pitch"].value,
		)
		params['sigma'].set(
			value = distance_for_laser_profile_deconv[peak], 
			vary = True,
			min = 0,
		)
		x_vals_for_fitting_gaussian[peak] = distance_for_laser_profile_deconv[peak-int(fit_results.params["pitch"].value/np.diff(distance)[0]/2+2):peak+int(fit_results.params["pitch"].value/np.diff(distance)[0]/2-2)]
		y_vals_for_fitting_gaussian[peak] = laser_profile_deconv[peak-int(fit_results.params["pitch"].value/np.diff(distance)[0]/2+2):peak+int(fit_results.params["pitch"].value/np.diff(distance)[0]/2-2)]
		y_vals_for_fitting_gaussian[peak] /= y_vals_for_fitting_gaussian[peak].max()
		laser_deconv_fit_results[peak] = model.fit(
			y_vals_for_fitting_gaussian[peak], 
			params, 
			x = x_vals_for_fitting_gaussian[peak],
		)
		
	fig = mpl.manager.new(
		title = 'Single peak deconvolution',
		subtitle = f'Measurement: {bureaucrat.measurement_name}',
		xlabel = 'Distance (m)',
	)
	fig.plot(
		distance,
		normalized_collected_charge,
		label = 'Normalized_collected_charge',
		marker = '.',
	)
	fig.plot(
		extended_distance,
		convolute_with,
		label = 'Inverse spaghetti'
	)
	fig.plot(
		distance_for_laser_profile_deconv,
		laser_profile_deconv,
		label = 'Normalized laser profile',
		marker = '.',
	)
	
	
	fig = mpl.manager.new(
		title = 'Gaussians fitted to each peak',
		subtitle = f'Measurement: {bureaucrat.measurement_name}',
		xlabel = 'Distance (m)',
	)
	fig.plot(
		distance_for_laser_profile_deconv,
		laser_profile_deconv,
		label = 'Normalized laser profile',
		marker = '.',
	)
	for idx, peak in enumerate(peaks):
		if idx == 0 or idx == len(peaks)-1: continue
		fig.plot(
			x_vals_for_fitting_gaussian[peak],
			gaussian(
				x_vals_for_fitting_gaussian[peak],
				laser_deconv_fit_results[peak].params['mu'],
				laser_deconv_fit_results[peak].params['sigma'],
			),
			label = f'µ={laser_deconv_fit_results[peak].params["mu"]*1e6:.2f} µm, σ={laser_deconv_fit_results[peak].params["sigma"]*1e6:.2f} µm'
		)
	
	fig = mpl.manager.new(
		title = 'Distribution of laser widths',
		subtitle = f'Measurement: {bureaucrat.measurement_name}',
		xlabel = 'σ (µm)',
		ylabel = 'Number of events',
	)
	fig.hist(
		[laser_deconv_fit_results[peak].params["sigma"]*1e6 for peak in peaks[1:-1]],
		bins = 'auto',
	)
	
	fig = mpl.manager.new(
		title = 'Estimated laser profile',
		subtitle = f'Measurement: {bureaucrat.measurement_name}',
		xlabel = 'Distance (m)',
		ylabel = 'Laser intensity (normalized)',
	)
	laser_sigma_final_estimation = np.mean([laser_deconv_fit_results[peak].params["sigma"] for peak in peaks[1:-1]])
	x_axis = np.linspace(-4*laser_sigma_final_estimation,4*laser_sigma_final_estimation,99)
	fig.plot(
		x_axis,
		gaussian(x_axis, mu=0, sigma=laser_sigma_final_estimation),
		label = f'σ = {laser_sigma_final_estimation*1e6:.2f} µm'
	)
	
	# Fourier space ----------------------------------------------------
	frequency_axis = np.fft.fftfreq(len(distance), d=np.diff(distance)[0])[:int(len(distance)/2)]
	collected_charge_fft = np.fft.fft(normalized_collected_charge-normalized_collected_charge.mean())[:int(len(distance)/2)]
	collected_charge_fft /= np.abs(collected_charge_fft).max()
	
	spaghetti_profile_fft = np.fft.fft(
		spaghetti_profile(
			distance,
			duty_cycle = fit_results.params["duty_cycle"].value,
			pitch = fit_results.params["pitch"].value,
			offset = fit_results.params["offset"].value, 
		) - spaghetti_profile(
			distance,
			duty_cycle = fit_results.params["duty_cycle"].value,
			pitch = fit_results.params["pitch"].value,
			offset = fit_results.params["offset"].value, 
		).mean()
	)[:int(len(distance)/2)]
	laser_fft = np.fft.fft(
		gaussian(
			distance, 
			mu = distance.mean(), 
			sigma = fit_results.params["beam_size"].value,
		) - gaussian(
			distance, 
			mu = distance.mean(), 
			sigma = fit_results.params["beam_size"].value,
		).mean()
	)[:int(len(distance)/2)]
	intensity_model_fft = np.fft.fft(
		intensity_profile(
			x = distance, 
			duty_cycle = fit_results.params["duty_cycle"].value,
			pitch = fit_results.params["pitch"].value,
			offset = fit_results.params["offset"].value, 
			beam_size = fit_results.params["beam_size"].value,
		) - intensity_profile(
			x = distance, 
			duty_cycle = fit_results.params["duty_cycle"].value,
			pitch = fit_results.params["pitch"].value,
			offset = fit_results.params["offset"].value, 
			beam_size = fit_results.params["beam_size"].value,
		).mean()
	)[:int(len(distance)/2)]
	spaghetti_profile_fft /= np.abs(spaghetti_profile_fft).max()
	laser_fft /= np.abs(laser_fft).max()
	intensity_model_fft /= np.abs(intensity_model_fft).max()
	
	fig = mpl.manager.new(
		title = f'Fourier space',
		subtitle = f'Measurement: {bureaucrat.measurement_name}',
		xlabel = 'Frequency (m⁻¹)',
		xscale = 'log',
	)
	fig.plot(
		frequency_axis,
		np.abs(collected_charge_fft),
		label = 'Collected charge FFT',
		marker = '.',
	)
	fig.plot(
		frequency_axis,
		np.abs(spaghetti_profile_fft),
		label = 'Spaghetti profile FFT',
		marker = '.',
	)
	fig.plot(
		frequency_axis,
		np.abs(intensity_model_fft),
		label = f'Fit (D={int(fit_results.params["pitch"].value*1e6)} µm, d={int(fit_results.params["pitch"].value*1e6*fit_results.params["duty_cycle"].value)} µm, o={int(fit_results.params["offset"].value*1e6)} µm, σ={fit_results.params["beam_size"].value*1e6:.2f} µm)',
		marker = '.',
	)
	fig.plot(
		frequency_axis,
		np.abs(laser_fft),
		label = 'Laser FFT',
		marker = '.',
	)
	# ~ fig.plot(
		# ~ frequency_axis,
		# ~ np.abs(collected_charge_fft/spaghetti_profile_fft),
		# ~ label = 'Q/spaghetti',
		# ~ marker = '.',
	# ~ )
	
	
	# ~ mpl.manager.show()
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
	)
