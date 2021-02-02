# This is an automatic copy of the script that processed the data in this directory.
# The script original location was scan_z_to_find_focus.py
# The timestamp for this processing is 20210129150154.
# The local variables in the script at the moment this copy was made were:
# __name__: __main__
# __doc__: None
# __package__: None
# __loader__: <_frozen_importlib_external.SourceFileLoader object at 0x0000023F80941AF0>
# __spec__: None
# __annotations__: {}
# __builtins__: <module 'builtins' (built-in)>
# __file__: scan_z_to_find_focus.py
# __cached__: None
# LecroyWR640Zi: <class 'PyticularsTCT.oscilloscope.LecroyWR640Zi'>
# TCTStages: <class 'PyticularsTCT.tct_setup.TCTStages'>
# save_4ch_trigger: <function save_4ch_trigger at 0x0000023F87B20DC0>
# np: <module 'numpy' from 'C:\\Users\\tct_cms\\AppData\\Roaming\\Python\\Python38\\site-packages\\numpy\\__init__.py'>
# sleep: <built-in function sleep>
# os: <module 'os' from 'C:\\Program Files\\WindowsApps\\PythonSoftwareFoundation.Python.3.8_3.8.1776.0_x64__qbz5n2kfra8p0\\lib\\os.py'>
# mpl: <module 'myplotlib' from 'c:\\users\\tct_cms\\desktop\\myplotlib\\myplotlib\\__init__.py'>
# save_tct_trigger: <function save_tct_trigger at 0x0000023F87B20F70>
# read_tct_trigger: <function read_tct_trigger at 0x0000023F87B3C040>
# LGADSignal: <class 'lgadtools.LGADSignal.LGADSignal'>
# random: <module 'random' from 'C:\\Program Files\\WindowsApps\\PythonSoftwareFoundation.Python.3.8_3.8.1776.0_x64__qbz5n2kfra8p0\\lib\\random.py'>
# Bureaucrat: <class 'data_processing_bureaucrat.Bureaucrat.Bureaucrat'>
# Path: <class 'pathlib.Path'>
# N_STEPS: 222
# X_POSITION: -0.004859150390625
# Y_POSITION: -0.023545966796875
# Z_START: 0.06267748046874999
# Z_END: 0.07267748046874999
# N_AVERAGE_TRIGGERS: 11
# -----------------------------------
from PyticularsTCT.oscilloscope import LecroyWR640Zi # https://github.com/SengerM/PyticularsTCT
from PyticularsTCT.tct_setup import TCTStages # https://github.com/SengerM/PyticularsTCT
from PyticularsTCT.utils import save_4ch_trigger # https://github.com/SengerM/PyticularsTCT
import numpy as np
from time import sleep
import os
import myplotlib as mpl
from PyticularsTCT.utils import save_tct_trigger, read_tct_trigger
from lgadtools.LGADSignal import LGADSignal # https://github.com/SengerM/lgadtools
import random
from data_processing_bureaucrat.Bureaucrat import Bureaucrat # https://github.com/SengerM/data_processing_bureaucrat
from pathlib import Path

############################################################

N_STEPS = 222
X_POSITION = -4.8591503906249995e-3
Y_POSITION = -23.545966796875003e-3
Z_START = 67.67748046874999e-3 - 5e-3
Z_END = Z_START + 10e-3
N_AVERAGE_TRIGGERS = 11

############################################################

bureaucrat = Bureaucrat(
	str(Path(f'C:/Users/tct_cms/Desktop/TCT_measurements_data/{input("Measurement name? ")}')),
	variables = locals(), # <-- Variables were registered at this point: {'__name__': '__main__', '__doc__': None, '__package__': None, '__loader__': <_frozen_importlib_external.SourceFileLoader object at 0x0000023F80941AF0>, '__spec__': None, '__annotations__': {}, '__builtins__': <module 'builtins' (built-in)>, '__file__': 'scan_z_to_find_focus.py', '__cached__': None, 'LecroyWR640Zi': <class 'PyticularsTCT.oscilloscope.LecroyWR640Zi'>, 'TCTStages': <class 'PyticularsTCT.tct_setup.TCTStages'>, 'save_4ch_trigger': <function save_4ch_trigger at 0x0000023F87B20DC0>, 'np': <module 'numpy' from 'C:\\Users\\tct_cms\\AppData\\Roaming\\Python\\Python38\\site-packages\\numpy\\__init__.py'>, 'sleep': <built-in function sleep>, 'os': <module 'os' from 'C:\\Program Files\\WindowsApps\\PythonSoftwareFoundation.Python.3.8_3.8.1776.0_x64__qbz5n2kfra8p0\\lib\\os.py'>, 'mpl': <module 'myplotlib' from 'c:\\users\\tct_cms\\desktop\\myplotlib\\myplotlib\\__init__.py'>, 'save_tct_trigger': <function save_tct_trigger at 0x0000023F87B20F70>, 'read_tct_trigger': <function read_tct_trigger at 0x0000023F87B3C040>, 'LGADSignal': <class 'lgadtools.LGADSignal.LGADSignal'>, 'random': <module 'random' from 'C:\\Program Files\\WindowsApps\\PythonSoftwareFoundation.Python.3.8_3.8.1776.0_x64__qbz5n2kfra8p0\\lib\\random.py'>, 'Bureaucrat': <class 'data_processing_bureaucrat.Bureaucrat.Bureaucrat'>, 'Path': <class 'pathlib.Path'>, 'N_STEPS': 222, 'X_POSITION': -0.004859150390625, 'Y_POSITION': -0.023545966796875, 'Z_START': 0.06267748046874999, 'Z_END': 0.07267748046874999, 'N_AVERAGE_TRIGGERS': 11}
	new_measurement = True,
)

def measure():
	osc = LecroyWR640Zi('USB0::0x05FF::0x1023::4751N40408::INSTR')
	stages = TCTStages()

	print('Moving to start position...')
	stages.move_to(
		z = Z_START,
		x = X_POSITION,
		y = Y_POSITION,
	)
	print(f'Current position is {stages.position} m')
	print('Measuring...')
	
	for nz,z in enumerate(np.linspace(Z_START,Z_END,N_STEPS)):
		print('#############################')
		print(f'nz = {nz}')
		stages.move_to(z=z)
		print(f'Current position is {stages.position} m')
		print('Acquiring signals...')
		sleep(0.01)
		osc.trig_mode = 'AUTO'
		sleep(0.01)
		osc.trig_mode = 'SINGLE'
		data = osc.get_wf(CH=1)
		averaged_signal = np.array(data['volt'])
		for n_average in range(N_AVERAGE_TRIGGERS):
			osc.trig_mode = 'AUTO'
			sleep(0.01)
			osc.trig_mode = 'SINGLE'
			data = osc.get_wf(CH=1)
			averaged_signal += np.array(data['volt'])
		averaged_signal /= N_AVERAGE_TRIGGERS
		
		fname = f'{bureaucrat.raw_data_dir_path}/{nz:05d}.txt'.replace("/","\\")
		print(f'Saving data in {fname}...')
		save_tct_trigger(
			fname = fname,
			position = stages.position,
			time = data['time'],
			ch1 = averaged_signal,
		)
		temp_fig = mpl.manager.new(
			title = f'Raw signal for nz={nz}',
			xlabel = 'Time (s)',
			ylabel = 'Amplitude (V)',
		)
		temp_fig.plot(
			data['time'],
			averaged_signal,
		)
		mpl.manager.save_all(mkdir=f'{bureaucrat.processed_data_dir_path}/raw_signals_plots')
		mpl.manager.delete_all_figs()
			
	print('Finished measuring! :)')

def parse_amplitudes():
	print('Reading data...')
	raw_data = []
	for fname in sorted(os.listdir(bureaucrat.raw_data_dir_path)):
		raw_data.append(read_tct_trigger(f'{bureaucrat.raw_data_dir_path}/{fname}'))
	
	print('Calculating amplitudes...')
	amplitudes = []
	zs = []
	for data in raw_data:
		zs.append(data['position'][2])
		ch = list(data['volt'].keys())[0] # CH1, CH2, etc...
		signal = LGADSignal(
			time = data['time'],
			samples = data['volt'][ch]*-1,
		)
		amplitudes.append(signal.amplitude)
	
	fname = f'{bureaucrat.processed_data_dir_path}/amplitude_vs_z.txt'
	print(f'Saving parsed data to file {fname}...')
	with open(fname, 'w') as ofile:
		print('#z (m)\tAmplitude (V)', file = ofile)
		for z,A in zip(zs, amplitudes):
			print(f'{z}\t{A}', file=ofile)

def plot_amplitudes():
	data = np.genfromtxt(f'{bureaucrat.processed_data_dir_path}/amplitude_vs_z.txt').transpose()
	z = data[0]
	amplitude = data[1]
	mpl.manager.set_plotting_package('plotly')
	fig = mpl.manager.new(
		title = f'Focus find for PIN diode at x = {X_POSITION} and y = {Y_POSITION}',
		xlabel = 'z position (m)',
		ylabel = 'Amplitude (V)',
		package = 'plotly',
	)
	fig.plot(
		z,
		amplitude,
		marker = '.',
		label = 'Measured data',
	)
	
	fig.save(f'{bureaucrat.processed_data_dir_path}/plot.pdf')
	mpl.manager.show()

if __name__ == '__main__':
	print('Measuring...')
	measure()
	print('Parsing amplitudes...')
	parse_amplitudes()
	print('Plotting amplitudes...')
	plot_amplitudes()
