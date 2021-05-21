# This is an automatic copy of the script that processed the data in this directory.
# The script original location was /home/alf/scripts_and_codes/repos/phd-scripts/tdc/TDC_V1_SW_28_10_19_analysis_scripts/analyze_and_plot_time_resolution_FOR_TIPP.py
# The timestamp for this processing is 20210513170525.
# The local variables in the script at the moment this copy was made were:
# directory: /home/alf/cernbox/measurements_data/TDC/TDC_V1_SW_28_10_19/20210429171133_TDC_V1_SW_28_10_19_full_characterization
# -----------------------------------
from data_processing_bureaucrat.Bureaucrat import Bureaucrat
import numpy as np
from pathlib import Path
import myplotlib as mpl
import pandas
from utils import TDCOutputDataType

TDC_NUMBERS = [2]

COLORS = [
	np.array((232, 237, 85))/255,
	np.array((255, 98, 87))/255,
	np.array((115, 222, 73))/255,
	np.array((91, 153, 252))/255,
]
COLORS_DARK = [
	np.array((158, 163, 0))/255,
	np.array((161, 18, 8))/255,
	np.array((35, 122, 0))/255,
	np.array((0, 47, 122))/255,
]

COLORS = [tuple(c) for c in COLORS]
COLORS_DARK = [tuple(c) for c in COLORS_DARK]

def script_core(directory):
	bureaucrat = Bureaucrat(
		directory,
		variables = locals(), # <-- Variables were registered at this point: {'directory': '/home/alf/cernbox/measurements_data/TDC/TDC_V1_SW_28_10_19/20210429171133_TDC_V1_SW_28_10_19_full_characterization'}
	)
	fig_mean = mpl.manager.new(
		title = f'Mean time vs output',
		show_title = False,
		# ~ subtitle = f'Dataset: {bureaucrat.measurement_name}',
		xlabel = f'TDC output',
		ylabel = f'Mean time (s)',
	)
	fig_quant = mpl.manager.new(
		title = f'Time resolution vs output',
		show_title = False,
		# ~ subtitle = f'Dataset: {bureaucrat.measurement_name}',
		xlabel = f'TDC output',
		ylabel = f'Temporal dispersion (s)',
	)
	fig_quant_distribution = mpl.manager.new(
		title = f'Time resolution histograms',
		show_title = False,
		# ~ subtitle = f'Dataset: {bureaucrat.measurement_name}',
		xlabel = f'Temporal dispersion (s)',
		ylabel = f'Number of outputs',
	)
	
	for n_TDC in TDC_NUMBERS:
		try:
			data = pandas.read_csv(
				bureaucrat.processed_by_script_dir_path('calculate_time_resolution.py')/Path(f'time_resolution_data_TDC_{n_TDC}.csv'),
				dtype = str, # Read everything as string to avoid errors in converting binary to decimals.
			)
			data['q95-q05 (s)'] = data['q95-q05 (s)'].apply(lambda x: float(x))
			data['Mean time (s)'] = data['Mean time (s)'].apply(lambda x: float(x) + 300e-12)
			data['std (s)'] = data['std (s)'].apply(lambda x: float(x))
		except FileNotFoundError as e:
			print(f'Cannot process data for TDC number {n_TDC}, reason: {e}')
			continue
		data['Output'] = data['Output'].apply(lambda x: TDCOutputDataType(COUNT=x.split('|')[0], SAFF=x.split('|')[-1].replace('(','').replace(')','')))
		
		data = data[data['Output'].apply(lambda x: not x.isnan and x.COUNT < 20 and x != TDCOutputDataType(COUNT=16, SAFF='000000000000000000000') and x.COUNT not in [7,11,15])]
		
		data = data.sort_values('Output')
		
		fig_mean.plot(
			x = [f'{o}' for o in data['Output']],
			y = data['Mean time (s)'],
			marker = '.',
			color = COLORS[n_TDC-1],
		)
		fig_quant.plot(
			[f'{o}' for o in data['Output']],
			data['q95-q05 (s)'],
			marker = '.',
			label = f'q<sub>95 %</sub> - q<sub>5 %</sub>',
			color = COLORS[n_TDC-1],
		)
		data_for_average = data[data['q95-q05 (s)']<100e-12]
		fig_quant.plot(
			[f'{o}' for o in data_for_average['Output']],
			data_for_average['q95-q05 (s)'].rolling(window=55, center=True, min_periods=1).mean(),
			marker = '.',
			label = f'Moving average',
			color = COLORS_DARK[n_TDC-1],
		)
		fig_quant.plot(
			[f'{o}' for o in data['Output']],
			data['std (s)'],
			marker = '.',
			label = f'std',
			color = COLORS[n_TDC],
		)
		data_for_average = data[data['std (s)']<100e-12]
		fig_quant.plot(
			[f'{o}' for o in data_for_average['Output']],
			data_for_average['std (s)'].rolling(window=55, center=True, min_periods=1).mean(),
			marker = '.',
			label = f'Moving average',
			color = COLORS_DARK[n_TDC],
		)
		fig_quant_distribution.hist(
			data['q95-q05 (s)'],
			label = f'q<sub>95 %</sub> - q<sub>5 %</sub>',
			color = COLORS[n_TDC-1],
		)
		fig_quant_distribution.hist(
			data['std (s)'],
			label = f'std',
			color = COLORS[n_TDC],
		)
	mpl.manager.save_all(mkdir = bureaucrat.processed_data_dir_path)
	
if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser(description='Parses the raw data to get parameters such as amplitude, collected charge, etc.')
	parser.add_argument(
		'--dir',
		metavar = 'path', 
		help = 'Path to the base directory of a measurement.',
		required = True,
		dest = 'directory',
		type = str,
	)
	args = parser.parse_args()
	script_core(args.directory)
