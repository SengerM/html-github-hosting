# This is an automatic copy of the script that processed the data in this directory.
# The script original location was /home/alf/scripts_and_codes/repos/phd-scripts/tdc/TDC_V1_SW_28_10_19_analysis_scripts/plot_output_distributions.py
# The timestamp for this processing is 20210513094745.
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

def script_core(directory):
	bureaucrat = Bureaucrat(
		directory,
		variables = locals(), # <-- Variables were registered at this point: {'directory': '/home/alf/cernbox/measurements_data/TDC/TDC_V1_SW_28_10_19/20210429171133_TDC_V1_SW_28_10_19_full_characterization'}
	)
	
	data = pandas.read_csv(
		bureaucrat.processed_by_script_dir_path('measure_TDC_structure_random_sampling.py')/Path('measured_data.csv'),
		dtype = str, # Read everything as string to avoid errors in converting binary to decimals.
	)
	# Now parse each column as it should be parsed:
	data['Delay (s)'] = data['Delay (s)'].apply(float)
	data['Delay (s)'] = data['Delay (s)'].apply(lambda x: -1*x + 300e-12)
	for n_TDC in TDC_NUMBERS:
		data[f'COUNT {n_TDC}'] = data[f'COUNT {n_TDC}'].apply(lambda x: int(x, 2))
	
	# ~ # The following filter is because there is so much data.
	# ~ data = data[0:99999]
	data = data[data['Delay (s)']<1777e-12]
	
	observed_valid_outputs = {}
	observed_invalid_output = {}
	all_observed_outputs = {}
	for n_TDC in TDC_NUMBERS:
		print(f'Processing data from TDC {n_TDC}...')
		data[f'Output {n_TDC}'] = [TDCOutputDataType(COUNT=row[2+2*(n_TDC-1)], SAFF=row[2+2*(n_TDC-1)+1]) for row in data.itertuples()]
		print(f'Creating the set of output sequences for TDC {n_TDC}...')
		all_observed_outputs[f'TDC {n_TDC}'] = set(data[f'Output {n_TDC}'])
		print(f'Sorting the set of valid output sequences for TDC {n_TDC}...')
		observed_valid_outputs[f'TDC {n_TDC}'] = sorted([s for s in all_observed_outputs[f'TDC {n_TDC}'] if not s.isnan])
		print(f'Selecting the non-valid output sequences for TDC {n_TDC}...')
		observed_invalid_output[f'TDC {n_TDC}'] = [s for s in all_observed_outputs[f'TDC {n_TDC}'] if s.isnan]
	
	# Outputs time distribution plot ---
	for n_TDC in TDC_NUMBERS:
		print(f'Doing distribution plot for TDC {n_TDC}...')
		fig = mpl.manager.new(
			title = f'Distribution of outputs for TDC {n_TDC}',
			show_title = False,
			# ~ subtitle = f'Dataset: {bureaucrat.measurement_name}',
			xlabel = f'Δt = t<sub>START</sub> - t<sub>STOP</sub> (s)',
			ylabel = f'Number of events',
		)
		bins = np.arange(start = min(data['Delay (s)']), stop = max(data['Delay (s)']), step = 2e-12)
		fig.hist(
			data['Delay (s)'],
			label = 'All events without filtering',
			bins = bins,
			color = (0,0,0),
		)
		outputs_to_plot = [output for output in observed_valid_outputs[f'TDC {n_TDC}'] + observed_invalid_output[f'TDC {n_TDC}'] if output.COUNT < 11]
		for idx, output in enumerate(outputs_to_plot):
			fig.hist(
				data.loc[data[f'Output {n_TDC}']==output,'Delay (s)'],
				label = str(output),
				bins = bins,
				# ~ color = (
					# ~ ((-1)**idx+1)/2*idx/len(outputs_to_plot), 
					# ~ 0, 
					# ~ ((-1)**(idx+1)+1)/2*idx/len(outputs_to_plot)
				# ~ ),
				color = tuple(np.random.random(3))
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
