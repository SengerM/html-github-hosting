import pandas
import grafica

DEVICE = '#11'

CAEN_MEASUREMENTS_PATHS = {
	'#5': '/home/alf/cernbox/measurements_data/LGAD/EPR2021_LGAD_long_term_test/daemon_test/log/20210803155653_log/IV_curves/20210803180352_#5/measure_iv_with_CAEN/measured_data.csv',
	'#10': '/home/alf/cernbox/measurements_data/LGAD/EPR2021_LGAD_long_term_test/daemon_test/log/20210803155653_log/IV_curves/20210803171348_#10/measure_iv_with_CAEN/measured_data.csv',
	'#11': '/home/alf/cernbox/measurements_data/LGAD/EPR2021_LGAD_long_term_test/daemon_test/log/20210803155653_log/IV_curves/20210803181430_#11/measure_iv_with_CAEN/measured_data.csv',
}
PROBE_STATION_MEASUREMENTS_PATHS = {
	'#5': '/home/alf/cernbox/measurements_data/LGAD/EPR2021_LGAD_long_term_test/FBK_UFSD3.2_IV_3Terminals_ProbeStation/measured_data/Run88/data@1[88].xls',
	'#10': '/home/alf/cernbox/measurements_data/LGAD/EPR2021_LGAD_long_term_test/FBK_UFSD3.2_IV_3Terminals_ProbeStation/measured_data/Run66/data@1[66].xls',
	'#11': '/home/alf/cernbox/measurements_data/LGAD/EPR2021_LGAD_long_term_test/FBK_UFSD3.2_IV_3Terminals_ProbeStation/measured_data/Run94/data@1[94].xls',
}

caen_iv_df = pandas.read_csv(CAEN_MEASUREMENTS_PATHS[DEVICE])
probe_station_df = pandas.read_excel(PROBE_STATION_MEASUREMENTS_PATHS[DEVICE])
probe_station_df['Current (A)'] = (probe_station_df['AI']**2)**.5
probe_station_df['Voltage (V)'] = probe_station_df['AV']

fig = grafica.manager.new(
	title = f'Device {DEVICE} IV',
	subtitle = f'Preliminary data from setup commissioning',
	xlabel = 'Bias voltage (V)',
	ylabel = 'Current (A)',
	yscale = 'log',
)
fig.scatter(
	x = caen_iv_df['Voltage (V)'],
	y = caen_iv_df['Current (A)'],
	label = 'CAEN',
)
values_to_plot_probe_station = probe_station_df['Voltage (V)']<float('inf')#(probe_station_df['Voltage (V)']>min(caen_iv_df['Voltage (V)']))&(probe_station_df['Voltage (V)']<max(caen_iv_df['Voltage (V)']))
fig.scatter(
	x = probe_station_df.loc[values_to_plot_probe_station, 'Voltage (V)'],
	y = probe_station_df.loc[values_to_plot_probe_station, 'Current (A)'],
	label = 'Probe station',
)
grafica.manager.save_unsaved()
