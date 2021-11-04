import pandas
import unifiedplottinginterface as upi
import numpy as np

RECTANGLES_COLORS = [
	(0,0,0), 
	(22/255, 171/255, 9/255), 
	(1,0,0),
	(25/255, 255/255, 247/255),
]

mle_reconstruction_data = pandas.read_csv('/home/alf/cernbox/measurements_data/AC-LGAD/20210422175231_ElHolandes_WholeArea_TrainingDataset_11um_999trig_1MIP/MLE_position_reconstruction_analysis_parallel/20210524000000_unidos/reconstruction_data.csv')
dnn_reconstruction_data = pandas.read_feather('/home/alf/cernbox/measurements_data/AC-LGAD/20210425120438_ElHolandes_WholeArea_TestingDataset_1um_9trig_1MIP/dnn_position_reconstruction/20210706132426_reconstruction_DNN-20210706132426_4_2_1_10_ACLGADDNNRegressor/reconstruction_data.fd')

for data in [mle_reconstruction_data, dnn_reconstruction_data]:
	data['reconstruction error (m)'] = ((data['real_x (m)'] - data['reconstructed_x (m)'])**2 + (data['real_y (m)'] - data['reconstructed_y (m)'])**2)**.5

rectangles_corners = pandas.read_csv('/home/alf/cernbox/measurements_data/AC-LGAD/20210425120438_ElHolandes_WholeArea_TestingDataset_1um_9trig_1MIP/dnn_position_reconstruction_analysis_and_plots/20210706132426_reconstruction_DNN-20210706132426_4_2_1_10_ACLGADDNNRegressor/analyses_results.csv')
rectangles_corners.drop(['x_mean (m)','y_mean (m)', 'x_std (m)','y_std (m)'], axis=1, inplace=True)

fig_r = upi.manager.new(
	title = f'r error reconstruction',
	subtitle = 'MLE vs DNN comparison',
	xlabel = 'Reconstruction error (m)',
	ylabel = 'Number of events',
)
fig_x = upi.manager.new(
	title = f'x error reconstruction',
	subtitle = 'MLE vs DNN comparison',
	xlabel = 'x reconstruction error (m)',
	ylabel = 'Number of events',
	yscale = 'log',
)
fig_y = upi.manager.new(
	title = f'y error reconstruction',
	subtitle = 'MLE vs DNN comparison',
	xlabel = 'y reconstruction error (m)',
	ylabel = 'Number of events',
	yscale = 'log',
)

figures = {'r': fig_r, 'x': fig_x, 'y': fig_y}

bins = np.arange(0, 222e-6, 1e-6)
for idx, (region_number,x1,y1,x2,y2) in enumerate(zip(rectangles_corners['region number'],rectangles_corners['x1'],rectangles_corners['y1'],rectangles_corners['x2'],rectangles_corners['y2'])):
	mle_samples_within_this_rectangle = mle_reconstruction_data.loc[(mle_reconstruction_data['real_x (m)']>=x1)&(mle_reconstruction_data['real_x (m)']<=x2)&(mle_reconstruction_data['real_y (m)']>=y1)&(mle_reconstruction_data['real_y (m)']>=y2)]
	dnn_samples_within_this_rectangle = dnn_reconstruction_data.loc[(dnn_reconstruction_data['real_x (m)']>=x1)&(dnn_reconstruction_data['real_x (m)']<=x2)&(dnn_reconstruction_data['real_y (m)']>=y1)&(dnn_reconstruction_data['real_y (m)']>=y2)]
	
	for coord,fig in figures.items():
		if coord == 'r':
			_ = mle_samples_within_this_rectangle['reconstruction error (m)']
		elif coord in {'x','y'}:
			_ = mle_samples_within_this_rectangle[f'real_{coord} (m)'] - mle_samples_within_this_rectangle[f'reconstructed_{coord} (m)']
		fig.histogram(
			_,
			label = f'MLE, region number {region_number}',
			marker = '.',
			color = RECTANGLES_COLORS[idx],
			# ~ bins = bins,
		)
		if coord == 'r':
			_ = dnn_samples_within_this_rectangle['reconstruction error (m)']
		elif coord in {'x','y'}:
			_ = dnn_samples_within_this_rectangle[f'real_{coord} (m)'] - dnn_samples_within_this_rectangle[f'reconstructed_{coord} (m)']
		fig.histogram(
			_,
			label = f'DNN, region number {region_number}',
			color = RECTANGLES_COLORS[idx],
			marker = '*',
			# ~ bins = bins,
		)

upi.manager.show()

