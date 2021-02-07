import numpy as np
import myplotlib as mpl
import scipy.constants as constants

detector_thickness = 50e-6
x = np.linspace(0, 333e-6, 99)

fig = mpl.manager.new(
	title = f'Beam profile along the silicon detector',
	xlabel = f'Distance within silicon (m)',
	ylabel = f'Relative number of photons in the beam',
)
fig.plot(
	x,
	np.exp(-x/1e-3)
)
mpl.manager.save_all()
