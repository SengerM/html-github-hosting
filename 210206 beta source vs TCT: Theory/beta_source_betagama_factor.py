import numpy as np
import myplotlib as mpl
import scipy.constants as constants

m_e_in_eV = constants.electron_mass*constants.c**2/constants.e # https://physics.nist.gov/cgi-bin/cuu/Convert?exp=0&num=1&From=kg&To=ev&Action=Convert+value+and+show+factor
E = np.linspace(m_e_in_eV, 3.1e6, 99) # MeV

gamma = E/m_e_in_eV 
beta = (1-1/gamma**2)**.5

fig = mpl.manager.new(
	title = 'Relativistic factors for electrons emitted by a Sr-90 source',
	xlabel = 'Total energy - electron mass (eV)',
)
# ~ fig.plot(
	# ~ E,
	# ~ gamma,
	# ~ label = 'γ',
# ~ )
fig.plot(
	E - m_e_in_eV,
	beta,
	label = 'β',
)
fig.plot(
	E - m_e_in_eV,
	gamma*beta,
	label = 'βγ',
)
mpl.manager.save_all()
