import numpy as np
from scipy.stats.kde import gaussian_kde
import myplotlib as mpl # https://github.com/SengerM/myplotlib

def generate_samples(x,y):
	return list(np.random.randn(99999)*(1+x)+y+x) + list((np.random.rand(9999)-.5)*3*(1+x)+y+x)

samples = generate_samples(5,3)
pdf = gaussian_kde(samples)

fig = mpl.manager.new(
	title = 'Example of gaussian_kde',
	xlabel = 'Random variable',
	ylabel = 'Probability density',
)
fig.hist(
	samples,
	density = True,
	label = 'Measured samples',
)
q = np.linspace(min(samples),max(samples))
fig.plot(
	q,
	pdf(q),
	label = 'Gaussian KDE approximation',
)
fig.show()
