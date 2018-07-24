import numpy as np


class kernel:
	def __init__ (self, name, parameters):
		self.name = self.identify(name)
		self.parameters = parameters

	def identify (self, name):
		if name in "polynomial":
			name = "poly"
		if name in "radial basis":
			name = "rbk"
		return name

	def __call__ (self, x, y):
		x = np.asarray(x).flatten()
		y = np.asarray(y).flatten()
		if self.name == "poly":
			# $K(x,y) = (x^Ty+c)^\delta$ where $parameters = [c,\delta]$
			c = self.parameters[0]
			delta = self.parameters[1]
			return (np.inner(x, y) + c)**delta
		if self.name == "rbk":
			# $K(x,y) = exp\{-\frac{||y-x||^2}{2\sigma^2}\}$
			invsigmasq2 = 1/(self.parameters[0] * self.parameters[0] * 2)
			return np.exp(-(np.inner(y - x, y - x)) * invsigmasq2)

	def __bool__ (self):
		return True
