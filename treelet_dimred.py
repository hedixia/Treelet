import numpy as np

from treelet import treelet


class treelet_dimred:
	def __init__ (self, t=0):
		self.t = t
		self.n = 0

	def fit (self, dataset_ref):
		self.dataset_ref = np.matrix(dataset_ref)
		self.avedat = np.average(self.dataset_ref, axis=0)
		self.cov = np.cov(self.dataset_ref.getT())
		psi = lambda x, y, z:abs(x) / np.sqrt(np.abs(y * z)) + abs(x) * self.t
		self.trl = treelet(self.cov, psi)
		self.trl.fullrotate()
		self.n = self.trl.n
		self.transform_list = self.trl.transform_list
		self.dfrk = self.trl.dfrk

	# Treelet Transform
	def transform (self, v, k=False, epsilon=0):
		v = np.matrix(v) - self.avedat
		k = k if k else 1
		for iter in range(self.n - k):
			(scv, cgs, cos_val, sin_val) = self.transform_list[iter]
			temp_scv = cos_val * v[:, scv] - sin_val * v[:, cgs]
			temp_cgs = sin_val * v[:, scv] + cos_val * v[:, cgs]
			v[:, scv] = temp_scv
			v[:, cgs] = temp_cgs
		if epsilon == 0:
			return (v, None)
		else:
			scaling_part = np.concatenate([v[:, self.dfrk[i]] for i in range(k, cols)], axis=1)
			difference_part = np.concatenate([v[:, self.dfrk[i]] for i in range(k)], axis=1)
			difference_mat = coo_matrix(abs(difference_part) > epsilon).multiply(difference_part)
			return (scaling_part, difference_mat)

	def inverse_transform (self, scaling_part, difference_mat=False):
		scaling_part = np.matrix(scaling_part)
		rows = scaling_part.shape[0]
		cols = self.n
		k = cols - scaling_part.shape[1]
		v = np.matrix(np.zeros((rows, cols)))
		for i in range(k, cols):
			v[:, self.dfrk[i]] = scaling_part[:, i]
		for iter in reversed(self.transform_list):
			(scv, cgs, cos_val, sin_val) = iter
			temp_scv = cos_val * v[:, scv] + sin_val * v[:, cgs]
			temp_cgs = -sin_val * v[:, scv] + cos_val * v[:, cgs]
			v[:, scv] = temp_scv
			v[:, cgs] = temp_cgs
		if difference_mat:
			for i in range(k):
				v[:, self.dfrk[i]] += difference_mat[:, i]
		return v + self.avedat

	def cluster (self, k):
		returnL = list(range(self.n))
		for i in range(self.n - k, -1, -1):
			returnL[self.transform_list[i][1]] = returnL[self.transform_list[i][0]]
		return returnL

	@property
	def mean_ (self):
		return self.avedat

	def __len__ (self):
		return self.n

	__call__ = transform

	def components_ (self, k):
		return self.transform(np.identity(self.n) + self.avedat, k=k)[0]
