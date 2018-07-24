import numpy as np
def default_psi (x, y, z):
	return np.abs(x) / np.sqrt(np.abs(y * z))

def jacobi_rotation (M, k, l, tol=0.00000000001):
	"""
	input: numpy matrix for rotation M, two different row number k and l 
	output: cos and sin value of rotation 
	change: M is inplace changed
	"""

	# rotation matrix calc
	if M[k, l] + M[l, k] < tol:
		cos_val = 1
		sin_val = 0
	else:
		b = (M[l, l] - M[k, k]) / (M[k, l] + M[l, k])
		tan_val = (1 if b >= 0 else -1) / (abs(b) + np.sqrt(b * b + 1))  # |tan_val| < 1
		cos_val = 1 / (np.sqrt(tan_val * tan_val + 1))  # cos_val > 0
		sin_val = cos_val * tan_val  # |cos_val| > |sin_val|

	# right multiplication by jacobian matrix
	temp1 = M[k, :] * cos_val - M[l, :] * sin_val
	temp2 = M[k, :] * sin_val + M[l, :] * cos_val
	M[k, :] = temp1
	M[l, :] = temp2

	# left multiplication by jacobian matrix transpose
	temp1 = M[:, k] * cos_val - M[:, l] * sin_val
	temp2 = M[:, k] * sin_val + M[:, l] * cos_val
	M[:, k] = temp1
	M[:, l] = temp2

	return cos_val, sin_val


class treelet:
	def __init__ (self, psi=False):
		self.psi = psi if psi else default_psi
		self.n = 0
		self.X = None
		self.phi = None
		self.max_row = None
		self.root = None
		self.dfrk = None
		self._tree = None
		self.transform_list = []
		self.dendrogram_list = []

	# Treelet Tree
	@property
	def tree (self):
		if self._tree is None:
			self._tree = [I[0:2] for I in self.transform_list]
		else:
			return self._tree

	def fit (self, X):
		self.X = np.matrix(X)
		self.phi = lambda x, y: self.psi(self.X[x, y], self.X[x, x], self.X[y, y])
		self.n = self.X.shape[0]
		self.max_row = {i: 0 for i in range(self.n)}
		self._rotate(self.n - 1)
		self.root = list(self.max_row)[0]

	def _rotate (self, multi=False):
		if multi:
			for _ in range(multi):
				self._rotate()
			self.dfrk = [self.transform_list[i][1] for i in range(self.n - 1)]
			self.dfrk.append(self.transform_list[-1][0])
		else:
			(p, q) = self._find()
			(cos_val, sin_val) = jacobi_rotation(self.X, p, q)
			self._record(p, q, cos_val, sin_val)

	def _find (self):
		if self.transform_list:
			k, l, *_ = self.current
			for i in self.max_row:
				if i in (k, l):
					self._max(i)
				if self.phi(self.max_row[i], i) < self.phi(l, i):
					self.max_row[i] = l
				if self.phi(self.max_row[i], i) < self.phi(k, i):
					self.max_row[i] = k
				if self.max_row[i] in (k, l):
					self._max(i)
		else:
			self.max_row_val = {}
			[self._max(i) for i in self.max_row]

		v = list(self.max_row_val.values())
		k = list(self.max_row_val.keys())
		max_v = max(v)
		i = k[v.index(max_v)]
		self.dendrogram_list.append(np.log(max_v))
		return self.max_row[i], i

	def _max (self, col_num):
		temp_max_row = 0
		max_temp = 0
		for i in self.max_row:
			if i == col_num:
				continue
			temp = self.phi(i, col_num)
			if temp >= max_temp:
				temp_max_row = i
				max_temp = temp
		self.max_row[col_num] = temp_max_row
		self.max_row_val[col_num] = max_temp

	def _record (self, l, k, cos_val, sin_val):
		if self.X[l, l] < self.X[k, k]:
			self.current = (k, l, cos_val, sin_val)
		else:
			self.current = (l, k, cos_val, sin_val)
		self.transform_list.append(self.current)
		del self.max_row[self.current[1]]
		del self.max_row_val[self.current[1]]

	def __len__ (self):
		return self.n