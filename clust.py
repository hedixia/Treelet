import numpy as np
from Dataset import Dataset

class RebuildError (Exception):
	pass

class clust:
	def __init__ (self, dataset_ref=[], slice=False):
		self.dataset_ref = dataset_ref
		self.input_slice(slice)
		self.been_built = False

	def build (self):
		if self.been_built:
			raise RebuildError
		else:
			self.been_built = True
		self.labels = {}
		self.clusters = {}

	def fit(self, X):
		self.dataset_ref = Dataset(X)
		self.input_slice()
		self.build()

	def get (self, return_type="clusters"):
		if return_type in ["C", "clusters"]:
			return self.clusters
		if return_type in ["L", "labels"]:
			return self.labels
		return None

	def show (self, title=False):
		if title:
			print("\n" + title + "\n")
		else:
			print("\nClusters\n")
		for i in self.clusters:
			print(i, self.clusters[i])

	def input_slice (self, slice=False, out=False):
		if slice is False:
			size = len(self.dataset_ref)
			slice = range(size)
		elif type(slice) is int:
			size = slice
			slice = range(slice)
		else:
			size = len(slice)
			slice = slice
		if out:
			return size, slice
		else:
			self.size = size
			self.slice = slice

	@property
	def labels_ (self):
		try:
			return self.sorted_labels
		except AttributeError:
			try:
				self.sorted_labels = [self.labels[i] for i in sorted(self.labels)]
				return self.sorted_labels
			except AttributeError:
				raise AttributeError("labels object does not exist.")

	def __len__ (self):
		return self.size

	def _l2c (self):
		self.clusters = {}
		for i in self.labels:
			self.clusters.setdefault(self.labels[i], []).append(i)

	def _c2l (self):
		self.labels = {}
		for i in self.clusters:
			for j in self.clusters[i]:
				self.labels[j] = i

	def psi (self, x, y, z):
		return np.abs(x) / np.sqrt(np.abs(y * z))

	def assign (self, data):
		#This function may not be necessarily implemented
		raise NotImplementedError
