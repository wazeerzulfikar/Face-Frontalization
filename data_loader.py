import numpy as np 
from scipy.io import loadmat

class CelebA(object):

	def __init__(self, path):
		self._load(path)

	def _load(self):
		print "Obtaining File..."
		x = loadmat(path)["images"]
		x = x.astype(np.float32)
		x = x / 255
		x -= 0.5
		print "File Obtained"
		self.x = x

	def norm(self, x):
		return x - 0.5

	def denorm(self, x):
		return np.clip(x + 0.5, 0, 1)

	def next_batch(self, batch_size):
		idx = np.random.choice(len(self.x), batch_size, replace=False)
		return self.x[idx]



