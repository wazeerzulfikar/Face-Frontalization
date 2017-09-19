import numpy as np 
import os
from scipy import misc

class CelebA(object):

	def __init__(self, path, img_size):
		self.img_size = img_size
		self._load(path)

	def _load(self, path):
		print ("Obtaining File...")
		x = self.load_contents(path, 1000)
		x = x.astype('float32')
		print ("File Obtained")
		self.x = x

	def load_contents(self, folder, batch_size):
		imgs = list()
		i = 0
		for filename in os.listdir(folder):
			if i == batch_size:
				break
			if filename.endswith('jpg'):
				img = misc.imread(os.path.join(folder,filename))
				img = misc.imresize(img, (self.img_size, self.img_size))
				imgs.append(img)
				i = i+1

		return np.array(imgs)
			

	def norm(self, x):
		return x/255

	def denorm(self, x):
		return np.clip((x+1)*127.5, 0, 255)

	def next_batch(self, batch_size):
		idx = np.random.choice(len(self.x), batch_size, replace=False) 
		return self.x[idx]


