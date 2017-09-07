from skimage.io import imsave
import numpy as np

def show_result(batch_res, fname, img_shape=(64,64), grid_size=(8,8), grid_pad=5):
	grid_height = img_shape[0] * grid_size[0] + grid_pad*(grid_size[0]-1)
	grid_width = img_shape[1] * grid_size[1] + grid_pad*(grid_size[1]-1)
	img_grid = np.zeros((grid_height, grid_width, 3), dtype = np.uint8)
	for i, res in enumerate(batch_res):
		if i >= grid_size[0]*grid_size[1]:
			break
		img = res * 255
		img = img.astype(np.uint8)
		grid_row = (i // grid_size[0]) * (img_shape[0] + grid_pad)
		grid_col = (i % grid_size[1]) * (img_shape[1] + grid_pad)
		img_grid[grid_row:grid_row+img_shape[0], grid_col:grid_col+img_shape[1],:] = img
	imsave(fname, img_grid)
