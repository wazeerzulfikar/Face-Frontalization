from skimage.io import imsave

def show_result(batch_res, fname, grid_size=(8,8), grid_pad=5):
	batch_res = 0.5*(batch_res.reshape((batch_res.shape[0], img_height, img_width)))+0.5
	grid_height = img_height * grid_size[0] + grid_pad*(grid_size[0]-1)
	grid_width = img_width * grid_size[1] + grid_pad*(grid_size[1]-1)
	img_grid = np.zeros((grid_height, grid_width), dtype = np.uint8)
	for i, res in enumerate(batch_res):
		if i >= grid_size[0]*grid_size[1]:
			break
		img = res * 255
		img = img.astype(np.uint8)
		grid_row = (i // grid_size[0]) * (img_height + grid_pad)
		grid_col = (i % grid_size[1]) * (img_width + grid_pad)
		img_grid[grid_row:grid_row+img_height, grid_col:grid_col+img_width] = img
	imsave(fname, img_grid)
