import numpy as np 
import tensorflow as tf 

slim = tf.contrib.slim

def int_shape(tensor):
	shape = tensor.get_shape().as_list()
	return [num if num is not None else -1 for num in shape]


def reshape(x, h, w, c):
	return tf.reshape(x, [-1, h, w, c])


def nchw_to_nhwc(x):
	return tf.transpose(x, [0, 2, 3, 1])


def resize_nearest_neighbor(x, new_size):
	return tf.image.resize_nearest_neighbor(x, new_size)


def upscale(x, scale):
	_, h, w, _ = int_shape(x)
	return resize_nearest_neighbor(x, (h*scale, w*scale))


def Build_Generator(z, n_filters, img_size):

	with tf.variable_scope('G') as vs:
		num_output = 8*8*n_filters

		x = slim.fully_connected (z, num_output, activation_fn = None)
		x = reshape(x, 8, 8, n_filters)

		while True:
			x = slim.conv2d(x, n_filters, 3, 1, activation_fn = tf.nn.elu)
			x = slim.conv2d(x, n_filters, 3, 1, activation_fn = tf.nn.elu)

			if int_shape(x)[1] >= img_size:
				break

			x = upscale(x, 2)

		out = slim.conv2d(x, 3, 3, 1, activation_fn = None)

	variables = tf.contrib.framework.get_variables(vs)

	return out, variables
	

def Build_Discriminator(x_real, x_fake, embedding_size, n_filters, img_size, reuse=None):

	x = tf.concat([x_real, x_fake], 0)

	with tf.variable_scope('D') as vs:

		#Encoder
		x = slim.conv2d(x, n_filters, 3, 1, activation_fn = tf.nn.elu)

		i = 1
		while True:
			n_channels = n_filters * i
			x = slim.conv2d(x, n_channels, 3, 1, activation_fn = tf.nn.elu)
			x = slim.conv2d(x, n_channels, 3, 1, activation_fn = tf.nn.elu)

			if int_shape(x)[1] <= 8:
				break

			x = slim.conv2d(x, n_channels, 3, 2, activation_fn = tf.nn.elu)

		x = tf.reshape(x, [-1, 8*8*n_channels])
		z = x = slim.fully_connected(x, embedding_size, activation_fn=None)
		
		#Decoder
		num_output = 8*8*n_filters

		x = slim.fully_connected (z, num_output, activation_fn = None)
		x = reshape(x, 8, 8, n_filters)

		while True:
			x = slim.conv2d(x, n_filters, 3, 1, activation_fn = tf.nn.elu)
			x = slim.conv2d(x, n_filters, 3, 1, activation_fn = tf.nn.elu)

			if int_shape(x)[1] >= img_size:
				break

			x = upscale(x, 2)

		out = slim.conv2d(x, 3, 3, 1, activation_fn = None)

	variables = tf.contrib.framework.get_variables(vs)

	return out, z, variables