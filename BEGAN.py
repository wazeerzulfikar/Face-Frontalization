import numpy as np 
import tensorflow as tf 

slim = tf.contrib.slim

def GeneratorCNN (z, hidden_num, output_num, repeat_num):

	with tf.variable_scope('G') as vs:
		num_output = int(np.prod([8,8,hidden_num]))

		x = slim.fully_connected (z, num_output, activation_fn = None)
		x = reshape(x, 8, 8, hidden_num)

		for idx in range(repeat_num):
			x = slim.conv2d(x, hidden_num, 3, 1, activation_fn = tf.nn.elu)
			x = slim.conv2d(x, hidden_num, 3, 1, activation_fn = tf.nn.elu)

			if idx < repeat_num - 1:
				x = upscale(x, 2)

		out = slim.conv2d(x, 3, 3, 1, activation_fn = None)

	variables = tf.contrib.framework.get_variables(vs)

	return out, variables

def DiscriminatorCNN (x, input_channel, z_num, hidden_num, repeat_num):
	with tf.variable_scope('D') as vs:
		#Encoder
		x = slim.conv2d(x, hidden_num, 3, 1, activation_fn = tf.nn.elu)

		for idx in range(repeat_num):
			channel_num = hidden_num * (idx+1)
			x = slim.conv2d(x, channel_num, 3, 1, activation_fn = tf.nn.elu)
			x = slim.conv2d(x, channel_num, 3, 1, activation_fn = tf.nn.elu)

			if idx < repeat_num - 1:
				x = slim.conv2d(x, channel_num, 3, 2, activation_fn = tf.nn.elu)

		x = tf.reshape(x, [-1, np.prod([8, 8, channel_num])])
		z = x = slim.fully_connected(x, z_num, 3, 1)
		#Decoder


		num_output = int(np.prod([8,8,hidden_num]))

		x = slim.fully_connected (z, num_output, activation_fn = None)
		x = reshape(x, 8, 8, hidden_num)

		for idx in range(repeat_num):
			x = slim.conv2d(x, hidden_num, 3, 1, activation_fn = tf.nn.elu)
			x = slim.conv2d(x, hidden_num, 3, 1, activation_fn = tf.nn.elu)

			if idx < repeat_num - 1:
				x = upscale(x, 2)

		out = slim.conv2d(x, input_channel, 3, 1, activation_fn = None)

	variables = tf.contrib.framework.get_variables(vs)

	return out, z, variables
