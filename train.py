import numpy as np 
import tensorflow as tf
import os
import shutil

import argparse

from models import *
from data_loader import CelebA

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--data", type=str, required=True)
args = parser.parse_args()

n_filters = 64
embedding_size = 128
image_height = 64
image_width = 64
n_channels = 3

lr = 1e-4
gamma = 0.5
lambda_ = 1e-3
k_val = 0

n_iter = 500
batch_size = 256

data_path = args.data

celeba = CelebA(data_path)

x_real = tf.placeholder(tf.float32, [batch_size, image_height, image_width, n_channels])
z = tf.placeholder(tf.float32, [batch_size, embedding_size])
k = tf.placeholder(tf.float32)

generator, g_vars = Build_Generator(z, n_filters)

discriminator, d_z, d_vars = Build_Discriminator(x_real, generator, embedding_size, n_filters, image_height)

discriminator_real, discriminator_fake = tf.split(discriminator, 2)

d_real_loss = tf.reduce_mean(tf.abs(x_real - discriminator_real))
d_fake_loss = tf.reduce_mean(tf.abs(generator - discriminator_fake))

d_loss = d_real_loss - k * d_fake_loss
g_loss = d_fake_loss

m_global = d_real_loss + tf.abs(gamma * d_real_loss - d_fake_loss)

optimizer = tf.train.AdamOptimizer(lr)

d_optimizer = optimizer.minimize(d_loss, var_list=d_vars)
g_optimizer = optimizer.minimize(g_loss, var_list=g_vars)

init = tf.global_variables_initializer()

print "Training ..."

with tf.Session() as sess:
	
	sess.run(init)

	for i in range(n_iter):
		z_value = np.random.uniform(-1, 1, (batch_size, embedding_size))

		batch_x = celeba.next_batch(batch_size)

		_, _, d_real_loss, d_fake_loss, m_global = sess.run([d_optimizer, g_optimizer, d_real_loss, d_fake_loss, m_global],
			feed_dict={x_real : batch_x, z : z_value, k : k_val})

		k_val = np.clip(k_val + lambda_ * ((gamma * d_real_loss) - d_fake_loss),0,1)

		if i%100 == 0:
			print "epoch %d : Real Loss %lf, Fake Loss %lf, m_global %lf" %(i, d_real_loss, d_fake_loss, m_global)




