import tensorflow as tf
import numpy as np
import pylab as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if not os.path.isdir('figures'):
	print('creating the figures folder')
	os.makedirs('figures')

NUM_FEATURES = 8

learning_rates = [0.5e-06, 1e-07, 0.5e-08, 1e-09, 1e-10]
beta = 1e-3
epochs = 500
batch_size = 32
n_folds=5
num_neuron = 30

seed = 10
np.random.seed(seed)
tf.set_random_seed(seed)

np.random.shuffle(cal_housing)
X_data, Y_data = cal_housing[:, :8], cal_housing[:, -1]
Y_data = (np.asmatrix(Y_data)).transpose()
