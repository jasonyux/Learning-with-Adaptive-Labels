import tensorflow as tf
import numpy as np
import os
import random

from utils.constants import *


def set_seeds(seed):
	os.environ['PYTHONHASHSEED'] = str(seed)
	random.seed(seed)
	tf.random.set_seed(seed)
	np.random.seed(seed)
	return


def set_global_determinism():
	os.environ['TF_DETERMINISTIC_OPS'] = '1'
	os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
	
	tf.config.threading.set_inter_op_parallelism_threads(1)
	tf.config.threading.set_intra_op_parallelism_threads(1)
	return


def grey_to_rgb(batched_img):
	batched_img = np.squeeze(batched_img)
	batched_img = np.stack([batched_img, batched_img, batched_img], axis=-1)
	return batched_img


def compute_centroids(z, in_y, num_classes=10):
	true_y = tf.argmax(in_y, axis=1)
	centroids = []
	for i in range(num_classes): # num classes may not be equal to latend dim
		class_i_mask = tf.cast(tf.expand_dims(true_y, axis=1) == i, tf.float32)
		num_class_i = tf.reduce_sum(class_i_mask)
		if num_class_i == 0:
			centroids.append(tf.zeros([z.shape[1]]))
		else:
			class_i_mask = tf.math.multiply(tf.ones_like(z), class_i_mask)
			masked_z_i = tf.math.multiply(z, class_i_mask)
			centroid_i = tf.reduce_sum(masked_z_i, axis=0) / num_class_i
			centroids.append(centroid_i)
	return tf.stack(centroids)


def update_learnt_centroids(learnt_y, centroids, decay_factor=1.0):
	latent_dim = learnt_y.shape[1] 
	num_classes = learnt_y.shape[0] # this is always correct
	new_learnt_y = []
	for i in range(num_classes):
		enc_y = centroids[i]
		if tf.math.count_nonzero(enc_y) == 0: # check if all zero
			enc_y = learnt_y[i]
		new_enc_y = decay_factor * enc_y + (1 - decay_factor) * learnt_y[i]
		new_learnt_y.append(new_enc_y)
	return tf.stack(new_learnt_y)


def pairwise_dist(A, B):
	na = tf.reduce_sum(tf.square(A), 1)
	nb = tf.reduce_sum(tf.square(B), 1)

	na = tf.reshape(na, [-1, 1])
	nb = tf.reshape(nb, [1, -1])

	D = tf.sqrt(tf.maximum(na - 2 * tf.matmul(A, B, False, True) + nb, 1e-12))
	return D


def sample_data(config):
	###### Load Data ######
	dset_name = config["dataset"]["name"]
	num_classes = config['dataset']['num_labels']

	# sample data
	train_prefix = f"{num_classes}_cls_all_percls"
	train_load_path = f"datasets/{dset_name}/train/{train_prefix}.npz"
	train_dset = np.load(train_load_path)
	x_train = train_dset["x"]
	y_train = train_dset["y"]

	test_prefix = f"{num_classes}_cls_all_percls"
	test_load_path = f"datasets/{dset_name}/test/{test_prefix}.npz"
	test_dset = np.load(test_load_path)
	x_test = test_dset["x"]
	y_test = test_dset["y"]

	return x_train, y_train, x_test, y_test


def to_onehot(y, num_classes):
	y_onehot = np.zeros((len(y), num_classes))
	y_onehot[np.arange(len(y)), y] = 1
	return y_onehot


def __resplit_data(x_train, y_train, x_test, y_test, combined_class_ids):
	x_in_train = x_train[np.isin(np.argmax(y_train, axis=1), combined_class_ids)]
	y_in_train = y_train[np.isin(np.argmax(y_train, axis=1), combined_class_ids)]
	y_in_reformated_train = []
	for new_i in np.argmax(y_in_train, axis=1): # convert to 23 dimensional labels
		y_in_reformated_train.append(combined_class_ids.index(new_i))
	y_in_train = to_onehot(y_in_reformated_train, len(combined_class_ids))

	x_in_test = x_test[np.isin(np.argmax(y_test, axis=1), combined_class_ids)]
	y_in_test = y_test[np.isin(np.argmax(y_test, axis=1), combined_class_ids)]
	y_in_reformated_test = []
	for new_i in np.argmax(y_in_test, axis=1):
		y_in_reformated_test.append(combined_class_ids.index(new_i))
	y_in_test = to_onehot(y_in_reformated_test, len(combined_class_ids))
	return x_in_train, y_in_train, x_in_test, y_in_test


def resplit_data(config, x_train, y_train, x_test, y_test):
	"""used specifically when doing the hierarchy prediction task, as it involves some
	special split of the data and it only includes a few datasets (cifar10, awa2, and 20newsgroup)
	"""
	dset_name = config["dataset"]["name"]
	if dset_name == "cifar10":
		# do nothing because all classes have its mapping in wordnet
		return x_train, y_train, x_test, y_test
	elif dset_name == "awa2_n_precomputed":
		combined_class_ids = list(AWA2_MAPPABLE_TRAIN_LABELS.keys()) + list(AWA2_MAPPABLE_TEST_LABELS.keys())
		x_in_train, y_in_train, x_in_test, y_in_test = __resplit_data(x_train, y_train, x_test, y_test, combined_class_ids)
		return x_in_train, y_in_train, x_in_test, y_in_test
	elif dset_name == "fashion_mnist":
		class_ids = list(FMNIST_MAPPABLE_LABELS.keys())
		x_in_train, y_in_train, x_in_test, y_in_test = __resplit_data(x_train, y_train, x_test, y_test, class_ids)
		return x_in_train, y_in_train, x_in_test, y_in_test
	return # error