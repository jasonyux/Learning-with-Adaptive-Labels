import tensorflow as tf
import sklearn
import numpy as np

from pathlib import Path

ROOT = str(Path(__file__).absolute().parent.parent)


INIT_LABELS_FUNC = {
	"onehot": lambda n,k,dset_name: tf.eye(n, num_columns=k),
	"bert": lambda n,k,dset_name: bert_init_labels(n, k, dset_name),
}

def __adjust_label_dim(labels, label_dim):
	"""Pad the label if label_dim is larger, else use PCA to reduce the dim.
	"""
	ori_label_shape = labels.shape[-1]
	if label_dim == ori_label_shape:
		return labels
	
	if ori_label_shape > label_dim: # use PCA
		n_components = min(labels.shape[0], label_dim)
		pca = sklearn.decomposition.PCA(n_components, random_state=0)
		labels = pca.fit_transform(labels)
		ori_label_shape = labels.shape[-1]
	
	if ori_label_shape < label_dim:
		labels = np.pad(labels, ((0, 0), (0, label_dim - ori_label_shape)), mode="constant")
	return labels

def bert_init_labels(n:int, latent_dim:int=None, dset_name:str=None):
	npz_file = np.load(f"{ROOT}/datasets/precomputed_labels/{dset_name}.npz")
	labels = npz_file["labels"][:n]

	# postprocess labels
	labels = __adjust_label_dim(labels, int(latent_dim))
	return labels