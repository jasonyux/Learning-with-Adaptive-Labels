import sys
import pathlib
root = str(pathlib.Path(__file__).parent.parent.resolve())
sys.path.insert(0, root)

import numpy as np
import json
import tensorflow as tf
import matplotlib.pyplot as plt

from utils.util import sample_data, set_seeds, set_global_determinism, grey_to_rgb, pairwise_dist
from utils.initializers import INIT_LABELS_FUNC
from utils.constants import *
from nltk.corpus import wordnet as wn
from matplotlib import cm
from sklearn.manifold import MDS
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from scipy.spatial import procrustes
from scipy import stats
from algorithms.encoders import *
from algorithms.base import ClassificationModel
from algorithms.labelembed import LabelEmbed_Model
from algorithms.lwr import LWR_Model
from algorithms.lwal import LwAL_Model


CONFIG = {
	"dataset": {
		"name": "cifar10", "num_labels": 10, 'text_labels': list(CIFAR10_LABELS.values()),
		# "name": "awa2_n_precomputed", "num_labels": 23, 'text_labels': list(AWA2_MAPPABLE_TRAIN_LABELS.values()) + list(AWA2_MAPPABLE_TEST_LABELS.values()),
		# "name": "fashion_mnist", "num_labels": 8, 'text_labels': list(FMNIST_MAPPABLE_LABELS.values()},
	},
	"model": {
		"encoder": "resnet50",
		"algo": 'lwal', #["std", "lwr", "label_embed", "lwal"]
		"rloss": "cos_repel_loss_z", # ["cos_repel_loss_z", "none"]
		"latent_dim": 100, # num_labels for normal algorithms, 10 * num_labels for LwAL10, 768 for StaticLabel
		"stationary_steps": 1, # used by LwAL
		"warmup_steps": 0, # [0, 2, 5] for LwAL
		"k": 5, # [2, 3, 5] for LWR
	},
	"training": {
		"lr": 1e-4, # [1e-4, 1e-3]. 1e-3 used only for efficientnet
		"epochs": 10,
	},
	"seed": 123 # [12, 123, 1234]
}

def get_wordnet_rank(label_dist_matrix, leftout_class_idx):
	wordnet_dist = label_dist_matrix[leftout_class_idx].tolist()
	wordnet_rank = wordnet_dist[:leftout_class_idx] + wordnet_dist[leftout_class_idx+1:]
	return wordnet_rank # smaller the better


def get_kendall_tau_score(true_rank, pred_rank):
	tau, p_value = stats.kendalltau(true_rank, pred_rank)
	return tau, p_value


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


def hardcoded_visualize(distance_matrix, class_text_labels):
	dset_name = CONFIG["dataset"]["name"]
	# assumes the distance_matrix[0][1] corresponds to distance between class_text_labels[0] and class_text_labels[1]

	num_labels = len(class_text_labels)
	colors = cm.rainbow(np.linspace(0, 1, num_labels))
	
	fig, ax = plt.subplots(figsize=(6, 6))
	
	embedding = MDS(n_components=2, dissimilarity='precomputed')
	X_transformed = embedding.fit_transform(distance_matrix)

	for i, (xi, yi) in enumerate(X_transformed):
		ax.scatter(xi, yi, color=colors[i], label=f"{class_text_labels[i]}")

	ax.axis('equal')
	if dset_name == "awa2_n_precomputed":
		ax.legend(ncol=1, fancybox=True, shadow=False, bbox_to_anchor=(1.0, 1.05))
	else: # this should work most of the time
		ax.legend(loc='center', ncol=1, fancybox=True, shadow=False, bbox_to_anchor=(1.15, 0.5))
	ax.grid()
	ax.set_xticklabels([])
	ax.xaxis.set_ticks_position('none')
	ax.set_yticklabels([])
	ax.yaxis.set_ticks_position('none')

	# for awa2 we need to have a tight bbox
	fig.savefig("outputs/centroid_mds.png", bbox_inches='tight')
	plt.close(fig)
	return "outputs/centroid_mds.png" # wandb can save this as well


def align_points(reference, target):
	mtx1, mtx2, disparity = procrustes(reference, target)
	return mtx1, mtx2, disparity


def visualize_embedding_matchup(reference_points:np.ndarray, target_points:np.ndarray, text_labels):
	num_labels = len(text_labels)
	fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
	colors = cm.rainbow(np.linspace(0, 1, num_labels))

	std_reference_points, aligned_points, disparity = align_points(reference_points, target_points)

	for i, (xi, yi) in enumerate(std_reference_points):
		ax.scatter(xi, yi, color=colors[i], label=f"{text_labels[i]}")

	for i, (xi, yi) in enumerate(aligned_points):
		ax.scatter(xi, yi, color=colors[i], marker='x')
		# plot the connecting line
		ax.plot([xi, std_reference_points[i, 0]], [yi, std_reference_points[i, 1]], color=colors[i], linestyle='--', alpha=0.1, linewidth=2.5)

	ax.axis('equal')
	if CONFIG["dataset"]["name"] == "awa2_n_precomputed":
		ax.legend(ncol=1, fancybox=True, shadow=False, bbox_to_anchor=(1.0, 1.05))
	else: # this should work most of the time
		ax.legend(loc='center', ncol=1, fancybox=True, shadow=False, bbox_to_anchor=(1.15, 0.5))
	ax.set_xticklabels([])
	ax.xaxis.set_ticks_position('none')
	ax.set_yticklabels([])
	ax.yaxis.set_ticks_position('none')
	ax.grid()

	fig.savefig("outputs/centroid_matchup.png", bbox_inches='tight')
	plt.close(fig)
	return "outputs/centroid_matchup.png", disparity
	

def compute_baseline_learnt_y(model, train_dataset, y_train, num_classes):
	x_train_encoded = []
	for i in train_dataset:
		*in_x, _ = i
		x = in_x[-1] # so that it works for lwr as well
		z, _ = model(x, training=True)
		x_train_encoded.append(z)
	x_train_encoded = tf.concat(x_train_encoded, axis=0)
	centroids = compute_centroids(x_train_encoded, y_train, num_classes)
	return centroids


def construct_dist_matrix(text_labels:list, wordsense_dict:dict={}):
	dist_matrix = np.zeros((len(text_labels), len(text_labels)))
	for i, label1 in enumerate(text_labels):
		wordsense1 = wordsense_dict.get(label1)
		wordsense1 = '.n.01' if wordsense1 is None else wordsense1
		l1 = wn.synset(f'{label1}{wordsense1}')
		for j, label2 in enumerate(text_labels):
			wordsense2 = wordsense_dict.get(label2)
			wordsense2 = '.n.01' if wordsense2 is None else wordsense2
			l2 = wn.synset(f'{label2}{wordsense2}')
			dist_matrix[i,j] = l1.shortest_path_distance(l2)
	return dist_matrix, text_labels


def get_wordnet_color_map(wordnet_label_dist, text_labels):
	dists = squareform(wordnet_label_dist)
	linkage_matrix = linkage(dists, "average")
	wordnet_d = dendrogram(linkage_matrix, labels=text_labels, leaf_rotation=90, distance_sort='descending')

	wordnet_color_map = {}
	for i, k in enumerate(wordnet_d['ivl']):
		wordnet_color_map[k] = wordnet_d['leaves_color_list'][i]
	plt.clf()
	return wordnet_color_map


def compute_dist_from_similarity_matrix(similarity_matrix):
	# first we make sure this is symmetrical
	similarity_matrix = (similarity_matrix + similarity_matrix.T) / 2
	# then we make sure that max is one, min is zero
	similarity_matrix = (similarity_matrix - np.min(similarity_matrix)) / (np.max(similarity_matrix) - np.min(similarity_matrix))
	# then we perform eigendecomposition and obtain the embedding
	eigenvalues, eigenvectors = np.linalg.eig(similarity_matrix)
	eigenvalues, eigenvectors = np.real(eigenvalues), np.real(eigenvectors)
	eigenvalues = np.diag(eigenvalues)
	embedding = eigenvalues**0.5 @ eigenvectors.T
	embedding = embedding.T
	return pairwise_dist(embedding, embedding).numpy()


if __name__ == "__main__":	
	dset_name = CONFIG['dataset']['name']
	num_labels = CONFIG['dataset']['num_labels']
	precomputed_features = CONFIG["dataset"]["name"] == "awa2_n_precomputed"
	text_labels  = CONFIG['dataset']['text_labels']

	encoder = CONFIG['model']['encoder']
	algo = CONFIG['model']['algo']
	rloss = CONFIG['model']['rloss']
	latent_dim = CONFIG['model']['latent_dim']
	stationary_steps = CONFIG['model']['stationary_steps']
	warmup_steps = CONFIG['model']['warmup_steps']
	lwr_k = CONFIG['model']['k']

	lr = CONFIG["training"]['lr']
	epochs = CONFIG["training"]["epochs"]

	seed = CONFIG['seed']

	if encoder != "efficientnet": # efficientnet does not support determinstic algos up to Aug 2022
		set_global_determinism()
	set_seeds(seed)
	print(f"Using {json.dumps(CONFIG, indent=4)}")

	### Prepare daraloaders
	x_train, y_train, x_test, y_test = sample_data(CONFIG)
	if len(x_train.shape) == 3:
		x_train = x_train[:,:,:,None]
		x_test = x_test[:,:,:,None]
	
	if x_train.shape[-1] == 1:
		x_train = grey_to_rgb(x_train)
		x_test = grey_to_rgb(x_test)
	
	img_shape = x_train[0].shape

	batch_size = num_labels * 50
	if algo == "lwr": # lwr needs data index
		train_dataset = tf.data.Dataset.from_tensor_slices((np.arange(len(x_train)), x_train, y_train))
		# this is needed to make trainng faster for LWR specifically, as it needs to remeber the logits
		train_dataset = train_dataset.shuffle(buffer_size=len(x_train), seed=seed, reshuffle_each_iteration=False)
	else:
		train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
		train_dataset = train_dataset.shuffle(buffer_size=len(x_train), seed=seed, reshuffle_each_iteration=True)
	train_dataset = train_dataset.batch(batch_size, deterministic=True, num_parallel_calls=tf.data.experimental.AUTOTUNE)

	test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
	test_dataset = test_dataset.batch(batch_size, deterministic=True, num_parallel_calls=tf.data.experimental.AUTOTUNE)

	### Prepare model and start training ###
	image_encoder = Resnet50Encoder(img_shape, num_labels)

	if algo == 'lwal':
		init_learnt_y = INIT_LABELS_FUNC["onehot"](num_labels, latent_dim, None)
		model = LwAL_Model(
			image_encoder, output_dim=latent_dim, 
			num_classes=num_labels, learnt_y_init=init_learnt_y, 
			stationary_steps=stationary_steps, warmup_steps=warmup_steps, # warm up isn't very useful for competitive test acc
			fixed_label=False,
			rloss=rloss)
		print(f'using {algo} and initialized {init_learnt_y.shape=}')
	elif algo == "lwr":
		assert(latent_dim == num_labels)
		model = LWR_Model(
			image_encoder,
			output_dim=num_labels,
			batch_in_epoch=len(train_dataset),
			total_epochs=epochs,
			k=lwr_k,
			reshuffle_each_iteration=False # if dataloader don't reshuffle per epoch, we can speed up significantly
		)
		print(f'using LWR_Model')
	elif algo == "label_embed":
		assert(latent_dim == num_labels)
		model  = LabelEmbed_Model(
			encoder = image_encoder,
			output_dim = num_labels,
			num_classes = num_labels,
			batch_in_epoch = len(train_dataset),
		)
		print(f'using LabelEmbed_Model')
	else:
		assert(latent_dim == num_labels)
		model = ClassificationModel(image_encoder, num_labels)


	model.compile(
		loss=tf.keras.losses.CategoricalCrossentropy(), # not used by some algos, such as LWR and LwAL
		optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
		metrics=[tf.keras.metrics.Accuracy()],
	)
		
	history = model.fit(
		train_dataset,
		epochs=epochs,
		validation_data=test_dataset,
	)

	### Compare the overall learnt structure with wordnet structure
	if dset_name == "awa2_n_precomputed":
		label_dist_matrix, _ = construct_dist_matrix(text_labels, AWA2_LABELS_TO_WORDSENCE)
	else:
		label_dist_matrix, _ = construct_dist_matrix(text_labels)
	# get the colors that the wordnet reference dendrogram uses
	wordnet_color_map = get_wordnet_color_map(label_dist_matrix, text_labels)

	if algo == "std" or algo == "lwr":
		model.learnt_y = compute_baseline_learnt_y(model, train_dataset, y_train, num_labels)

	if algo == "label_embed":
		all_class_dist = compute_dist_from_similarity_matrix(model.emb.weights[0].numpy())
	else:
		all_class_dist = pairwise_dist(model.learnt_y, model.learnt_y).numpy()

	for i in range(len(all_class_dist)):
		all_class_dist[i][i] = 0.0
	## since there are some numeric instability, let us make sure this is symmetric
	all_class_dist = all_class_dist + all_class_dist.T
	all_class_dist = all_class_dist / 2.0

	overall_kdt_scores = []
	for i in range(num_labels):
		wordnet_rank = get_wordnet_rank(label_dist_matrix, i)
		all_class_dist_i = all_class_dist[i].tolist()
		all_class_dist_i.pop(i)
		tau, p_value = get_kendall_tau_score(wordnet_rank, all_class_dist_i)
		overall_kdt_scores.append((tau, p_value))
	print(overall_kdt_scores)
	
	overall_tau_score = np.mean([tau for (tau, _) in overall_kdt_scores])
	overall_p_score = np.mean([p for (_, p) in overall_kdt_scores])
	print({
		"overall_tau_score": overall_tau_score, 
		"overall_p_score": overall_p_score
	})

	### just in case if needed, plot the embedding and dendrograms as well
	fig = hardcoded_visualize(all_class_dist, text_labels)
	if not isinstance(fig, str):
		plt.close(fig)

	##### Plot Dendrogram #####
	## we use average linkage for all for consistency
	dists = squareform(all_class_dist)
	linkage_matrix = linkage(dists, method="average")
	for i, row in enumerate(linkage_matrix): # so that they are not too small
		linkage_matrix[i][2] += 0.1 * np.max(linkage_matrix[:, 2])
	
	fig, ax = plt.subplots(figsize=(8, 5), dpi=120)
	dendrogram(linkage_matrix, labels=text_labels, leaf_rotation=90, distance_sort='descending', link_color_func=lambda k: 'C0')
	plt.yticks([])
	# change the label color to be the same as the gold standard wornet labels
	xlbls = ax.get_xmajorticklabels()
	for lbl in xlbls:
		lbl.set_color(wordnet_color_map[lbl.get_text()])
		if dset_name == "awa2_n_precomputed":
			lbl.set_size(14)
		else:
			lbl.set_size(18)
	fig.savefig("outputs/dendrogram.png", bbox_inches='tight')

	## perform a embedding match up
	embedding = MDS(n_components=2, dissimilarity='precomputed', random_state=123)
	reference_points = embedding.fit_transform(label_dist_matrix)
	target_points = embedding.fit_transform(all_class_dist)

	fig, disparity = visualize_embedding_matchup(reference_points, target_points, text_labels)
	print({"matched disparity": disparity})
	if not isinstance(fig, str):
		plt.close(fig)