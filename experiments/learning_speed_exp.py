import sys
import pathlib
root = str(pathlib.Path(__file__).parent.parent.resolve())
sys.path.insert(0, root)

import numpy as np
import json
import tensorflow as tf

from utils.util import sample_data, set_seeds, set_global_determinism, grey_to_rgb
from utils.initializers import INIT_LABELS_FUNC
from algorithms.encoders import *
from algorithms.base import ClassificationModel
from algorithms.labelembed import LabelEmbed_Model
from algorithms.lwr import LWR_Model
from algorithms.lwal import LwAL_Model
from algorithms.staticlabel import StaticLabel_Model


CONFIG = {
	"dataset": {
		"name": "cifar10", # ["mnist", "fashion_mnist", "cifar10", "cifar100", "food101"]
		"num_labels": 10, # [10, 10, 10, 100, 101]	
	},
	"model": {
		"encoder": "densenet", #["resnet50", "efficientnet", "densenet"]
		"algo": 'lwal', #["std", "staticlabel", "lwr", "label_embed", "lwal"]
		"rloss": "cos_repel_loss_z", # ["cos_repel_loss_z", "none"]
		"latent_dim": 100, # num_labels for normal algorithms, 10 * num_labels for LwAL10, 768 for StaticLabel
		"stationary_steps": 1, # used by LwAL
		"warmup_steps": 0, # [0, 2, 5] for LwAL
		"k": 5, # [2, 3, 5] for LWR
	},
	"training": {
		"lr": 1e-4, # [1e-4, 1e-3]. 1e-3 used only for efficientnet
		"epochs": 10, # [10, 20]. 20 used for large datasets (CIFAR100, Food101)
	},
	"seed": 123 # [12, 123, 1234]
}


if __name__ == "__main__":	
	dset_name = CONFIG['dataset']['name']
	num_labels = CONFIG['dataset']['num_labels']
	precomputed_features = CONFIG["dataset"]["name"] == "awa2_n_precomputed"

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
	if encoder == 'resnet50':
		image_encoder = Resnet50Encoder(img_shape, num_labels)
	elif encoder == "densenet":
		image_encoder = DenseNet121Encoder(img_shape, num_labels)
	elif encoder == "efficientnet":
		image_encoder = EfficientNetEncoder(img_shape, num_labels)

	if algo == 'lwal':
		init_learnt_y = INIT_LABELS_FUNC["onehot"](num_labels, latent_dim, None)
		model = LwAL_Model(
			image_encoder, output_dim=latent_dim, 
			num_classes=num_labels, learnt_y_init=init_learnt_y, 
			stationary_steps=stationary_steps, warmup_steps=warmup_steps, # warm up isn't very useful for competitive test acc
			fixed_label=False,
			rloss=rloss)
		print(f'using {algo} and initialized {init_learnt_y.shape=}')
	elif algo == "staticlabel":
		assert(latent_dim == 768) # since we are using pretrained BERT
		init_learnt_y = INIT_LABELS_FUNC["bert"](num_labels, latent_dim, f"{dset_name}_bert")
		model = StaticLabel_Model(
			image_encoder,
			output_dim=latent_dim,
			num_classes=num_labels,
			learnt_y_init=init_learnt_y,
		)
		print(f'using StaticLabel_Model and initialized {init_learnt_y.shape=}, {latent_dim=}')
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

	# done
	print(f"""
		avg_train_acc = {np.mean(history.history['accuracy'])}
		best_train_acc = {np.max(history.history['accuracy'])}
		avg_test_acc = {np.mean(history.history['val_accuracy'])}
		best_test_acc = {np.max(history.history['val_accuracy'])}
	""")