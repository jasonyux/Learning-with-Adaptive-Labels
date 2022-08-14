import tensorflow as tf

from algorithms.base import ClassificationModel
from utils.util import pairwise_dist, compute_centroids, update_learnt_centroids


def cos_repel_loss_z(z, in_y, num_labels):
	norm_z = z/tf.norm(z, axis=1, keepdims=True)
	cos_dist = norm_z @ tf.transpose(norm_z)

	# we only penalize vectors that are in differnt classes
	true_y = tf.argmax(in_y, axis=1)
	true_y_ = tf.expand_dims(true_y, axis=1)
	same_class_mask = tf.ones((in_y.shape[0], in_y.shape[0]), dtype=tf.float32)
	for i in range(num_labels):
		true_y_i = tf.cast(true_y_ == i, dtype=tf.float32)
		class_i_mask = 1 - (true_y_i @ tf.transpose(true_y_i)) # 0 if same class, 1 otherwise
		same_class_mask *= class_i_mask
	
	return tf.reduce_mean(cos_dist * same_class_mask)


def ce_pull_loss(enc_x, in_y, learnt_y):
	cce = tf.keras.losses.CategoricalCrossentropy(label_smoothing=1e-6) # label_smoothing
	enc_x_dist = pairwise_dist(enc_x, learnt_y) # + 1e-6 no longer needed since we added 1e-10 when doing pairwise_dist
	logits = tf.nn.softmax(-1. * enc_x_dist, axis=1)  # the closer the better
	loss = cce(in_y, logits)
	return loss


def ce_nn_pred(enc_x, in_y, learnt_y):
	enc_x_dist = pairwise_dist(enc_x, learnt_y)
	logits = tf.nn.softmax(-1. * enc_x_dist, axis=1)  # the closer the better
	preds = tf.argmax(logits, axis=1)
	true_y = tf.math.argmax(in_y, axis=1)
	return preds, true_y


class LwAL_Model(ClassificationModel):
	def __init__(self, encoder, output_dim, num_classes, learnt_y_init, stationary_steps=1, warmup_steps=0, fixed_label=False, rloss="none"):
		super(LwAL_Model, self).__init__(encoder, output_dim)
		self.learnt_y = learnt_y_init
		self.num_classes = num_classes
		self.fixed_label = fixed_label #TODO: if label is learnt or fixed
		self.current_step = 0
		self.stationary_steps = stationary_steps # number of steps to wait before updating learnt_y. 1 seems to work the best
		self.warmup_steps = warmup_steps # number of steps to warm up the model, idea from DEC's autoencoder init
		self.rloss = rloss # either "none" or a callable

	def call(self, x, training = False):
		z = self.encoder(x, training=training)
		z = self.classification_head(z)
		return z, None

	def predict(self, x, training = False):
		z, _ = self(x, training=training)
		z_dist = pairwise_dist(z, self.learnt_y)
		logits = tf.nn.softmax(-1. * z_dist, axis=1)  # the closer the better
		preds = tf.argmax(logits, axis=1)
		return preds

	def train_step(self, data):
		# from models.base import compute_centroid, update_learnt_centroids
		in_x, in_y = data
		num_labels = self.num_classes

		with tf.GradientTape() as tape:
			z, _ = self(in_x, training=True)

			structure_loss = 0.0
			if not self.fixed_label:
				warming_up = self.current_step < self.warmup_steps
				__current_step = self.current_step - self.warmup_steps
				if not warming_up and (__current_step % self.stationary_steps == 0): # will contain first update
					centroids = compute_centroids(z, in_y, self.num_classes)
					self.learnt_y = update_learnt_centroids(self.learnt_y, centroids)
					
					if self.rloss == "cos_repel_loss_z":
						structure_loss = cos_repel_loss_z(z, in_y, num_labels)
				
				self.current_step += 1
			
			# promote centroids to be further apart
			input_loss = ce_pull_loss(z, in_y, self.learnt_y)
			em_loss = 10.0 * structure_loss + 1.0 * input_loss
		
		predictions, true_y = ce_nn_pred(z, in_y, self.learnt_y)

		gradients = tape.gradient(em_loss, self.trainable_weights)
		self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))

		for metric in self.custom_metrics:
			metric.update_state(true_y, predictions)
		
		return {
			"em_loss": em_loss, 
			"structure_loss": structure_loss,
			"input_loss": input_loss,
			**{m.name: m.result() for m in self.custom_metrics}
		}

	def test_step(self, data):
		in_x, in_y = data
		z, _ = self(in_x, training=True)

		input_loss = ce_pull_loss(z, in_y, self.learnt_y)
		structure_loss = cos_repel_loss_z(z, in_y, self.num_classes)
		em_loss = 10.0 * structure_loss + 1.0 * input_loss
		
		predictions, true_y = ce_nn_pred(z, in_y, self.learnt_y)
		for metric in self.custom_metrics:
			metric.update_state(true_y, predictions)
		
		return {
			"em_loss": em_loss, 
			"structure_loss": structure_loss,
			"input_loss": input_loss,
			**{m.name: m.result() for m in self.custom_metrics}
		}