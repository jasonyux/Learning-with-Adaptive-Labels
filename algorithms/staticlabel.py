import tensorflow as tf

from algorithms.base import ClassificationModel
from utils.util import pairwise_dist


class StaticLabel_Model(ClassificationModel):
	def __init__(self, encoder, output_dim, num_classes, learnt_y_init):
		super(StaticLabel_Model, self).__init__(encoder, output_dim)
		self.label_embeds = learnt_y_init
		self.num_classes = num_classes
		self.loss_fn = tf.keras.losses.Huber(delta=1.0)

	def call(self, x, training = False):
		prev_z = self.encoder(x, training=training)
		z = self.classification_head(prev_z)
		return z, prev_z

	def predict(self, x, training = False):
		z, _ = self(x, training=training)
		z_dist = pairwise_dist(z, self.learnt_y)
		logits = tf.nn.softmax(-1. * z_dist, axis=1)  # the closer the better
		preds = tf.argmax(logits, axis=1)
		return preds

	def huber_loss(self, z, in_y):
		class_list = tf.argmax(in_y, axis=1)
		ground_truth = tf.gather(self.label_embeds, class_list, axis=0)
		loss = self.loss_fn(ground_truth, z)
		return loss

	def train_step(self, data):
		# from models.base import compute_centroid, update_learnt_centroids
		in_x, in_y = data

		with tf.GradientTape() as tape:
			z, _ = self(in_x, training=True)

			# promote centroids to be further apart
			huber_loss = self.huber_loss(z, in_y)

		gradients = tape.gradient(huber_loss, self.trainable_weights)
		self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))

		true_y = tf.math.argmax(in_y, axis=1)
		predictions = pairwise_dist(z, self.label_embeds)
		predictions = tf.argmin(predictions, axis=1)
		for metric in self.custom_metrics:
			metric.update_state(true_y, predictions)
		
		return {
			"huber_loss": huber_loss,
			**{m.name: m.result() for m in self.custom_metrics}
		}

	def test_step(self, data):
		in_x, in_y = data

		z, _ = self(in_x, training=True)
		huber_loss = self.huber_loss(z, in_y)

		true_y = tf.math.argmax(in_y, axis=1)
		predictions = pairwise_dist(z, self.label_embeds)
		predictions = tf.argmin(predictions, axis=1)
		for metric in self.custom_metrics:
			metric.update_state(true_y, predictions)
		
		return {
			"huber_loss": huber_loss,
			**{m.name: m.result() for m in self.custom_metrics}
		}