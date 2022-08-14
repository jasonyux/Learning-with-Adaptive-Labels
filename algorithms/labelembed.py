import tensorflow as tf

from algorithms.base import ClassificationModel


class LabelEmbed_Model(ClassificationModel):
	def __init__(self, encoder, output_dim, num_classes, batch_in_epoch=-1):
		super(LabelEmbed_Model, self).__init__(encoder, output_dim)
		self.num_classes = num_classes
		self.batch_in_epoch = batch_in_epoch # not really used
		self.linear2 = tf.keras.layers.Dense(num_classes)
		self.emb = tf.keras.layers.Embedding(num_classes, num_classes, embeddings_initializer='identity')

		self.step_num = 0

	def my_loss(self, logit, prob):
		""" Cross-entropy function"""
		soft_logit = tf.nn.log_softmax(logit)
		loss = tf.reduce_sum(prob*soft_logit, 1)
		return loss

	def comp_Loss(self, epoch, out1, out2, tar, emb_w, targets, mode):
		"""Compute the total loss"""
		ce_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

		out2_prob = tf.nn.softmax(out2)
		tau2_prob = tf.nn.softmax(out2 / 2)
		tau2_prob = tf.stop_gradient(tau2_prob)
		soft_tar = tf.nn.softmax(tar)
		soft_tar = tf.stop_gradient(soft_tar)
		L_o1_y = ce_loss(targets, out1)
		if mode == 'baseline':
			return L_o1_y

		alpha = 0.9
		beta = 0.5
		pred = tf.reduce_max(out2, 1)
		tmp_target = tf.cast(tf.argmax(targets, axis=1), tf.float32)
		mask = tf.cast(tf.math.round(pred) == tmp_target, tf.float32)
		mask = tf.stop_gradient(mask)
		L_o1_emb = -tf.reduce_mean(self.my_loss(out1, soft_tar))

		L_o2_y = ce_loss(targets, out2)
		L_emb_o2 = -tf.reduce_sum(self.my_loss(tar, tau2_prob)*mask)/(tf.reduce_sum(mask)+1e-8)
		gap = tf.gather(out2_prob, 1, tf.reshape(tmp_target, (-1,1))) - alpha
		L_re = tf.reduce_sum(tf.nn.relu(gap))
		
		loss = beta*L_o1_y + (1-beta)*L_o1_emb +L_o2_y +L_emb_o2 +L_re
		return loss


	def call(self, data, training = False):
		x, targets = data
		z_prev = self.encoder(x, training=training)
		out1 = self.classification_head(z_prev)
		out2 = self.linear2(tf.stop_gradient(z_prev))
		tmp_targets = tf.cast(tf.argmax(targets, axis=1), tf.float32)
		tar = self.emb(tmp_targets)
		return out1, out2, tar, None


	def train_step(self, data):
		# from models.base import compute_centroid, update_learnt_centroids
		in_x, in_y = data
		current_epoch = 1 + self.step_num // self.batch_in_epoch

		with tf.GradientTape() as tape:
			_, targets = in_x, in_y
			outputs = self(data, training=True)
			out1, out2, tar, emb_w = outputs

			loss = self.comp_Loss(current_epoch, out1, out2, tar, emb_w, targets, "labelemb")

		gradients = tape.gradient(loss, self.trainable_weights)
		self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))

		# adjust learning rate is removed for consistency with other algorithms

		true_y = tf.math.argmax(in_y, axis=1)
		predictions = tf.math.argmax(out1, axis=1)
		for metric in self.custom_metrics:
			metric.update_state(true_y, predictions)

		self.step_num += 1
		
		return {
			"labelemb_loss": loss,
			**{m.name: m.result() for m in self.custom_metrics}
		}

	def test_step(self, data):
		in_x, in_y = data

		_, targets = in_x, in_y
		out, _, _,_ = self(data, training=True)
		ce_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
		loss = ce_loss(targets, out)		

		true_y = tf.math.argmax(in_y, axis=1)
		predictions = tf.math.argmax(out, axis=1)

		for metric in self.custom_metrics:
			metric.update_state(true_y, predictions)
		
		return {
			"labelemb_loss": loss,
			**{m.name: m.result() for m in self.custom_metrics}
		}