import tensorflow as tf

from algorithms.base import ClassificationModel


class LWR_Model(ClassificationModel):
	def __init__(self, encoder, output_dim, batch_in_epoch=-1, k=5, total_epochs=10, reshuffle_each_iteration=True):
		super(LWR_Model, self).__init__(encoder, output_dim)
		assert(batch_in_epoch > 0)
		self.batch_in_epoch = batch_in_epoch
		self.k = k
		self.total_epochs = total_epochs
		self.reshuffle_each_iteration = reshuffle_each_iteration
		self.loss_fn = tf.keras.losses.Huber(delta=1.0)

		self.step_num = 0
		self.train_soft_labels = {}

	def call(self, x, training = False):
		z_prev = self.encoder(x, training=training)
		z = self.classification_head(z_prev)
		return z, z_prev

	def predict(self, x, training = False):
		z, _ = self(x, training=training)
		logits = tf.nn.softmax(z, axis=1)
		preds = tf.argmax(logits, axis=1)
		return preds
	
	def loss_cls(self, z, in_y):
		loss_fn = tf.keras.losses.CategoricalCrossentropy()
		y = tf.nn.softmax(z, axis=1)
		return loss_fn(in_y, y)

	def update_soft_labels(self, data_idx, z):
		# if it is reshuffuled, we need to remeber every single idx
		if self.reshuffle_each_iteration:
			for idx, zi in zip(data_idx, z):
				# use 5.0 as temperature
				self.train_soft_labels[idx.numpy()] = tf.nn.softmax(zi / 5.0)
		else:
			first_idx = data_idx[0].numpy()
			self.train_soft_labels[first_idx] = tf.nn.softmax(z / 5.0)
		return

	def loss_kd(self, data_idx, z):
		# first we get the previous softlabels
		if self.reshuffle_each_iteration:
			batched_soft_labels = []
			for idx in data_idx:
				batched_soft_labels.append(self.train_soft_labels[idx.numpy()])
			batched_soft_labels = tf.stack(batched_soft_labels)
		else: # assumes same data within a batch
			first_idx = data_idx[0].numpy()
			batched_soft_labels = self.train_soft_labels[first_idx]
		# then we calculate the loss
		loss_fn = tf.keras.losses.KLDivergence()
		y = tf.nn.softmax(z, axis=1)
		return loss_fn(batched_soft_labels, y)

	def train_step(self, data):
		data_idx, in_x, in_y = data

		with tf.GradientTape() as tape:
			z, _ = self(in_x, training=True)

			# promote centroids to be further apart
			current_epoch = 1 + self.step_num // self.batch_in_epoch
			if current_epoch <= self.k:
				loss = self.loss_cls(z, in_y)
				if current_epoch == self.k:
					self.update_soft_labels(data_idx, z)
			else: # now we should have all soft labels
				beta = (0.9 * current_epoch / self.total_epochs)
				alpha = 1.0 - beta
				loss = alpha * self.loss_cls(z, in_y) +  beta * self.loss_kd(data_idx, z)
				if current_epoch % self.k == 0:
					self.update_soft_labels(data_idx, z)

		gradients = tape.gradient(loss, self.trainable_weights)
		self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))

		true_y = tf.math.argmax(in_y, axis=1)
		predictions = tf.math.argmax(z, axis=1)
		for metric in self.custom_metrics:
			metric.update_state(true_y, predictions)

		self.step_num += 1
		
		return {
			"lwr_loss": loss,
			**{m.name: m.result() for m in self.custom_metrics}
		}

	def test_step(self, data):
		in_x, in_y = data

		z, _ = self(in_x, training=True)
		loss = self.loss_cls(z, in_y)

		true_y = tf.math.argmax(in_y, axis=1)
		predictions = tf.math.argmax(z, axis=1)
		for metric in self.custom_metrics:
			metric.update_state(true_y, predictions)
		
		return {
			"lwr_loss": loss,
			**{m.name: m.result() for m in self.custom_metrics}
		}