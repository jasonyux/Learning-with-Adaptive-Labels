import tensorflow as tf


class ClassificationModel(tf.keras.Model):
	"""Standard Trainnig Procedure"""
	def __init__(self, encoder, output_dim):
		super(ClassificationModel, self).__init__()
		self.encoder = encoder
		self.output_dim = output_dim # e.g. number of classes
		
		regularizer = tf.keras.regularizers.L1L2(0, 0.1)
		self.classification_head = tf.keras.layers.Dense(
			self.output_dim,
			kernel_regularizer=regularizer, 
			bias_regularizer=regularizer
		)

	def call(self, x, training = False, logit = False):
		z = self.encoder(x, training=training)
		y = self.classification_head(z, training=training)
		if not logit:
			y = tf.keras.layers.Softmax()(y)
		return z, y

	def compile(self, loss, optimizer, metrics:list):
		super(ClassificationModel, self).compile(run_eagerly=True)
		self.loss = loss
		self.optimizer = optimizer
		self.custom_metrics = metrics
		return

	@property
	def metrics(self):
		# We list our `Metric` objects here so that `reset_states()` can be
		# called automatically at the start of each epoch
		# or at the start of `evaluate()`.
		# If you don't implement this property, you have to call
		# `reset_states()` yourself at the time of your choosing.
		return self.custom_metrics

	def train_step(self, data):
		in_x, in_y = data
		with tf.GradientTape() as tape:
			_, pred_y = self(in_x, training=True)
			total_loss = self.loss(in_y, pred_y)

		gradients = tape.gradient(total_loss, self.trainable_weights)
		self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))
		
		# update metrics
		true_y = tf.math.argmax(in_y, axis=1)
		predictions = tf.math.argmax(pred_y, axis=1)

		for metric in self.custom_metrics:
			metric.update_state(true_y, predictions)
		
		return {
			"total_loss": total_loss, 
			**{m.name: m.result() for m in self.custom_metrics}
		}

	def test_step(self, data):
		in_x, in_y = data
		_, pred_y = self(in_x, training=True) # so that batch norm is applied
		total_loss = self.loss(in_y, pred_y)

		# update metrics
		true_y = tf.math.argmax(in_y, axis=1)
		predictions = tf.math.argmax(pred_y, axis=1)
		
		for metric in self.custom_metrics:
			metric.update_state(true_y, predictions)

		return {
			"total_loss": total_loss,
			**{m.name: m.result() for m in self.custom_metrics}
		}