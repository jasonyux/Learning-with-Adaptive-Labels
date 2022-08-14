import tensorflow as tf

class Resnet50Encoder(tf.keras.layers.Layer):
	def __init__(self, input_shape, num_classes, **kwargs):
		super(Resnet50Encoder, self).__init__(**kwargs)
		if input_shape[1] < 32: # up sample it
			input_shape = (input_shape[0]*2, input_shape[1]*2, input_shape[2])
		self.resnet_encoder = tf.keras.applications.resnet50.ResNet50(
			include_top=False,
			weights='imagenet',
			input_tensor=None,
			input_shape=input_shape,
			pooling=None,
			classes=num_classes
		)
		self.up_sample = tf.keras.layers.UpSampling2D(size=(2,2), data_format="channels_last")

	def call(self, inputs, training=False):
		if inputs.shape[1] < 32: # up sample it
			inputs = self.up_sample(inputs)
		z = self.resnet_encoder(inputs, training=training)
		z = tf.keras.layers.Flatten(data_format="channels_last")(z)
		return z


class DenseNet121Encoder(tf.keras.layers.Layer):
	def __init__(self, input_shape, num_classes, **kwargs):
		super(DenseNet121Encoder, self).__init__(**kwargs)
		if input_shape[1] < 32: # up sample it
			input_shape = (input_shape[0]*2, input_shape[1]*2, input_shape[2])
		self.densenet_encoder = tf.keras.applications.densenet.DenseNet121(
			include_top=False,
			weights='imagenet',
			input_tensor=None,
			input_shape=input_shape,
			pooling=None,
			classes=num_classes
		)
		self.up_sample = tf.keras.layers.UpSampling2D(size=(2,2), data_format="channels_last")

	def call(self, inputs, training=False):
		if inputs.shape[1] < 32:
			inputs = self.up_sample(inputs)
		z = self.densenet_encoder(inputs, training=training)
		z = tf.keras.layers.Flatten(data_format="channels_last")(z)
		return z


class EfficientNetEncoder(tf.keras.layers.Layer):
	def __init__(self, input_shape, num_classes, **kwargs):
		super(EfficientNetEncoder, self).__init__(**kwargs)
		if input_shape[1] < 32: # up sample it
			input_shape = (input_shape[0]*2, input_shape[1]*2, input_shape[2])
		self.efficientnet_encoder = tf.keras.applications.efficientnet.EfficientNetB0(
			include_top=False,
			weights='imagenet',
			input_tensor=None,
			input_shape=input_shape,
			pooling=None,
			classes=num_classes
		)
		self.up_sample = tf.keras.layers.UpSampling2D(size=(2,2), data_format="channels_last")

	def call(self, inputs, training=False):
		if inputs.shape[1] < 32:
			inputs = self.up_sample(inputs)
		z = self.efficientnet_encoder(inputs, training=training)
		z = tf.keras.layers.Flatten(data_format="channels_last")(z)
		return z