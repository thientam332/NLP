import tensorflow as tf
from tensorflow.python.keras.engine.base_layer import Layer
class Embedding(tf.keras.layers.Embedding,Layer):
  pass
