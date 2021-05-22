import tensorflow as tf
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras import activations
    

class Dense(tf.keras.layers.Dense,Layer):
  def __init__(self,
               units,
               activation=None,
               **kwargs
               ):
    super(Dense, self).__init__(
        units,
        use_bias=True,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs
        )
    self.units = int(units) if not isinstance(units, int) else units
    self.activation = activations.get(activation)