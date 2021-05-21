from tensorflow.python.keras.layers import recurrent
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.keras import activations
from tensorflow.python.ops import nn
from tensorflow.python.framework import config
from tensorflow.python.framework import ops
import tensorflow as tf
def _use_new_code():
  return False

class LSTM(tf.keras.layers.LSTM,recurrent.DropoutRNNCellMixin, recurrent.LSTM):
  def __init__(self,
               units,
               activation='tanh',
               recurrent_activation='sigmoid',
               use_bias=True,
               kernel_initializer='glorot_uniform',
               recurrent_initializer='orthogonal',
               bias_initializer='zeros',
               unit_forget_bias=True,
               kernel_regularizer=None,
               recurrent_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               recurrent_constraint=None,
               bias_constraint=None,
               dropout=0.,
               recurrent_dropout=0.,
               return_sequences=False,
               return_state=False,
               go_backwards=False,
               stateful=False,
               time_major=False,
               unroll=False,
               **kwargs):
    self.return_runtime = kwargs.pop('return_runtime', False)

    super(LSTM, self).__init__(
        units,
        activation=activation,
        recurrent_activation=recurrent_activation,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        recurrent_initializer=recurrent_initializer,
        bias_initializer=bias_initializer,
        unit_forget_bias=unit_forget_bias,
        kernel_regularizer=kernel_regularizer,
        recurrent_regularizer=recurrent_regularizer,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        kernel_constraint=kernel_constraint,
        recurrent_constraint=recurrent_constraint,
        bias_constraint=bias_constraint,
        dropout=dropout,
        recurrent_dropout=recurrent_dropout,
        implementation=kwargs.pop('implementation', 2),
        return_sequences=return_sequences,
        return_state=return_state,
        go_backwards=go_backwards,
        stateful=stateful,
        time_major=time_major,
        unroll=unroll,
        **kwargs)

    self.state_spec = [
        InputSpec(shape=(None, dim)) for dim in (self.units, self.units)
    ]
    self._could_use_gpu_kernel = (
        self.activation in (activations.tanh, nn.tanh) and
        self.recurrent_activation in (activations.sigmoid, nn.sigmoid) and
        recurrent_dropout == 0 and not unroll and use_bias and
        ops.executing_eagerly_outside_functions())
    if config.list_logical_devices('GPU'):
      # Only show the message when there is GPU available, user will not care
      # about the cuDNN if there isn't any GPU.
      if self._could_use_gpu_kernel:
        logging.debug(_CUDNN_AVAILABLE_MSG % self.name)
      else:
        logging.warn(_CUDNN_NOT_AVAILABLE_MSG % self.name)

    if _use_new_code():
      self._defun_wrapper = _DefunWrapper(time_major, go_backwards, 'lstm')
