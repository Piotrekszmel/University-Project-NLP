from keras import backend as K, regularizers, constraints, initializers
from keras.engine.topology import Layer


def dot_product(x, kernel):
  return K.dot(x, kernel)


class MeanOverTime(Layer):
  
  """
  Layer that computes the mean of timesteps returned from an RNN and supports masking
  Example:
    activations = LSTM(64, return_sequences=True)(words)
    mean = MeanOverTime()(activations)
  """

  def __init__(self, **kwargs):
    self.supports_masking = True
    super().__init__(**kwargs)

  def call(self, x, mask=None):
        if mask is not None:
            mask = K.cast(mask, 'float32')
            return K.cast(K.sum(x, axis=1) / K.sum(mask, axis=1, keepdims=True),
                          K.floatx())
        else:
            return K.mean(x, axis=1)
  
  def compute_output_shape(self, input_shape):
    return input_shape[0], input_shape[-1]
  
  def compute_mask(self, input, input_mask=None):
    return None


class Attention(Layer):
  def __init__(self,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True,
                 return_attention=False,
                 **kwargs):

  pass
