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


