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

    """
    Keras Layer that implements an Attention mechanism for temporal data.
    Supports Masking.
    Follows the work of Raffel et al. [https://arxiv.org/abs/1512.08756]
    # Input shape
      3D tensor with shape: `(samples, steps, features)`.
    # Output shape
       2D tensor with shape: `(samples, features)`.
    :param kwargs:
      Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
      The dimensions are inferred based on the output shape of the RNN.
      Note: The layer has been tested with Keras 1.x
      Example:
          # 1
          model.add(LSTM(64, return_sequences=True))
          model.add(Attention())
          # next add a Dense layer (for classification/regression) or whatever...
          # 2 - Get the attention scores
          hidden = LSTM(64, return_sequences=True)(words)
          sentence, word_scores = Attention(return_attention=True)(hidden)
    """
    
    self.supports_masking = True
    self.return_attention = return_attention
    self.init = initializers.get('glorot_uniform')

    self.W_regularizer = regularizers.get(W_regularizer)
    self.b_regularizer = regularizers.get(b_regularizer)

    self.W_constraint = constraints.get(W_constraint)
    self.b_constraint = constraints.get(b_constraint)

    self.bias = bias
    super().__init__(**kwargs)

  def build(self, input_shape):
    assert len(input_shape) == 3

    self.W = self.add_weight(name='{}_W'.format(self.name), 
                                 shape=(input_shape[-1],),
                                 initializer=self.init,
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
    if self.bias:
      self.b = self.add_weight(name='{}_b'.format(self.name),
                               shape=(input_shape[1],), 
                               initializer="zero",
                               regularizer=self.b_regularizer,
                               constraint=self.b_constraint)
    else:
      self.b = None
    
    self.built = True

