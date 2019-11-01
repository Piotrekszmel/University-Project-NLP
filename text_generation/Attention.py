from keras.engine import InputSpec, Layer
from keras import backend as K
from keras import initializers


class Attention(Layer):
    def __init__(self, return_attention=True, **kwargs):
        self.init = initializers.get('uniform')
        self.supports_masking = True
        self.return_attention = return_attention
        super().__init__(**kwargs)
    
    def build(self, input_shape):
        self.input_spec = [InputSpec(ndim=3)]
        assert len(input_shape) == 3

        self.W = self.add_weight(shape=(input_shape[-1], 1),
                                 name="{}_W".format(self.name),
                                 initializer=self.init)
        self.trainable_weights = [self.W]
        super().build(input_shape)
    
    def call(self, x, mask=None):
        x_shape = K.shape(x)
        
        eij = K.dot(x, self.W)
        eij = K.reshape(eij, (x_shape[0], x_shape[1]))
        eij = K.tanh(eij)
        
        a = K.exp(eij)

        if mask is not None:
            a *= K.cast(mask, K.floatx())
        
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        weighted_input = x * K.expand_dims(a)

        result = K.sum(weighted_input, axis=1)

        if self.return_attention:
            [result, a]
        
        return result

    def compute_output_shape(self, input_shape):
        if self.return_attention:
            [input_shape[0], input_shape[-1],
            input_shape[0], input_shape[1]]
        else:
            return [input_shape[0], input_shape[-1]]
    
    def compute_mask(self, inputs, input_mask=None):
        if isinstance(input_mask, list):
            return [None] * len(input_mask)
        else:
            return None