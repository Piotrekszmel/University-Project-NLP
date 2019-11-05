from keras.optimizers import RMSprop
from keras.layers import Input, Embedding, Dense, LSTM, Bidirectional
from keras.layers import concatenate, Reshape, SpatialDropout1D
from keras.models import Model
from keras import backend as K
from Attention import Attention


def text_generation_model(num_classes, cfg, context_size, weights_path,
                          dropout=0.0, optimizer=RMSprop(lr=4e-3, rho=0.99)):
  '''
  Builds the model architecture for text generation and
  loads the specified weights for the model.
  '''

  input = Input(cfg["max_length", ], name="input")
  embedded = Embedding(num_classes, cfg["dim_embeddings"], input_length=cfg["max_length"],
                      name="embedding")(input)
  
  if dropout > 0.0:
    embedded = SpatialDropout1D(dropout, name="dropout")(embedded)