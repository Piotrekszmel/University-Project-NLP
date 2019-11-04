from keras.optimizers import RMSprop
from keras.layers import Input, Embedding, Dense, LSTM, Bidirectional
from keras.layers import concatenate, Reshape, SpatialDropout1D
from keras.models import Model
from keras import backend as K
from Attention import Attention

