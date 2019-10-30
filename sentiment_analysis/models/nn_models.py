from keras.constraints import maxnorm
from keras.engine import Input
from keras.engine import Model
from keras.layers import Dropout, Dense, Bidirectional, LSTM, \
    Embedding, GaussianNoise, Activation, Flatten, \
    RepeatVector, MaxoutDense, GlobalMaxPooling1D, \
    Convolution1D, MaxPooling1D, concatenate, Conv1D
from keras.models import Sequential
from keras.optimizers import Adam
from keras.regularizers import l2
from utilities.layers import AttentionWithContext, Attention, MeanOverTime
from sklearn import preprocessing


def embeddings_layer(max_length, embeddings, trainable=False, masking=False,
                     scale=False, normalize=False):
    if scale:
        print("Scaling embedding weights...")
        embeddings = preprocessing.scale(embeddings)
    if normalize:
        print("Normalizing embedding weights...")
        embeddings = preprocessing.normalize(embeddings)

    vocab_size = embeddings.shape[0]
    embedding_size = embeddings.shape[1]

    _embedding = Embedding(
        input_dim=vocab_size,
        output_dim=embedding_size,
        input_length=max_length if max_length > 0 else None,
        trainable=trainable,
        mask_zero=masking if max_length > 0 else False,
        weights=[embeddings]
    )

    return _embedding