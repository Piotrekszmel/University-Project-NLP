import sys
sys.path.append('utilities/')
sys.path.append("models/")

from sentiment_analysis.utilities.data_loader import get_embeddings, Loader, prepare_dataset
from sentiment_analysis.models.nn_models import build_attention_RNN
from keras.layers import LSTM
import numpy as np

def predict_class(X, y, corpus, dim):
    tweets = []
    labels = []
    
    embeddings, word_indices = get_embeddings(corpus=corpus, dim=dim)
    loader = Loader(word_indices, text_lengths=50)
    X, y = prepare_dataset(X, y, loader.pipeline, False, True)

    nn_model = build_attention_RNN(embeddings, classes=3, max_length=50,
                                unit=LSTM, layers=2, cells=150,
                                bidirectional=True,
                                attention="simple",
                                noise=0.3,
                                final_layer=False,
                                dropout_final=0.5,
                                dropout_attention=0.5,
                                dropout_words=0.3,
                                dropout_rnn=0.3,
                                dropout_rnn_U=0.3,
                                clipnorm=1, lr=0.001, loss_l2=0.0001)
    
    nn_model.load_weights('sentiment_analysis/weights/bi_model_weights_1.h5')
    
    for tweet in X:
        tweet = tweet.reshape(50, 1).T
        predicted_y = nn_model.predict_classes(tweet)
        tweets.append(tweet)
        labels.append(predicted_y)
    return tweets, labels, y