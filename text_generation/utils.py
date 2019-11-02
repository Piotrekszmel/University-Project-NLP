from keras.callbacks import LearningRateScheduler, Callback
from keras.models import Model, load_model
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras import backend as K
from sklearn.preprocessing import LabelBinarizer
from random import shuffle
from tqdm import trange
import numpy as np
import json
import h5py
import csv
import re


def text_generate_sample(preds, temperature, top_n=3):
    preds = np.asarray(preds).astype(np.float64)

    if temperature is None or temperature == 0.0:
        return np.argmax(preds)
    
    preds = np.log(preds + K.epsilon()) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)

    index = np.argmax(probas)

    if index == 0:
        index = np.argsort(preds)[-2]
    
    return index