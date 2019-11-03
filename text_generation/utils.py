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


def textgenrnn_generate(model, vocab,
                        indices_char, temperature=0.5,
                        maxlen=40, meta_token='<s>',
                        single_text=False,
                        max_gen_length=300,
                        top_n=3,
                        prefix=None,
                        synthesize=False,
                        stop_tokens=[' ', '\n']):
    '''
    Generates and returns a single text.
    '''

    collapse_char = " " 
    end = False

    if prefix:
        punct = '!"#$%&()*+,-./:;<=>?@[\]^_`{|}~\\n\\t\'‘’“”’–—'
        prefix = re.sub("([{}])".format(punct), r' \1', prefix)
        prefix_t = [w.lower() for w in prefix.split()]
    
    if single_text: 
        text = prefix_t if prefix else [""]
        max_gen_length += maxlen
    else:
        text = [meta_token] + prefix_t if prefix else [meta_token]