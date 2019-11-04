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
                        word_level=False,
                        single_text=False,
                        max_gen_length=300,
                        top_n=3,
                        prefix=None,
                        synthesize=False,
                        stop_tokens=[' ', '\n']):
    '''
    Generates and returns a single text.
    '''

    collapse_char = ' ' if word_level else ''
    end = False

    if word_level and prefix:
        punct = '!"#$%&()*+,-./:;<=>?@[\]^_`{|}~\\n\\t\'‘’“”’–—'
        prefix = re.sub('([{}])'.format(punct), r' \1 ', prefix)
        prefix_t = [x.lower() for x in prefix.split()]

    if not word_level and prefix:
        prefix_t = list(prefix)

    if single_text:
        text = prefix_t if prefix else ['']
        max_gen_length += maxlen
    else:
        text = [meta_token] + prefix_t if prefix else [meta_token]

    next_char = ''

    if not isinstance(temperature, list):
        temperature = [temperature]

    if len(model.inputs) > 1:
        model = Model(inputs=model.inputs[0], outputs=model.outputs[1])

    while not end and len(text) < max_gen_length:
        encoded_text = text_generation_encode_sequence(text[-maxlen:],
                                                  vocab, maxlen)
        next_temperature = temperature[(len(text) - 1) % len(temperature)]

        next_index = text_generate_sample(model.predict(encoded_text, batch_size=1)[0], next_temperature)
        next_char = indices_char[next_index]
        text += [next_char]
        if next_char == meta_token or len(text) >= max_gen_length:
            end = True
        generation_break = (next_char in stop_tokens or word_level or len(stop_tokens) == 0)
        if synthesize and generation_break:
            break
            
    if single_text:
        text = text[:maxlen]
    else:
        text = text[1:]
        if meta_token in text:
            text.remove(meta_token)
    
    text_linked = collapse_char.join(text)

    if word_level:
        punct = '\\n\\t'
        text_linked = re.sub(" ([{}]) ".format(punct), r'\1', text_linked)
    
    return text_linked, end


def text_generation_encode_sequence(text, vocab, maxlen):
    encoded = np.array([vocab.get(val, 0) for val in text])
    
    return sequence.pad_sequences([encoded], maxlen=maxlen) 


def text_generation_texts_from_file(file_path, header=True, delim="\n", is_csv=False):
    '''
    Retrieves texts from a newline-delimited file and returns as a list.
    '''

    with open(file_path, "r", encoding="utf=8", errors="ignore") as f:
        if header:
            f.readline()
        if is_csv:
            texts = []
            reader = csv.reader(f)
            for row in reader:
                texts.append(row)
        else:
            texts = [line.rstrip(delim) for line in f]
    