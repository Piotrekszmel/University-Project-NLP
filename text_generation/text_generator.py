from keras.callbacks import LearningRateScheduler, Callback
from keras.models import Model, load_model
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.utils import multi_gpu_model
from keras.optimizers import RMSprop
from keras import backend as K
from sklearn.preprocessing import LabelBinarizer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
import h5py
from pkg_resources import resource_filename
from model import text_generation_model
from generate_sequences import *
from utils import *
import csv
import re


class text_generator:
  META_TOKEN = "<s>"
  config = config = {
        'rnn_layers': 2,
        'rnn_size': 128,
        'rnn_bidirectional': False,
        'max_length': 40,
        'max_words': 10000,
        'dim_embeddings': 100,
        'word_level': False,
        'single_text': False
    }

  default_config = config.copy()
  
  def __init__(self, weights_path, vocab_path, config_path, name="text_generator"):
    
    if weights_path is None:
      weights_path = resource_filename(__name__, "text_generator_weights.hdf5")
    
    if vocab_path is None:
      vocab_path = resource_filename(__name__, "text_generator_vocab.json")
    
    if config_path is not None:
      with open("config_path", "r", encoding="utf-8", errors="ignore") as json_file:
        self.config = json.load(json_file)
    
    self.config.update({'name': name})
    self.default_config.update({'name': name})

    with open(vocab_path, 'r', encoding='utf8', errors='ignore') as json_file:
      self.vocab = json.load(json_file)

    self.tokenizer = Tokenizer(filters="", lower=False, char_level=True)
    self.tokenizer.word_index = self.vocab
    self.num_classes = len(self.vocab) + 1
    self.model = text_generation_model(self.num_classes, 
                                       cfg=self.config,
                                       weights_path=weights_path)
    self.indices_char = dict((self.vocab[c], c) for c in self.vocab)
    