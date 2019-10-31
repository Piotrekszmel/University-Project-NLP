import pickle
import os
import numpy as np
import sys
from keras.callbacks import ModelCheckpoint
from keras.layers import LSTM
from utilities.callbacks import MetricsCallback, PlottingCallback
from utilities.data_preparation import get_labels_to_categories_map, get_class_weights2, onehot_to_categories
from sklearn.metrics import f1_score, precision_score
from sklearn.metrics import recall_score
from data.data_loader import DataLoader
from models.nn_models import build_attention_RNN
from utilities.data_loader import get_embeddings, Loader, prepare_datasetimport pickle

np.random_seed(1337)



def Sentiment_Analysis(WV_CORPUS, WV_DIM, max_length, PERSIST,  FINAL=True):
  """
  ##Final:
  - if FINAL == False,  then the dataset will be split in {train, val, test}
  - if FINAL == True,   then the dataset will be split in {train, val}.

  ##PERSIST
  # set PERSIST = True, in order to be able to use the trained model later
  """
