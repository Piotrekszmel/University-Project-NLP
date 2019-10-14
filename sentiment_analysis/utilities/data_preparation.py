from collections import Counter
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import compute_class_weight
import numpy as np


def get_class_labels(y):
  """
    Get the class labels
    :param y: list of labels, ex. ['positive', 'negative', 'positive', 'neutral', 'positive', ...]
    :return: sorted unique class labels
  """
  return np.unique(y)


def labels_to_categories(y):
  """
    Labels to categories
    :param y: list of labels, ex. ['positive', 'negative', 'positive', 'neutral', 'positive', ...]
    :return: list of categories, ex. [0, 2, 1, 2, 0, ...]
  """
  encoder = LabelEncoder()
  encoder.fit(y)
  y_num = encoder.transform(y)
  return y_num


def get_labels_to_categories_map(y):
  """
    Get the mapping of class labels to numerical categories
    :param y: list of labels, ex. ['positive', 'negative', 'positive', 'neutral', 'positive', ...]
    :return: dictionary with the mapping
  """
  labels = get_class_labels(y)
  return {l: i for i, l in enumerate(labels)}


def categories_to_onehot(y):
  """
   Transform categorical labels to one-hot vectors
   :param y: list of categories, ex. [0, 2, 1, 2, 0, ...]
   :return: list of one-hot vectors, ex. [[0, 0, 1], [1, 0, 0], [0, 0, 1], [0, 1, 0], [0, 0, 1], ...]
  """
  return np_utils.to_categorical(y)


def onehot_to_categories(y):
  """
   Transform categorical labels to one-hot vectors
   :param y: list of one-hot vectors, ex. [[0, 0, 1], [1, 0, 0], [0, 0, 1], [0, 1, 0], [0, 0, 1], ...]
   :return: list of categories, ex. [0, 2, 1, 2, 0, ...]
  """
  return np.asarray(y).argmax(axis=1)


