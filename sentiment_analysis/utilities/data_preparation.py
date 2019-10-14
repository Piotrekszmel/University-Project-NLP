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
  


