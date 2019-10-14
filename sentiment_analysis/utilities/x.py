import numpy as np


def get_class_labels(y):
  """
    Get the class labels
    :param y: list of labels, ex. ['positive', 'negative', 'positive', 'neutral', 'positive', ...]
    :return: sorted unique class labels
  """
  return np.unique(y)




def labels_to_categories_map(y):
  """
    Get the mapping of class labels to numerical categories
    :param y: list of labels, ex. ['positive', 'negative', 'positive', 'neutral', 'positive', ...]
    :return: dictionary with the mapping
  """
  labels = get_class_labels(y)
  return {l: i for i, l in enumerate(labels)}


print(labels_to_categories_map([1,3,7, 11, 11, 11, 7, 3, 1, 1]))