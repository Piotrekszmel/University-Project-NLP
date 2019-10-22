from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
from utilities.data_preparation import print_dataset_statistics, labels_to_categories, categories_to_onehot
from sklearn.model_selection import train_test_split
from embeddings.WordVectorsManager import WordVectorsManager
from modules.CustomPreProcessor import CustomPreProcessor
from modules.EmbeddingsExtractor import EmbeddingsExtractor
import numpy as np
import random


def prepare_dataset(X, y, pipeline, y_one_hot=True, y_as_is=False):
    try:
        print_dataset_statistics(y)
    except:
        pass

    X = pipeline.fit_transform(X)

    if y_as_is:
        try:
            return X, np.asarray(y, dtype=float)
        except:
            return X, y

    y_cat = labels_to_categories(y)

    if y_one_hot:
        return X, categories_to_onehot(y_cat)


