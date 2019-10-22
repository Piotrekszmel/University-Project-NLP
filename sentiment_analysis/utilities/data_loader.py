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


def get_embeddings(corpus, dim):
    vectors = WordVectorsManager(corpus, dim).read()
    vocab_size = len(vectors)
    
    print("Loaded {} word vectors.".format(vocab_size))

    wv_map = {}
    pos = 0
    emb_matrix = np.ndarray((vocab_size + 2, dim), dtype="float32")

    for i, (word, vector) in enumerate(vectors.items()):
        if len(vector) > 199:
            pos = i + 1
            wv_map[word] = pos
            emb_matrix[pos] = vector

    pos += 1
    wv_map["<unk>"] = pos
    emb_matrix[pos] = np.random.uniform(low=-0.05, high=0.05, size=dim)

    return emb_matrix, wv_map