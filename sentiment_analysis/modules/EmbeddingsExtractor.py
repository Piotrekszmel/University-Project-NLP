import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class EmbeddingsExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, word_indices, max_lengths=None, add_tokens=None, unk_policy="random", **kwargs):
        """
            :param word_indices:
            :param max_lengths: list of integers indicating the max limit of words
                                for each data list in X
            :param unk_policy: "random","zero","ignore"
        """

        self.word_indices = word_indices
        self.max_lengths = max_lengths
        self.add_tokens = add_tokens
        self.unk_policy = unk_policy
        self.hierarchical = kwargs.get("hierarchical", False)

    @staticmethod
    def sequences_to_fixed_length(X, length):
        Xs = np.zeros((X.size, length), dtype="int32")
        for i, x in enumerate(X):
            if x.size < length:
                Xs[i] = np.pad(x, (0, length - len(x) % length), "constant")
            elif x.size > length:
                Xs[i] = x[0:length]
        
        return Xs

    def get_fixed_size_topic(self, X, max_lengths):
        X = list(X)
        Xs = np.zeros((len(X), max_lengths), dtype="int32")
        
        for i, doc in enumerate(X)

