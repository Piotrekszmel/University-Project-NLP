import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class EmbeddingsExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, word_indices, max_lengths=None, add_tokens=None, unk_policy="random", **kwargs):
        """
            :param word_indices:
            :param max_lengths: list of integers indicating the max limit of words
                                for each data list in X
            :param unk_policy: "random","zero"
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
        
        for i, doc in enumerate(X):
            Xs[i, 0] = self.word_indices.get("<s>", 0)
            for j, token in enumerate(doc[:max_lengths]):
                if token in self.word_indices:
                    Xs[i, min(j + 1, max_lengths - 1)] = self.word_indices[token]

                else: 
                    if self.unk_policy == "random":
                        Xs[i, min(j + 1, max_lengths - 1)] = self.word_indices["<unk>"]
                    elif self.unk_policy == "zero":
                        Xs[i, min(j + 1, max_lengths - 1)] = 0

            if len(doc) + 1 < max_lengths:
                Xs[i, len(doc) + 1] = self.word_indices.get("</s>", 0)
        
        return Xs
    
    def index_text(self, sent, add_tokens=False):
        sent_words = []
