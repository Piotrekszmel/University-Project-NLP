import errno
import os
import pickle
import numpy as np
import sys
sys.path.append('..')
from utilities.ResourceManager import ResourceManager


class WordVectorsManager(ResourceManager):
    def __init__(self, corpus=None, dim=None, omit_non_english=False):
        super().__init__()

        self.omit_non_english = omit_non_english
        self.wv_filename = "{}.{}d.txt".format(corpus, str(dim))
        self.parsed_filename = "{}.{}d.pickle".format(corpus, str(dim))

    def is_ascii(self, text):
      try:
        text.encode('ascii')
        return True
      except:
        return False

    def write(self):