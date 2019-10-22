import re
import os
import glob
from utilities.data_preparation import clean_text


class DataLoader:
  def __init__(self, verbose=True):
    self.verbose = verbose
    self.separator = "\t"
    self.datasets_path = os.path.join(os.getcwd(), 'data/datasets')

    print()

  
  def parse_file(self, filename, with_topic=False):
    
    """
        Reads the text file and returns a dictionary in the form:
        tweet_id = (sentiment, text)
        :param with_topic:
        :param filename: the complete file name
        :return:
    """

    data ={}
    


