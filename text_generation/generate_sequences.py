from keras.callbacks import LearningRateScheduler, Callback
from keras.models import Model, load_model
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.utils import Sequence
from keras import backend as K
from utils import text_generation_encode_cat
import numpy as np


def generate_sequences_from_texts(texts, indices_list, text_gen, context_labels, batch_size=128):
  word_level = text_gen.config["word_level"]
  single_text = text_gen.config["single_text"]
  max_length = text_gen.config["max_len"]
  meta_token = text_gen.META_TOKEN

  if word_level:
    tokenizer = Tokenizer(filters="", char_level=True)
    tokenizer.index_word = text_gen.vocab
  else:
    tokenizer = text_gen.tokenizer
  
  while True:
    np.random.shuffle(indices_list)

    X_batch = []
    y_batch = []
    context_batch = []
    count_batch = 0

    for row in range(indices_list.shape[0]):
      text_index = indices_list[row, 0]
      end_index = indices_list[row, 1]

      text = texts[text_index]

      if not single_text:
        text = [meta_token] + list(text) + [meta_token]
      
      if end_index > max_length:
        x = text[end_index - max_length : end_index + 1]
      else:
        x = text[0: end_index + 1]
      
      y = text[end_index + 1]
      
      if y in text_gen.vocab:
        x = process_sequence([x], text_gen, tokenizer)
        y = text_generation_encode_cat([y], text_gen.vocab)

        X_batch.append(x)
        y_batch.append(y)

        if context_labels is not None:
          context_batch.append(context_labels[text_index])

        count_batch += 1

        if count_batch % batch_size == 0:
          X_batch = np.squeeze(np.array(X_batch))
          Y_batch = np.squeeze(np.array(Y_batch))
          context_batch = np.squeeze(np.array(context_batch))

          if context_labels is not None:
            yield ([X_batch, context_batch], [y_batch, y_batch])
          else:
            yield (X_batch, y_batch)
          X_batch = []
          y_batch = []
          context_batch = []
          count_batch = 0

      
def process_sequence(X, text_gen, tokenizer):
  X = tokenizer.texts_to_sequences(X)
  X = sequence.pad_sequences(X, maxlen=text_gen.config["max_length"])

  return X