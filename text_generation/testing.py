from text_generator import text_generator
import glob
import os



model_cfg = {
    'word_level': False,   # set to True if want to train a word-level model (requires more data and smaller max_length)
    'rnn_size': 128,   # number of LSTM cells of each layer (128/256 recommended)
    'rnn_layers': 3,   # number of LSTM layers (>=2 recommended)
    'rnn_bidirectional': False,   # consider text both forwards and backward, can give a training boost
    'max_length': 30,   # number of tokens to consider before predicting the next (20-40 for characters, 5-10 for words recommended)
    'max_words': 10000,   # maximum number of words to model; the rest will be ignored (word-level model only)
}
train_cfg = {
    'line_delimited': True,   # set to True if each text has its own line in the source file
    'num_epochs': 22,   # set higher to train the model for longer
    'gen_epochs': 5,   # generates sample text from model after given number of epochs
    'train_size': 0.8,   # proportion of input data to train on: setting < 1.0 limits model from learning perfectly
    'dropout': 0.0,   # ignore a random proportion of source tokens each epoch, allowing model to generalize better
    'validation': False,   # If train__size < 1.0, test on holdout dataset; will make overall training slower
    'is_csv': False   # set to True if file is a CSV exported from Excel/BigQuery/pandas
}

"""
texts = []
files = glob.glob("/home/pito/Desktop/projects/University-Project-NLP/sentiment_analysis/data/datasets" + "/*.tsv")
for file in files:
    for line in open(os.path.join("/home/pito/Desktop/projects/University-Project-NLP/sentiment_analysis/data/datasets/", file), "r", encoding="utf-8").readlines():
        columns = line.strip().split("\t")
        texts.append(columns[2:])

with open('datasets/twitter.txt', 'w') as f:
    for text in texts :
        f.write("{}\n".format(*text))

"""

model_name = 'twitter_128_LSTM'
file_name = "datasets/twitter.txt" 
textgen = text_generator(name=model_name)
train_function = textgen.train_from_file if train_cfg['line_delimited'] else textgen.train_from_largetext_file
train_function(
    file_path=file_name,
    new_model=True,
    num_epochs=train_cfg['num_epochs'],
    gen_epochs=train_cfg['gen_epochs'],
    batch_size=1024,
    train_size=train_cfg['train_size'],
    dropout=train_cfg['dropout'],
    validation=train_cfg['validation'],
    is_csv=train_cfg['is_csv'],
    rnn_layers=model_cfg['rnn_layers'],
    rnn_size=model_cfg['rnn_size'],
    rnn_bidirectional=model_cfg['rnn_bidirectional'],
    max_length=model_cfg['max_length'],
    dim_embeddings=100,
    word_level=model_cfg['word_level'])

textgen = text_generator(weights_path='weights/shakespeare_128_BIDIRECTIONAL_weights.hdf5',
                       vocab_path='vocabs/shakespeare_128_BIDIRECTIONAL_vocab.json',
                       config_path='configs/shakespeare_128_BIDIRECTIONAL_config.json')
                       
textgen.generate_samples(max_gen_length=1000, temperatures=[0.2])

