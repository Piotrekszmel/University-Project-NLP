from text_generator import text_generator

text_gen = text_generator("text_generation_weights.hdf5", "text_generation_vocab.json")
#text_gen.load("text_generation_weights.hdf5")
print(chr(27) + "[2J")
text_gen.generate_samples()