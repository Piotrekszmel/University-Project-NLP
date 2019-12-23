from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.python.keras.backend import set_session

from sentiment_analysis.evaluate import predict_sentiment_single_tweet 
from sentiment_analysis.models.utils import create_model
from sentiment_analysis.utilities.data_loader import Loader


app = Flask(__name__)


@app.route("/", methods=['GET', 'POST'])
def index():
  return render_template("index.html")

@app.route("/Sentiment", methods=['GET', 'POST'])
def sentiment():
  if request.method == "POST":
    message = request.form["text"]
    with graph.as_default():
      set_session(sess)
      predict_sentiment_single_tweet(message, model, loader.pipeline)
  return render_template("sentiment.html")
      

@app.route("/Generation", methods=['GET', 'POST'])
def text_generation():
  return render_template("text_generation.html")

if __name__ == "__main__":
  sess = tf.Session()
  set_session(sess)
  global graph, model
  model, word_indices = create_model("datastories.twitter", 300, "sentiment_analysis/weights/bi_model_weights_1.h5")
  graph = tf.get_default_graph()

  loader = Loader(word_indices, text_lengths=50)
  app.run(debug=True, host="0.0.0.0", port=5004)
