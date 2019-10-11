from flask import Flask, render_template


app = Flask(__name__)


@app.route("/", methods=['GET', 'POST'])
def index():
  return render_template("index.html")


@app.route("/Sentiment")
def sentiment():
  return render_template("sentiment.html")

if __name__ == "__main__":
  app.run(debug=True, host="localhost", port=5004)