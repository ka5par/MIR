

from flask import Flask, render_template
app = Flask(__name__,  static_folder='static')


@app.route("/")
def hello():
    message = "The Flask Shop"
    return render_template('index.html')


if __name__ == "__main__":
  app.debug = False
  app.run(host='0.0.0.0',port=8887)

