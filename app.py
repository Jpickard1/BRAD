from flask import Flask
from BRAD.brad import chatbot

app = Flask(__name__)

@app.route("/")
def hello_world():
    brad = chatbot()
    brad.invoke("Hey brad. what are you upto?")
    return "<p>Hello, World!</p>"