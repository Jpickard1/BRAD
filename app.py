from flask import Flask, request, jsonify
from BRAD.brad import chatbot

app = Flask(__name__)
brad = chatbot(interactive=False)

@app.route("/invoke", methods=['POST'])
def hello_world():
    request_data = request.json()
    brad_query = request_data.get("message")
    brad_response = brad.invoke(brad_query)
    response_data = {
        "response": brad_response
    }
    return jsonify(response_data)
    # brad.invoke("Hey brad. what are you upto?")
    # return "<p>Hello, World!</p>"
