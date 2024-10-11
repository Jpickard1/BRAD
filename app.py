from flask import Flask, request, jsonify
from flask import flash, redirect, url_for
from werkzeug.utils import secure_filename
import os
from BRAD.brad import chatbot

UPLOAD_FOLDER = '/usr/src/uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

brad = chatbot(interactive=False)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/invoke", methods=['POST'])
def invoke_request():
    request_data = request.json
    brad_query = request_data.get("message")
    brad_response = brad.invoke(brad_query)
    response_data = {
        "response": brad_response
    }
    return jsonify(response_data)
    # brad.invoke("Hey brad. what are you upto?")
    # return "<p>Hello, World!</p>"

@app.route("/rag_upload", methods=['POST'])
def upload_file():
    for _file_id, file in request.files.items():
        if file.filename == '':
            response = {"message": "no uploaded file"}
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            response = {"message": "File uploaded successfully"}
        return jsonify(response)
    print("Uploaded file successfully")