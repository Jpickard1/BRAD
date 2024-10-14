from flask import Flask, request, jsonify
from flask import flash, redirect, url_for
from werkzeug.utils import secure_filename
import os
from BRAD.brad import chatbot
from BRAD.rag import create_database

UPLOAD_FOLDER = '/usr/src/uploads'
DATABASE_FOLDER = '/usr/src/RAG_Database/'
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

@app.route("/rag_upload", methods=['POST'])
def upload_file():
    file_list = request.files.getlist("rag_files")
    for file in file_list:
        if file.filename == '':
            response = {"message": "no uploaded file"}
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            response = {"message": "File uploaded successfully"}
    
    print("File uploads done")

    # creating chromadb with uploaded data
    print("running database creation")
    create_database(docsPath=UPLOAD_FOLDER, dbPath=DATABASE_FOLDER)
    return jsonify(response)