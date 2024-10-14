# STANDARD python imports
import os
import json

# Imports for building RESTful API
from flask import Flask, request, jsonify
from flask import flash, redirect, url_for
from werkzeug.utils import secure_filename

# Imports for BRAD library
from BRAD.brad import chatbot
from BRAD.rag import create_database

# HARDCODED VALUES
UPLOAD_FOLDER = '/usr/src/uploads'
DATABASE_FOLDER = '/usr/src/RAG_Database/'
SOURCE_FOLDER = '/usr/src/brad'
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

@app.route("/open_sessions", methods=['GET'])
def get_open_sessions():
    """
    This endpoint lets the front end access previously opened chat sessions.
    """
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: October 14, 2024

    # Get path to output directories
    path_to_output_directories = brad.chatstatus['config']['log_path']
    
    # Get list of directories at this location
    try:
        open_sessions = [name for name in os.listdir(path_to_output_directories) 
                         if os.path.isdir(os.path.join(path_to_output_directories, name))]
        
        # Return the list of open sessions as a JSON response
        return jsonify({"open_sessions": open_sessions})
    
    except FileNotFoundError:
        return jsonify({"error": "Directory not found"})
    
    except Exception as e:
        return jsonify({"error": str(e)})

