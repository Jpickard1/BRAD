# STANDARD python imports
import os
import json
import shutil

# Imports for building RESTful API
from flask import Flask, request, jsonify
from flask import flash, redirect, url_for
from werkzeug.utils import secure_filename

# Imports for BRAD library
from BRAD.agent import Agent
from BRAD.rag import create_database

# HARDCODED VALUES
UPLOAD_FOLDER = '/usr/src/uploads'
DATABASE_FOLDER = '/usr/src/RAG_Database/'
SOURCE_FOLDER = '/usr/src/brad'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

brad = Agent(interactive=False)

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
    path_to_output_directories = brad.state['config']['log_path']
    
    # Get list of directories at this location
    try:
        open_sessions = [name for name in os.listdir(path_to_output_directories) 
                         if os.path.isdir(os.path.join(path_to_output_directories, name))]
        
        # Return the list of open sessions as a JSON response
        message = jsonify({"open_sessions": open_sessions})
        print(f"{message=}")
        return message
    
    except FileNotFoundError:
        return jsonify({"error": "Directory not found"})
    
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/remove_sessions", methods=['POST'])
def remove_open_sessions():
    request_data = request.json
    session = request_data.get("message")  # Get the session name from the request
    path_to_output_directories = brad.state['config']['log_path']

    # Construct the full path to the session directory
    session_path = os.path.join(path_to_output_directories, session)

    # Check if the session directory exists
    if os.path.exists(session_path):
        try:
            # Remove the session directory
            shutil.rmtree(session_path)
            return jsonify({"success": True, "message": f"Session '{session}' removed."}), 200
        except Exception as e:
            return jsonify({"success": False, "message": str(e)}), 500
    else:
        return jsonify({"success": False, "message": f"Session '{session}' does not exist."}), 404
