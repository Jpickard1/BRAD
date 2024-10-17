# STANDARD python imports
import os
import json
import shutil
import logging
import time

# Imports for building RESTful API
from flask import Flask, request, jsonify
from flask import flash, redirect, url_for
from werkzeug.utils import secure_filename

# Imports for BRAD library
from BRAD.agent import Agent
from BRAD.rag import create_database

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def delete_dirs_without_log(directory):
    # List only first-level subdirectories
    for subdir in os.listdir(directory):
        subdir_path = os.path.join(directory, subdir)
        
        # Check if it's a directory
        if os.path.isdir(subdir_path):
            log_file_path = os.path.join(subdir_path, 'log.json')
            
            # If log.json does not exist in the subdirectory, delete the subdirectory
            if not os.path.exists(log_file_path):
                shutil.rmtree(subdir_path)  # Recursively delete directory and its contents
                print(f"Deleted directory: {subdir_path}")


# Directory structure
UPLOAD_FOLDER = '/usr/src/uploads'
DATABASE_FOLDER = '/usr/src/RAG_Database/'

ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

TOOL_MODULES = ['RAG']

brad = Agent(interactive=False, tools=TOOL_MODULES)
PATH_TO_OUTPUT_DIRECTORIES = brad.state['config'].get('log_path')
delete_dirs_without_log(PATH_TO_OUTPUT_DIRECTORIES)

app = Flask(__name__)

DATA_FOLDER = os.path.join(app.root_path, 'data')
UPLOAD_FOLDER = os.path.join(DATA_FOLDER, 'uploads')
DATABASE_FOLDER = os.path.join(DATA_FOLDER, 'RAG_Database/')
if not os.path.exists(DATA_FOLDER):
    os.makedirs(DATA_FOLDER)
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(DATABASE_FOLDER):
    os.makedirs(DATABASE_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def parse_log_for_one_query(chatlog_query):
    # print(" ")
    # print(" ")
    # print("parse_log_for_one_query")
    if chatlog_query['process']['module'].upper() == 'RAG':
        # print("    RAG FOUND!")
        process = []
        for step in chatlog_query['process']['steps']:
            # print(f"        {step=}")
            if 'func' in step.keys() and step['func'].lower() == 'rag.retrieval':
                # print("retrieval !")
                docsText = step['docs-to-gui']
                sources = []
                chunks = []
                for doc in docsText:
                    sources.append(doc['source'])
                    chunks.append(doc['text'])
                process.append(('RAG-R', sources))
                process.append(('RAG-G', chunks))
            elif 'purpose' in step.keys() and step['purpose'].lower() == 'chat without RAG':
                # print("LLM solo !")
                process.append(('LLM-Generation', "Response generated with only the LLM."))
            # print(f"{process=}")
        # print(f"{process=}")
        return process
    else:
        return None


def parse_log_for_process_display(chat_history):
    # This is the pattern that the front end expects
    # passed_log_stages = [
    #     ('RAG-R', ['source 1', 'source 2', 'source 3']),
    #     ('RAG-G', ['This is chunk 1', 'This is chunk 2', 'This is chunk 3'])
    # ]
    for i in range(len(chat_history)):
        if chat_history[i][1] is not None:
            # print('replacing logs')
            # print(f"{chat_history=}")
            chat_history[i] = (chat_history[i][0], parse_log_for_one_query(chat_history[i][1]))
            # print(f"{chat_history=}")
    return chat_history # passed_log_stages

@app.route("/invoke", methods=['POST'])
def invoke_request():
    request_data = request.json
    brad_query = request_data.get("message")
    brad_response = brad.invoke(brad_query)

    agent_response_log = brad.chatlog[list(brad.chatlog.keys())[-1]]
    passed_log_stages = parse_log_for_one_query(agent_response_log)
    
    response_data = {
        "response": brad_response,
        "response-log": passed_log_stages
    }
    return jsonify(response_data)

@app.route("/rag_upload", methods=['POST'])
def upload_file():
    file_list = request.files.getlist("rag_files")
    # Creates new folder with the current statetime
    timestr = time.strftime("%Y%m%d-%H%M%S")
    directory_with_time = os.path.join(app.config['UPLOAD_FOLDER'], timestr) 
    if not os.path.exists(directory_with_time):
        os.makedirs(directory_with_time)

    for file in file_list:
        if file.filename == '':
            response = {"message": "no uploaded file"}
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_location = os.path.join(directory_with_time, filename) 
            file.save(file_location)
            response = {"message": "File uploaded successfully"}

    # print("File uploads done")
    # creating chromadb with uploaded data
    print("running database creation")
    # Count the number of directories in DATABASE_FOLDER
    num_dirs = len([d for d in os.listdir(DATABASE_FOLDER) if os.path.isdir(os.path.join(DATABASE_FOLDER, d))])

    # Create the database with the count included in the dbPath
    db = create_database(
        docsPath=directory_with_time, 
        dbPath=os.path.join(DATABASE_FOLDER, str(num_dirs)),  # Convert the number to a string
        v=True
    )
    print("database created")
    
    brad.state['databases']['RAG'] = db
    
    # print("brad agent database is set")
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
        return message
    
    except FileNotFoundError:
        return jsonify({"error": "Directory not found"})
    
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/remove_session", methods=['POST'])
def remove_open_sessions():
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: October 15, 2024

    # Parse the request data
    request_data = request.json
    session_name = request_data.get("message")  # Get the session name from the request body

    # Log the incoming request
    logger.info(f"Received request to remove session: {session_name}")

    if not session_name:
        logger.error("No session name provided in the request.")
        return jsonify({"success": False, "message": "No session name provided."}), 400

    path_to_output_directories = PATH_TO_OUTPUT_DIRECTORIES

    # Validate the log path
    if not path_to_output_directories:
        logger.error("Log path is not set in the configuration.")
        return jsonify({"success": False, "message": "Log path not configured."}), 500

    session_path = os.path.join(path_to_output_directories, session_name)

    # Check if the session directory exists
    if not os.path.exists(session_path):
        logger.warning(f"Session '{session_name}' does not exist at path: {session_path}")
        return jsonify({"success": False, "message": f"Session '{session_name}' does not exist."}), 404

    # Try to remove the session directory
    try:
        shutil.rmtree(session_path)
        logger.info(f"Successfully removed session: {session_name}")
        return jsonify({"success": True, "message": f"Session '{session_name}' removed."}), 200

    except PermissionError as e:
        logger.error(f"Permission denied while trying to remove session '{session_name}': {str(e)}")
        return jsonify({"success": False, "message": f"Permission denied: {str(e)}"}), 403

    except FileNotFoundError as e:
        logger.error(f"Session '{session_name}' not found during deletion: {str(e)}")
        return jsonify({"success": False, "message": f"Session not found: {str(e)}"}), 404

    except Exception as e:
        logger.error(f"An error occurred while trying to remove session '{session_name}': {str(e)}")
        return jsonify({"success": False, "message": f"Error removing session: {str(e)}"}), 500

@app.route("/change_session", methods=['POST'])
def change_session():
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: October 15, 2024

    request_data = request.json
    print(f"{request_data=}")
    session_name = request_data.get("message")  # Get the session name from the request body
    print(f"{session_name=}")

    # Log the incoming request
    logger.info(f"Received request to change session to: {session_name}")

    if not session_name:
        logger.error("No session name provided in the request.")
        return jsonify({"success": False, "message": "No session name provided."}), 400

    path_to_output_directories = PATH_TO_OUTPUT_DIRECTORIES

    # Validate the log path
    if not path_to_output_directories:
        logger.error("Log path is not set in the configuration.")
        return jsonify({"success": False, "message": "Log path not configured."}), 500

    session_path = os.path.join(path_to_output_directories, session_name)

    # Check if the session directory exists
    if not os.path.exists(session_path):
        logger.warning(f"Session '{session_name}' does not exist at path: {session_path}")
        return jsonify({"success": False, "message": f"Session '{session_name}' does not exist."}), 404

    # Try to remove the session directory
    try:
        brad = Agent(interactive=False,
                     tools=TOOL_MODULES,
                     restart=session_path
                     )
        logger.info(f"Successfully changed to: {session_name}")
        chat_history = brad.get_display()
        chat_history = parse_log_for_process_display(chat_history)
        print("Dump Chat History")
        print(json.dumps(chat_history, indent=4))
        response = jsonify({
            "success": True,
            "message": f"Session '{session_name}' activated.",
            "display": chat_history
            }
        )
        return response, 200

    except PermissionError as e:
        logger.error(f"Permission denied while trying to change session '{session_name}': {str(e)}")
        return jsonify({"success": False, "message": f"Permission denied: {str(e)}"}), 403

    except FileNotFoundError as e:
        logger.error(f"Session '{session_name}' not found during session change: {str(e)}")
        return jsonify({"success": False, "message": f"Session not found: {str(e)}"}), 404

    except Exception as e:
        logger.error(f"An error occurred while trying to change session: '{session_name}': {str(e)}")
        return jsonify({"success": False, "message": f"Error changing session: {str(e)}"}), 500
