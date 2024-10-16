# STANDARD python imports
import os
import json
import shutil
import logging

# Imports for building RESTful API
from flask import Flask, request, jsonify
from flask import flash, redirect, url_for
from werkzeug.utils import secure_filename

# Imports for BRAD library
from BRAD.agent import Agent
from BRAD.rag import create_database
from BRAD import llms # import load_nvidia, load_openai

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# HARDCODED VALUES
UPLOAD_FOLDER = '/usr/src/uploads'
DATABASE_FOLDER = '/usr/src/RAG_Database/'
SOURCE_FOLDER = '/usr/src/brad'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

TOOL_MODULES = ['RAG']

brad = Agent(interactive=False, tools=TOOL_MODULES)
PATH_TO_OUTPUT_DIRECTORIES = brad.state['config'].get('log_path')

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
        response = jsonify({
            "success": True,
            "message": f"Session '{session_name}' activated.",
            "display": brad.get_display()
            }
        )
        logger.info(f"Successfully changed to: {session_name}")
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

@app.route("/set_llm", methods=['POST'])
def set_llm():
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: October 16, 2024

    request_data = request.json
    llm_choice = request_data.get("llm")  # Get the LLM name from the request body
    logger.info(f"Received request to set LLM to: {llm_choice}")

    # Validate the LLM choice
    if not llm_choice:
        logger.error("No LLM choice provided in the request.")
        return jsonify({"success": False, "message": "No LLM choice provided."}), 400

    # Determine which API hosts the selected LLM
    llm_host = 'OPENAI' if 'gpt' in llm_choice.lower() else 'NVIDIA'
    logger.info(f"Using LLM host: {llm_host}")

    try:
        # Load the chosen LLM (e.g., from NVIDIA or OpenAI)
        if llm_host == "NVIDIA":
            print(f"{llm_choice=}")
            llm = llms.load_nvidia(
                model_name = llm_choice,
                temperature = 0,
            )
            print(f"{llm=}")
        elif llm_host == "OPENAI":
            llm = llms.load_openai(
                model_name = llm_choice,
                temperature = 0,
            )
        else:
            logger.error(f"Invalid LLM choice: {llm_choice}")
            return jsonify({"success": False, "message": f"Invalid LLM choice: {llm_choice}"}), 400

        logger.info(f"Sucessfully loaded: {llm_choice} from {llm_host}")

        # Set the LLM in BRAD's status
        brad.set_llm(llm)
        logger.info(f"Successfully set the agent's LLM to: {llm_choice}")

        # Respond with success
        return jsonify({"success": True, "message": f"LLM set to {llm_choice}"}), 200

    except ValueError as e:
        logger.error(f"Invalid LLM choice: {str(e)}")
        return jsonify({"success": False, "message": str(e)}), 400

    except Exception as e:
        logger.error(f"An error occurred while setting LLM to '{llm_choice}': {str(e)}")
        return jsonify({"success": False, "message": f"Error setting LLM: {str(e)}"}), 500

@app.route("/set_llm_api_key", methods=['POST'])
def set_llm_api_key():
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: October 16, 2024

    # TODO: implement logic to allow OpenAI keys to be processed similarly
    # TODO: allow these keys to be written to a file that can be read from later

    request_data = request.json
    print(request_data)
    nvidia_key = request_data.get("nvidia-api-key")  # Get the NVIDIA API key from the request body

    if not nvidia_key:
        logger.error("No NVIDIA API key provided.")
        return jsonify({"message": "NVIDIA API key is required."}), 400  # Return error if no key provided

    # Here, you can add logic to store the API key securely
    # For example, save it to a configuration file or a secure database

    logger.info(f"Received NVIDIA API key: {nvidia_key}")
    
    # Example of saving the key (you might want to implement proper security measures)
    os.environ["NVIDIA_API_KEY"] = nvidia_key

    return jsonify({"message": "NVIDIA API key set successfully."}), 200  # Success response
