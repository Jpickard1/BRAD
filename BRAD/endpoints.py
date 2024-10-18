"""
Endpoints to use BRAD GUI and Server

"""

###############################################################################
#                                  IMPORTS                                    #
###############################################################################


# STANDARD python imports
import os
import json
import shutil
import logging
import time

# Imports for building RESTful API
from flask import Flask, request, jsonify, Blueprint
from flask import flash, redirect, url_for
from werkzeug.utils import secure_filename

# Imports for BRAD library
from BRAD.agent import Agent
from BRAD.rag import create_database
from BRAD import llms # import load_nvidia, load_openai

bp = Blueprint('endpoints', __name__)
brad = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def set_brad_instance(instance):
    '''
    :nodoc:
    '''
    global brad
    brad = instance

UPLOAD_FOLDER = None
DATABASE_FOLDER = None
ALLOWED_EXTENSIONS = None
TOOL_MODULES = None
DATA_FOLDER = None
PATH_TO_OUTPUT_DIRECTORIES = None
def set_globals(data_folder, upload_folder, database_folder, allowed_extensions, tool_modules, path2outputDirs):
    '''
    :nodoc:
    '''
    global UPLOAD_FOLDER, DATABASE_FOLDER, ALLOWED_EXTENSIONS, TOOL_MODULES, DATA_FOLDER, PATH_TO_OUTPUT_DIRECTORIES
    
    # Set the global values
    DATA_FOLDER = upload_folder
    UPLOAD_FOLDER = upload_folder
    DATABASE_FOLDER = database_folder
    ALLOWED_EXTENSIONS = allowed_extensions
    TOOL_MODULES = tool_modules
    PATH_TO_OUTPUT_DIRECTORIES = path2outputDirs

###############################################################################
#                               HELPER METHODS                                #
###############################################################################

def allowed_file(filename):
    '''
    :nodoc:
    '''
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def normalize_keys_upper(d):
    """Recursively normalize dictionary keys to lowercase.

    :nodoc:
    """
    if isinstance(d, dict):
        return {k.upper(): normalize_keys_upper(v) for k, v in d.items()}
    return d


def parse_log_for_one_query(chatlog_query):
    """
    Safely parses a single chat log query for RAG or LLM processes.
    Returns a list of tuples with process steps and relevant sources or chunks.
    
    Args:
        chatlog_query (dict): A single chat query log to parse.
        
    Returns:
        process (list): A list of tuples with parsed information or None if the module is not RAG.

    Expected Pattern:
    
    >>> passed_log_stages = [
    ...   ('RAG-R', ['source 1', 'source 2', 'source 3']),
    ...   ('RAG-G', ['This is chunk 1', 'This is chunk 2', 'This is chunk 3'])
    >>> ]

    :nodoc:
    """

    # Ensure that keys will all be uppercase
    chatlog_query = normalize_keys_upper(chatlog_query)

    # Ensure that 'process' and 'module' keys exist in the query
    process_data = chatlog_query.get('PROCESS', {})
    module_name = process_data.get('MODULE', '').upper()
    
    if module_name == 'RAG':
        process = []
        steps = process_data.get('STEPS', [])
        
        for step in steps:
            # Check if 'func' key exists and is 'rag.retrieval'
            if step.get('func', '').lower() == 'rag.retrieval':
                docs_text = step.get('docs-to-gui', [])
                
                sources = []
                chunks = []
                
                for doc in docs_text:
                    # Safely access 'source' and 'text', provide fallback if missing
                    sources.append(doc.get('source', 'Unknown Source'))
                    chunks.append(doc.get('text', 'Unknown Text'))
                
                # Add the sources and chunks to the process list
                process.append(('RAG-R', sources))
                process.append(('RAG-G', chunks))
            
            # Check if 'purpose' key exists and is 'chat without RAG'
            elif step.get('purpose', '').lower() == 'chat without rag':
                process.append(('LLM-Generation', "Response generated with only the LLM."))
        
        return process
    
    return None


def parse_log_for_process_display(chat_history):
    '''
    :nodoc:
    '''
    for i in range(len(chat_history)):
        if chat_history[i][1] is not None:
            # print('replacing logs')
            # print(f"{chat_history=}")
            chat_history[i] = (chat_history[i][0], parse_log_for_one_query(chat_history[i][1]))
            # print(f"{chat_history=}")
    return chat_history # passed_log_stages

###############################################################################
#                                 ENDPOINTS                                   #
###############################################################################

@bp.route("/invoke", methods=['POST'])
def ep_invoke():
    invoke(request)

def invoke(request):
    """
    Invoke a query using the BRAD agent.

    This function handles an incoming request from the user, extracts the message, and sends it to the BRAD agent for processing. 
    It then returns a JSON response containing both the BRAD agent's reply and the associated log stages.

    **Input Request Structure**:
    The input request should be a JSON object with the following format:
    ```json
    {
        "message": "Your query here"
    }
    ```

    **Output Response Structure**:
    The response will be a JSON object containing the agent's response and a log of the processing stages:
    ```json
    {
        "response": "Generated response from BRAD agent",
        "response-log": {
            "stage_1": "log entry for stage 1",
            "stage_2": "log entry for stage 2",
            ...
        }
    }
    ```

    :param request: A Flask request object containing JSON data with the user message.
    :type request: flask.Request
    :param brad_query: The message to be processed by the BRAD agent.
    :type brad_query: str
    :return: A JSON response containing the agent's reply and the log of query stages.
    :rtype: dict
    """
    global brad
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

@bp.route("/databases/create", methods=['POST'])
def ep_databases_create():
    databases_create(request)

def databases_create(request):
    """
    Upload files and create a retrieval-augmented generation (RAG) database.

    This endpoint allows users to upload multiple files, which are saved to the server. After the files are uploaded, a new folder is created, and the files are used to generate a RAG database.

    **Input Request Structure**:
    The request should include a list of files (for database creation) and a form field specifying the database name:
    - The files are uploaded through the key `"rag_files"`.
    - The database name is provided in the form field `"name"`.

    Example request format:
    ```
    POST /databases_create
    Form data:
    - name: "example_database"
    Files:
    - rag_files: file1.txt
    - rag_files: file2.txt
    ```

    **Output Response Structure**:
    The response will return a JSON object with a message indicating the success or failure of the file uploads:
    ```json
    {
        "message": "File uploaded successfully"
    }
    ```
    If no files were uploaded, the response will indicate an error:
    ```json
    {
        "message": "no uploaded file"
    }
    ```

    :param request: A Flask request object that includes uploaded files and form data.
    :type request: flask.Request
    :param file_list: A list of files uploaded through the request.
    :type file_list: list
    :return: A JSON response indicating the success or failure of the file upload and database creation process.
    :rtype: dict
    """
    file_list = request.files.getlist("rag_files")
    dbName = request.form.get('name')

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
    # num_dirs = len([d for d in os.listdir(DATABASE_FOLDER) if os.path.isdir(os.path.join(DATABASE_FOLDER, d))])

    # Create the database with the count included in the dbPath
    db = create_database(
        docsPath=directory_with_time, 
        dbPath=os.path.join(DATABASE_FOLDER), # str(num_dirs)),  # Convert the number to a string
        dbName=dbName,
        v=True
    )
    print("database created")
    
    brad.state['databases']['RAG'] = db
    
    # print("brad agent database is set")
    return jsonify(response)

@bp.route("/databases/available", methods=['GET'])
def ep_databases_available():
    databases_available(request)

def databases_available():
    """
    Retrieve a list of available retrieval-augmented generation (RAG) databases.

    This endpoint lists all available databases stored in the designated database folder. The function checks the folder for subdirectories, which represent the databases, and returns the list in JSON format. If no databases are found, the response includes "None" as the first entry in the list.

    **Input Request Structure**:
    This is a `GET` request and does not require any parameters.

    Example request:
    ```
    GET /databases/available
    ```

    **Output Response Structure**:
    A JSON object is returned with the list of available databases. In case of errors (e.g., folder not found), an error message is returned.

    Example success response:
    ```json
    {
        "databases": ["None", "database1", "database2"]
    }
    ```

    Example error response (if folder is not found):
    ```json
    {
        "error": "Directory not found"
    }
    ```

    :return: A JSON response containing a list of available databases or an error message.
    :rtype: dict
    """
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: October 17, 2024    
    
    # Get list of directories at this location
    try:
        databases = [name for name in os.listdir(DATABASE_FOLDER) 
                         if os.path.isdir(os.path.join(DATABASE_FOLDER, name))]
        databases.insert(0, "None")

        # Return the list of open sessions as a JSON response
        response = jsonify({"databases": databases})
        return response
    
    except FileNotFoundError:
        return jsonify({"error": "Directory not found"})
    
    except Exception as e:
        return jsonify({"error": str(e)})
    
@bp.route("/databases/set", methods=['POST'])
def ep_databases_set():
    databases_set(request)

def databases_set():
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: October 17, 2024    
    
    # Get list of directories at this location
    try:
        global brad

        request_data = request.json
        logger.info(f"{request_data=}")
        database_name = request_data.get("database")
        logger.info(f"{database_name=}")    
        if database_name == "None":
            brad.state['databases']['RAG'] = None
            logger.info(f"Successfully disconnected RAG database")
        else:
            database_directory = os.path.join(DATABASE_FOLDER, database_name)
            logger.info(f"{database_directory=}")
            db, _ = brad.load_literature_db(persist_directory=database_directory)
            logger.info(f"{db=}")
            # logger.info(f"{len(db.get()['id'])=}")
            brad.state['databases']['RAG'] = db
            logger.info(f"Successfully set the database to: {database_name}")

        # Respond with success
        return jsonify({"success": True, "message": f"Database set to {database_name}"}), 200
   
    except FileNotFoundError:
        return jsonify({"error": "Directory not found"})
    
    except Exception as e:
        return jsonify({"error": str(e)})
    



@bp.route("/open_sessions", methods=['GET'])
def get_open_sessions():
    """
    Retrieve a list of currently open chat sessions.

    This endpoint allows the front end to access previously opened chat sessions.
    It returns a list of directories representing the open sessions.

    :return: A JSON response containing the list of open session names.
    :rtype: dict
    :raises FileNotFoundError: If the directory for session storage is not found.
    :raises Exception: For any other exceptions encountered during execution.
    """
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: October 14, 2024
    
    # Get list of directories at this location
    try:
        open_sessions = [name for name in os.listdir(PATH_TO_OUTPUT_DIRECTORIES) 
                         if os.path.isdir(os.path.join(PATH_TO_OUTPUT_DIRECTORIES, name))]
        
        # Return the list of open sessions as a JSON response
        message = jsonify({"open_sessions": open_sessions})
        return message
    
    except FileNotFoundError:
        return jsonify({"error": "Directory not found"})
    
    except Exception as e:
        return jsonify({"error": str(e)})

@bp.route("/remove_session", methods=['POST'])
def remove_open_sessions():
    """
    Remove a specified open chat session.

    This endpoint allows users to remove a previously opened chat session by its name.
    If the session exists, it will be deleted from the server.

    :param session_name: The name of the session to be removed.
    :type session_name: str
    :return: A JSON response indicating the success or failure of the removal.
    :rtype: dict
    :raises ValueError: If no session name is provided.
    :raises FileNotFoundError: If the session directory does not exist.
    :raises PermissionError: If there are permission issues while deleting the session.
    :raises Exception: For any other exceptions encountered during execution.
    """

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

@bp.route("/create_session", methods=['GET'])
def create_session():
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: October 17, 2024

    # request_data = request.json
    # print(f"{request_data=}")

    # Log the incoming request
    logger.info(f"Received request to create a new session")

    # Try to remove the session directory
    try:
        # Delete the old agent
        global brad
        
        # Delete the old agent
        brad.save_state()
        logger.info(f"Saving state of agent before deleting")
        del brad
        logger.info(f"Removing current agent with `del brad`")

        # Create the new agent
        logger.info(f"Activating new agent")
        # Create the new agent
        brad = Agent(interactive=False,
                     tools=TOOL_MODULES
                     )
        logger.info(f"Agent active at path: {brad.chatname}")
        
        chat_history = brad.get_display()
        logger.info(f"Retrieved chat history")
        chat_history = parse_log_for_process_display(chat_history)
        logger.info(f"Extracted agent history for display:")
        logger.info(json.dumps(chat_history, indent=4))
        response = jsonify({
            "success": True,
            "message": f"New session activated.",
            "display": chat_history
            }
        )
        logger.info(f"Response constructed: {response}")
        return response, 200

    except Exception as e:
        logger.error(f"An error occurred while trying to create a new session': {str(e)}")
        return jsonify({"success": False, "message": f"Error session"}), 500

@bp.route("/change_session", methods=['POST'])
def change_session():
    """
    Change the active session to a specified session name.

    This endpoint allows users to activate a specific session, making it the current working session.
    It returns the display of the newly activated session.

    :param session_name: The name of the session to activate.
    :type session_name: str
    :return: A JSON response indicating success and the display of the activated session.
    :rtype: dict
    :raises ValueError: If no session name is provided.
    :raises FileNotFoundError: If the session directory does not exist.
    :raises PermissionError: If there are permission issues while activating the session.
    :raises Exception: For any other exceptions encountered during execution.
    """
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: October 15, 2024
    global brad
    
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

    global brad

    # Check if trying to change to the active session
    if os.path.join(session_path, 'log.json') == brad.chatname:
        logger.warning(f"Session '{session_name}' does not exist at path: {session_path}")
        return jsonify({"success": False, "message": f"Cannot change to the current session."}), 404
    else:
        logger.info(f"{os.path.join(session_path, 'log.json')}")
        logger.info(f"{brad.chatname}")

    # Try to remove the session directory
    try:
        
        # Delete the old agent
        brad.save_state()
        logger.info(f"Saving state of agent before deleting")
        del brad
        logger.info(f"Removing current agent with `del brad`")

        # Create the new agent
        logger.info(f"Activating agent from: {session_path}")
        brad = Agent(interactive=False,
                     tools=TOOL_MODULES,
                     restart=session_path
                     )
        logger.info(f"Successfully activated agent: {session_name}")
        chat_history = brad.get_display()
        logger.info(f"Retrieved chat history")
        chat_history = parse_log_for_process_display(chat_history)
        logger.info(f"Extracted agent history for display:")
        logger.info(json.dumps(chat_history, indent=4))
        response = jsonify({
            "success": True,
            "message": f"Session '{session_name}' activated.",
            "display": chat_history
            }
        )
        logger.info(f"Response constructed: {response}")
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

@bp.route("/rename_session", methods=['POST'])
def rename_session():
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: October 17, 2024

    # Parse the request data
    request_data = request.json
    session_name = request_data.get("session_name")  # Get the session name from the request body
    updated_name = request_data.get("updated_name")  # Get the session name from the request body

    # Log the incoming request
    logger.info(f"Received request to rename session: {session_name} to {updated_name}")

    # Validate the sent arguments
    if not session_name:
        logger.error("No session_name provided in the request.")
        return jsonify({"success": False, "message": "No session name provided."}), 400
    if not updated_name:
        logger.error("No updated_name provided in the request.")
        return jsonify({"success": False, "message": "No updated name provided."}), 400

    # Validate the log path
    if not PATH_TO_OUTPUT_DIRECTORIES:
        logger.error("Log path is not set in the configuration.")
        return jsonify({"success": False, "message": "Log path not configured."}), 500

    session_path = os.path.join(PATH_TO_OUTPUT_DIRECTORIES, session_name)
    updated_path = os.path.join(PATH_TO_OUTPUT_DIRECTORIES, updated_name)

    # Check if the session directory exists
    if not os.path.exists(session_path):
        logger.warning(f"Session '{session_name}' does not exist at path: {session_path}")
        return jsonify({"success": False, "message": f"Session '{session_name}' does not exist."}), 404

    # Try to rename the session directory
    try:
        global brad

        changeAgent =  brad.chatname != os.path.join(session_path, 'log.json')
        logger.info(f"Change Agent for rename: {changeAgent}")

        # Activate session being renamed if it is not currently active
        if changeAgent:
            # Delete the old agent
            brad.save_state()
            logger.info(f"Saving state of agent before deleting")
            del brad
            logger.info(f"Removing current agent with `del brad`")

            # Create the new agent
            logger.info(f"Activating agent from: {session_path}")
            brad = Agent(interactive=False,
                        tools=TOOL_MODULES,
                        restart=session_path
                        )
            logger.info(f"Successfully activated agent: {session_name}")

        else:
            logger.info(f"Continuing with same agent")

        # Rename the directory
        os.rename(session_path, updated_path)
        logger.info(f"Renamed directories")

        # Rename the chat log location for the agent
        brad.chatname = os.path.join(updated_path, 'log.json')
        logger.info(f"Changed agent output log brad.chatname: {brad.chatname}")

        logger.info(f"Successfully renamed session: {session_name} -> {updated_name}")

        chat_history = brad.get_display()
        chat_history = parse_log_for_process_display(chat_history)
        response = jsonify({
            "success": True,
            "message": f"Session '{session_name}' activated.",
            "display": chat_history
            }
        )
        return response, 200

        # return jsonify({"success": True, "message": f"Session '{session_name}' removed."}), 200

    except PermissionError as e:
        logger.error(f"Permission denied while trying to rename session '{session_name}': {str(e)}")
        return jsonify({"success": False, "message": f"Permission denied: {str(e)}"}), 403

    except FileNotFoundError as e:
        logger.error(f"Session '{session_name}' not found during rename: {str(e)}")
        return jsonify({"success": False, "message": f"Session not found: {str(e)}"}), 404

    except Exception as e:
        logger.error(f"An error occurred while trying to rename session '{session_name}': {str(e)}")
        return jsonify({"success": False, "message": f"Error removing session: {str(e)}"}), 500

@bp.route("/set_llm", methods=['POST'])
def set_llm():
    """
    Set the language model (LLM) for the BRAD agent.

    This endpoint allows users to specify which language model should be used by the BRAD agent.
    It updates the BRAD agent's configuration and responds with the current LLM setting.

    :param model_name: The name of the LLM to set.
    :type model_name: str
    :return: A JSON response indicating success and the name of the active LLM.
    :rtype: dict
    :raises ValueError: If no model name is provided.
    :raises Exception: For any other exceptions encountered during execution.
    """
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

@bp.route("/set_llm_api_key", methods=['POST'])
def set_llm_api_key():
    """
    Set the NVIDIA API key for the BRAD agent.

    This endpoint allows users to provide an NVIDIA API key, which will be stored securely
    for use by the BRAD agent. The key can be used for authentication when accessing NVIDIA services.
    The function currently supports only the NVIDIA API key but may be extended to process other API keys in the future.

    :param request_data: JSON data containing the NVIDIA API key.
    :type request_data: dict
    :param nvidia_key: The NVIDIA API key to be set.
    :type nvidia_key: str
    :return: A JSON response indicating the success or failure of setting the API key.
    :rtype: dict
    :raises ValueError: If no NVIDIA API key is provided.
    :raises Exception: For any other exceptions encountered during the process.
    """

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

