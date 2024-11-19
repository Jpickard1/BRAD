"""
.. warning::

    This is the most active part of the software being developed. It is necessary for the research tool's deployment, but this part
    of the software do not impact the backend `BRAD` python package's utility or function.

The GUI for BRAD uses the python package as a backend and deploys a local server with the following structure:

- **Backend**: A Flask API that handles requests and communicates with the Python-based BRAD package.
- **Frontend**: A React GUI that provides an interactive user interface for sending queries and displaying responses.

.. image:: images/gui-schematic.png
  :scale: 100%
  :alt: Digital Laboratory Vision
  :align: center

React GUI
---------
The React frontend offers a graphical user interface for users to interact with the chatbot. Users can send messages, build RAG databases, or change the system configurations, which the frontend captures and sends to the Flask API. Upon receiving the chatbot's response, the GUI updates the chat interface to display both user and bot messages, facilitating a smooth and engaging conversation experience.


Flask API
---------
The Flask API serves as the backend, enabling communication between the `Agent` logic and the front-end React GUI. This API exposes the following endpoints:

- `sessions`: These endpoints provide information about the open and previously created sessions.
- `databases`: These endpoints allow the user to construct and modify different RAG databases.
- `configure`: This endpoint allows the frontend to reset configuration variables of the `Agent`.
- `llm`: These endpoints allow the frontend to change the LLM of the active `Agent`.
- `invoke`: This endpoint queries the `Agent` class.

The API processes the messages using the logic in the `Agent` class and returns a response to the frontend.

Naming Conventions
~~~~~~~~~~~~~~~~~~

The following conventions are used for writing a new endpoint:

- Select one of the main ednpoints listed above
- Select a secondary name such as: add, set, change, etc. or something descriptive to the endpoint task
- Define a method called `<main endpoint>_<secondary name>` that handles the logic
    - this is where the `Agent` class or server information can directly be manipulated
    - this method has a `request` aregument if it takes parameters (endpoint is `POST`)
    - this method has no parameters otherwise
    - this method requires detailed docstrings for the structure of the request and the response
- Define a method called `ep_<main endpoint>_<secondary name>`
    - accepts no arguments
    - returns `<main endpoint>_<secondary name>`
    - used to mount the logic to the Flast `Blueprint`
    - place this directly above the method `<main endpoint>_<secondary name>`
- Attach this method to the `flask.Blueprint` with the line:
    - `@bp.route("/<main endpoint>/<secondary name>", methods=['POST' or 'GET'])`

Below is an example for adding a generic ednpoint:

    >>> @bp.route("/<main endpoint>/<secondary name>", methods=['POST' or 'GET'])
    >>> def ep_<main endpoint>_<secondary name>():
    >>>     return <main endpoint>_<secondary name>(request)
    >>>     
    >>> def <main endpoint>_<secondary name>(request):
    >>>     try:
    >>>         # TODO: put endpoint logic here
    >>>         response = jsonify(# TODO: put response variables here)
    >>>         return response, 200
    >>>     except:
    >>>         response = jsonify(# TODO: put detailed error message)
    >>>         return response, # TODO put error code


Endpoints
~~~~~~~~~

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
from itertools import filterfalse

# Imports for building RESTful API
from flask import Flask, request, jsonify, Blueprint
from flask import flash, redirect, url_for
from werkzeug.utils import secure_filename

# Used to get list of OpenAI models
from openai import OpenAI

# Imports for BRAD library
from BRAD.agent import Agent, AgentFactory
from BRAD.utils import delete_dirs_without_log 
from BRAD.constants import DEFAULT_SESSION_EXTN
from BRAD.rag import create_database
from BRAD import llms # import load_nvidia, load_openai


bp = Blueprint('endpoints', __name__)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

###############################################################################
#                                  GLOBALS                                    #
###############################################################################


UPLOAD_FOLDER = None
DATABASE_FOLDER = None
ALLOWED_EXTENSIONS = None
TOOL_MODULES = None
DATA_FOLDER = None
CACHE = None
def set_globals(data_folder, upload_folder, database_folder, allowed_extensions, tool_modules, cache):
    '''
    :nodoc:
    '''
    global UPLOAD_FOLDER, DATABASE_FOLDER, ALLOWED_EXTENSIONS, TOOL_MODULES, DATA_FOLDER, CACHE
    
    # Set the global values
    DATA_FOLDER = upload_folder
    UPLOAD_FOLDER = upload_folder
    DATABASE_FOLDER = database_folder
    ALLOWED_EXTENSIONS = allowed_extensions
    TOOL_MODULES = tool_modules
    CACHE = cache

PATH_TO_OUTPUT_DIRECTORIES = None
DEFAULT_SESSION = None
def set_global_output_path(output_path, default_session):
    '''
    :nodoc:
    '''
    global PATH_TO_OUTPUT_DIRECTORIES, DEFAULT_SESSION
    PATH_TO_OUTPUT_DIRECTORIES = output_path
    DEFAULT_SESSION = default_session




###############################################################################
#                               HELPER METHODS                                #
###############################################################################

def initiate_start():
    '''
    Initializer method for important health checks before starting backend
    '''
    initial_agent = AgentFactory(
        tool_modules=TOOL_MODULES, 
        interactive=False,
        persist_directory=DATABASE_FOLDER,
        db_name=CACHE.get('rag_name'),
        gui=True
    ).get_agent()
#    delete_dirs_without_log(initial_agent)
    log_path = initial_agent.state['config'].get('log_path')
    default_session = os.path.join(log_path, DEFAULT_SESSION_EXTN)
    set_global_output_path(log_path, default_session)
    # default agent to be used
    default_agent = AgentFactory(
        tool_modules=TOOL_MODULES, 
        start_path=default_session, 
        interactive=False, 
        persist_directory=DATABASE_FOLDER,
        db_name=CACHE.get('rag_name'),
        gui=True
    ).get_agent()



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
    
    .. note:

        The current positioning of this method will make it challenging for new users to adopt into their own tools.
        It would be structured better if these features were hard coded for how they will come out of the tool.

    Args:
        chatlog_query (dict): A single chat query log to parse.
        
    Returns:
        process (list): A list of tuples with parsed information or None if the module is not RAG.
        llm_usage (dict): A dictionary with information about LLM utilization for the query

    Expected Patterns:
    
    >>> passed_log_stages = [
    >>>   ('RAG-R', ['source 1', 'source 2', 'source 3']),
    >>>   ('RAG-G', ['This is chunk 1', 'This is chunk 2', 'This is chunk 3'])
    >>> ]
    >>> llm_usage = {
    >>>     'llm-calls': (int),
    >>>     'api-fees' : (float)
    >>> }
    """

    # Ensure that keys will all be uppercase
    chatlog_query = normalize_keys_upper(chatlog_query)

    # Ensure that 'process' and 'module' keys exist in the query
    process_data = chatlog_query.get('PROCESS', {})
    module_name = process_data.get('MODULE', '').upper()
    
    llm_usage = {
        'llm-calls': 0,
        'api-fees': 0.0
    }

    if module_name == 'RAG':
        process = []
        process_dict = {}
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

                process_dict['RAG-R'] =  sources
                process_dict['RAG-G'] =  chunks
            
            # Check if 'purpose' key exists and is 'chat without RAG'
            elif step.get('purpose', '').lower() == 'chat without rag':
                process.append(('LLM-Generation', "Response generated with only the LLM."))
                process_dict['LLM-Generation'] = "Response generated with only the LLM."

            # Check if LLM was used in the query
            if 'llm' in step.keys():
                if 'api-info' in step.keys():
                    llm_usage['llm-calls'] += 1
                    llm_usage['api-fees'] += step['api-info']['Total Cost (USD)']
            llm_usage['process'] = process_dict

        return process, llm_usage

    elif module_name == 'SCRAPE':
        return {}, {}
    
    return None, None


def parse_log_for_process_display(chat_history):
    '''
    :nodoc:
    '''
    for i in range(len(chat_history)):
        if chat_history[i][1] is not None:
            # print('replacing logs')
            # print(f"{chat_history=}")
            history_name = chat_history[i][0]
            parsed_log = parse_log_for_one_query(chat_history[i][1])
            chat_history[i] = (history_name, parsed_log)
            # print(f"{chat_history=}")

    return chat_history # passed_log_stages

###############################################################################
#                                 ENDPOINTS                                   #
###############################################################################

@bp.route("/invoke", methods=['POST'])
def ep_invoke():
    return invoke(request)

def invoke(request):
    """
    Invoke a query using the BRAD agent.

    This function handles an incoming request from the user, extracts the message, and sends it to the BRAD agent for processing. 
    It then returns a JSON response containing both the BRAD agent's reply and the associated log stages.

    **Input Request Structure**:
    The input request should be a JSON object with the following format:
    json

    >>> {
    >>>     "message": "Your query here"
    >>> }


    **Output Response Structure**:
    The response will be a JSON object containing the agent's response and a log of the processing stages:

    >>> { 
    >>>      "response": "Generated response from BRAD agent", 
    >>>      "response-log": { 
    >>>          "stage_1": "log entry for stage 1", 
    >>>          "stage_2": "log entry for stage 2", 
    >>>          ...
    >>>      },
    >>>      "llm-usage": {
    >>>          "llm-calls": number of new llm calls,
    >>>          "api-fees":  cost of api fees,
    >>>      }
    >>> }


    :param request: A Flask request object containing JSON data with the user message.
    :type request: flask.Request
    :return: A JSON response containing the agent's reply and the log of query stages.
    :rtype: dict
    """
    request_data = request.json
    brad_session = request_data.get("session", None)
    brad_query = request_data.get("message")
    # session_path = os.path.join(PATH_TO_OUTPUT_DIRECTORIES, brad_session) if brad_session else None
    brad = AgentFactory(
        session_path=brad_session, 
        persist_directory=DATABASE_FOLDER,
        db_name=CACHE.get('rag_name'),
        gui=True
    ).get_agent()

    brad_response = brad.invoke(brad_query)
    brad_name = brad.chatname

    agent_response_log = brad.chatlog[list(brad.chatlog.keys())[-1]]
    passed_log_stages, llm_usage = parse_log_for_one_query(agent_response_log)

    response_data = {
        "response": brad_response,
        "session-name": brad_name,
        "response-log": passed_log_stages,
        "response-log-dict": llm_usage.get('process'),
        "llm-usage": llm_usage
    }
    brad.save_state()
    return jsonify(response_data)

@bp.route("/databases/create", methods=['POST'])
def ep_databases_create():
    return databases_create(request)

def databases_create(request):
    """
    Upload files and create a retrieval-augmented generation (RAG) database.

    This endpoint allows users to upload multiple files, which are saved to the server. After the files are uploaded, a new folder is created, and the files are used to generate a RAG database.

    **Input Request Structure**:
    The request should include a list of files (for database creation) and a form field specifying the database name:

    - The files are uploaded through the key `"rag_files"`.
    - The database name is provided in the form field `"name"`.

    Example request format:

    >>> POST /databases/create
    >>> Form data:
    >>> - name: "example_database"
    >>> Files:
    >>> - rag_files: file1.txt
    >>> - rag_files: file2.txt


    **Output Response Structure**:
    The response will return a JSON object with a message indicating the success or failure of the file uploads:

    >>> { 
    >>>     "message": "File uploaded successfully" 
    >>> } 


    If no files were uploaded, the response will indicate an error:

    >>> { 
    ...     "message": "no uploaded file" 
    >>> }


    :param request: A Flask request object that includes uploaded files and form data.
    :type request: flask.Request
    :param file_list: A list of files uploaded through the request.
    :type file_list: list
    :return: A JSON response indicating the success or failure of the file upload and database creation process.
    :rtype: dict
    """
    brad = AgentFactory(
        session_path=DEFAULT_SESSION, 
        persist_directory=DATABASE_FOLDER,
        db_name=CACHE.get('rag_name'),
        gui=True
    ).get_agent()

    file_list = request.files.getlist("rag_files")
    dbName = request.form.get('name')

    # Creates new folder with the current statetime
    timestr = time.strftime("%Y%m%d-%H%M%S")
    directory_with_time = os.path.join(UPLOAD_FOLDER, timestr) 
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
    return databases_available()

def databases_available():
    """
    Retrieve a list of available retrieval-augmented generation (RAG) databases.

    This endpoint lists all available databases stored in the designated database folder. The function checks the folder for subdirectories, which represent the databases, and returns the list in JSON format. If no databases are found, the response includes "None" as the first entry in the list.

    This is a `GET` request and does not require any parameters.

    Example request:

    >>> GET /databases/available
    
    A JSON object is returned with the list of available databases. In case of errors (e.g., folder not found), an error message is returned.

    Example success response:

    >>> {
    >>>     "databases": ["None", "database1", "database2"]
    >>> }

    Example error response (if folder is not found):

    >>> {
    >>>     "error": "Directory not found"
    >>> }
    
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
    return databases_set(request)

def databases_set(request):
    """
    Set the active retrieval-augmented generation (RAG) database for the BRAD agent.

    This uses the flask system cache to set the active database and updates it.

    This endpoint allows users to select and set an available database from the server. The selected database will be loaded and set as the active RAG database for the BRAD agent. If "None" is selected, it will disconnect the current database.

    **Request Structure**:
    The input should be a JSON object containing the name of the database to be set.

    Example request:

    >>> {
    >>>     "database": "database_name"
    >>> }

    If the database name is `"None"`, the current RAG database will be disconnected.

    **Response Structure**:
    A JSON response is returned indicating whether the database was successfully set or if an error occurred.

    Example success response:

    >>> {
    >>>     "success": True,
    >>>     "message": "Database set to database_name"
    >>> }

    Example response for disconnecting the database:

    >>> {
    >>>     "success": True,
    >>>     "message": "Database set to None"
    >>> }

    Example error response (if the directory is not found):

    >>> {
    >>>     "error": "Directory not found"
    >>> }

    :param request: The HTTP POST request containing the database name in JSON format.
    :type request: flask.Request
    :return: A JSON response with a success message or an error message.
    :rtype: dict
    """
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: October 17, 2024
    
    # Get list of directories at this location

    brad = AgentFactory(
        session_path=DEFAULT_SESSION, 
        persist_directory=DATABASE_FOLDER,
        db_name=CACHE.get('rag_name'),
        gui=True
    ).get_agent()
    try:

        request_data = request.json
        logger.info(f"{request_data=}")
        database_name = request_data.get("database")
        logger.info(f"{database_name=}")    

        rag_database = CACHE.get('rag_name')
        if rag_database != database_name:
            CACHE.set('rag_name', database_name, timeout=0)
        logger.info(f"{database_name=}")    

        if database_name == "None":
            brad.state['databases']['RAG'] = None
            logger.info(f"Successfully disconnected RAG database")
        else:
            database_directory = os.path.join(DATABASE_FOLDER, database_name)
            logger.info(f"{database_directory=}")
            db, _ = brad.load_literature_db(persist_directory=DATABASE_FOLDER, db_name=database_name)
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


@bp.route("/sessions/open", methods=['GET'])
def ep_sessions_open():
    return sessions_open()

def sessions_open():
    """
    Retrieve a list of currently open chat sessions.

    This endpoint allows the front end to access previously opened chat sessions.
    It returns a list of directories representing the open sessions.

    Request:

    >>> GET /sessions/open

    Successful response example:
        >>> {
        >>>     "open_sessions": ["session_1", "session_2", "session_3"]
        >>> }

    Error response example:
        >>> {
        ...     "error": "Directory not found"
        >>> }

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
        open_sessions = sorted([name for name in os.listdir(PATH_TO_OUTPUT_DIRECTORIES) 
                         if os.path.isdir(os.path.join(PATH_TO_OUTPUT_DIRECTORIES, name))], reverse=True)
        
        # Return the list of open sessions as a JSON response
        message = jsonify({"open_sessions": open_sessions})
        return message
    
    except FileNotFoundError:
        return jsonify({"error": "Directory not found"})
    
    except Exception as e:
        return jsonify({"error": str(e)})

@bp.route("/sessions/remove", methods=['POST'])
def ep_sessions_remove():
    return sessions_remove(request)

def sessions_remove(request):
    """
    Remove a specified open chat session.

    This endpoint allows users to remove a previously opened chat session by its name.
    If the session exists, it will be deleted from the server.

    Example request:
        >>> POST /sessions/remove
        >>> {
        >>>   "session_name": "my_chat_session"
        >>> }

    On success, the response will contain:
        >>> {
        >>>   "success": true,
        >>>   "message": "Session 'my_chat_session' removed."
        >>> }

    On failure (e.g., session does not exist), the response will contain:
        >>> {
        >>>   "success": false,
        >>>   "message": "Session 'my_chat_session' does not exist."
        >>> }


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

@bp.route("/sessions/create", methods=['GET'])
def ep_sessions_create():
    return sessions_create()

def sessions_create():
    """
    Create a new chat session by resetting the current BRAD agent.

    This function handles the creation of a new chat session by first saving the state of the current agent, 
    deleting it, and then activating a new agent. The new session is initialized with tools specified 
    in the global `TOOL_MODULES`. The function returns a JSON response indicating the success or failure of the operation.

    **Process**:

    1. Save the state of the current BRAD agent.
    2. Delete the existing agent.
    3. Instantiate a new BRAD agent with the specified tools.
    4. Retrieve and display the chat history for the new session.

    **Request**:
        >>> GET /sessions/create`

    **Response**:
    A JSON response will be returned with the following structure:

    Successful response example: 
        >>> {
        >>>     "success": True,
        >>>     "message": "New session activated.",
        >>>     "display": {
        >>>         "history": "Extracted history logs for display"
        >>>     }
        >>> }

    Error response example:
        >>> {
        >>>     "success": False,
        >>>     "message": "Error session"
        >>> }

    :return: A JSON response indicating whether the session creation was successful or not.
    :rtype: tuple (flask.Response, int)
    :raises Exception: For any general errors encountered during the creation of the session.
    """
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: October 17, 2024

    # Log the incoming request
    logger.info(f"Received request to create a new session")
    # Create the new agent
    logger.info(f"Activating agent")
    brad = AgentFactory(
        persist_directory=DATABASE_FOLDER,
        db_name=CACHE.get('rag_name'),
        gui=True
    ).get_agent()

    # Create the new agent
    logger.info(f"Agent active at path: {brad.chatname}")

    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Define source and destination paths
    source_path = os.path.join(script_dir, 'config', 'config.json')

    # brad.chatname ends with 'log.json' which is removed with [:-8]
    destination_path = os.path.join(brad.chatname[:-8], 'config.json')

    # Copy the file
    try:
        shutil.copy(source_path, destination_path)
        print(f"Configuration file copied to: {destination_path}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")


    # Try to remove the session directory
    try:
        # Delete the old agent
        # brad.save_state()
        logger.info(f"Saving state of agent")
        
        chat_history = brad.get_display()
        logger.info(f"Retrieved chat history")
        chat_history = parse_log_for_process_display(chat_history)
        logger.info(f"Extracted agent history for display:")
        logger.info(json.dumps(chat_history, indent=4))
        brad_name = brad.chatname
        response = jsonify({
            "success": True,
            "session-name": brad_name,
            "message": f"New session activated.",
            "display": [], # chat_history
            }
        )
        logger.info(f"Response constructed: {response}")
        return response, 200

    except Exception as e:
        logger.error(f"An error occurred while trying to create a new session': {str(e)}")
        return jsonify({"success": False, "message": f"Error session"}), 500

@bp.route("/sessions/change", methods=['POST'])
def ep_sessions_change():
    return sessions_change(request)

def sessions_change(request):
    """
    Change the active session to a specified session name.

    This endpoint allows users to activate a specific session, making it the current working session.
    The session change involves saving the state of the current session, deleting the current agent, 
    and activating the new session by loading its logs. The function returns the chat history display 
    of the newly activated session.

    **Request**:
    The request must be a POST request with a JSON body containing the session name.

    Example request:
        >>> {
        >>>     "message": "desired_session_name"
        >>> }

    **Response**:
    A JSON response will be returned with the following structure:

    Successful response example:
        >>> {
        >>>     "success": True,
        >>>     "message": "Session 'desired_session_name' activated.",
        >>>     "display": {
        >>>         "history": "Extracted chat history for the activated session"
        >>>     }
        >>> }

    Error response example (when session is not found):
        >>> {
        >>>     "success": False,
        >>>     "message": "Session 'desired_session_name' does not exist."
        >>> }

    Error response example (permission error):
        >>> {
        >>>     "success": False,
        >>>     "message": "Permission denied: PermissionError message"
        >>> }

    **Exceptions**:
    - **ValueError**: If no session name is provided in the request.
    - **FileNotFoundError**: If the specified session directory does not exist.
    - **PermissionError**: If there are permission issues while accessing or activating the session.
    - **Exception**: For any other general errors during execution.

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
    brad = AgentFactory(
        interactive=False,
        tool_modules=TOOL_MODULES,
        session_path=session_path,
        persist_directory=DATABASE_FOLDER,
        db_name=CACHE.get('rag_name'),
        gui=True
    ).get_agent()

    # Check if the session directory exists
    if not os.path.exists(session_path):
        logger.warning(f"Session '{session_name}' does not exist at path: {session_path}")
        return jsonify({"success": False, "message": f"Session '{session_name}' does not exist."}), 404

    # Check if trying to change to the active session
    if os.path.join(session_path, '/log.json') == brad.chatname:
        logger.warning(f"Session '{session_name}' does not exist at path: {session_path}")
        return jsonify({"success": False, "message": f"Cannot change to the current session."}), 404
    else:
        logger.info(f"{os.path.join(session_path, 'log.json')}")
        logger.info(f"{brad.chatname}")

    # Try to remove the session directory
    try:
        
        # save agent state
        brad.save_state()
        logger.info(f"Saving state of agent")

        chat_history = brad.get_display()
        logger.info(f"Retrieved chat history")
        chat_history = parse_log_for_process_display(chat_history)
        logger.info(f"Extracted agent history for display:")
        logger.info(json.dumps(chat_history, indent=4))
        brad_name = brad.chatname
        data = {
            "success": True,
            "session-name": brad_name,
            "message": f"Session '{session_name}' activated.",
            "display": chat_history
        }
        response = jsonify(
            data
        )
        logger.info(f"Response constructed: {data}")
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

@bp.route("/sessions/rename", methods=['POST'])
def ep_sessions_rename():
    return sessions_rename(request)

def sessions_rename(request):
    """
    Rename an existing session to a new session name.

    This function allows users to rename a session by updating the session's directory and the agent's chat log location.
    If the session to be renamed is not the currently active session, it activates the session first, then proceeds
    with renaming it. The chat history of the renamed session is returned upon success.

    **Request**:
    The request must be a POST request with a JSON body containing the session's current name and the desired updated name.

    Example request:
        >>> {
        >>>     "session_name": "old_session_name",
        >>>     "updated_name": "new_session_name"
        >>> }

    **Response**:
    A JSON response will be returned with the following structure:

    Successful response example:
        >>> {
        >>>     "success": True,
        >>>     "message": "Session 'old_session_name' renamed to 'new_session_name'.",
        >>>     "display": {
        >>>         "history": "Chat history for the renamed session"
        >>>     }
        >>> }

    Error response example (when session does not exist):
        >>> {
        >>>     "success": False,
        >>>     "message": "Session 'old_session_name' does not exist."
        >>> }

    **Exceptions**:
    - **ValueError**: If the current or updated session name is not provided in the request.
    - **FileNotFoundError**: If the specified session directory does not exist.
    - **PermissionError**: If there are permission issues while renaming the session.
    - **Exception**: For any other general errors during execution.

    :return: A JSON response indicating success or failure, along with the chat history of the renamed session.
    :rtype: tuple (flask.Response, int)
    """
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
    # Create the new agent
    logger.info(f"Activating agent from: {session_path}")
    brad = AgentFactory(
        interactive=False,
        tool_modules=TOOL_MODULES,
        session_path=session_path,
        persist_directory=DATABASE_FOLDER,
        db_name=CACHE.get('rag_name'),
        gui=True
    ).get_agent()
    logger.info(f"Successfully activated agent: {session_name}")

    # Check if the session directory exists
    if not os.path.exists(session_path):
        logger.warning(f"Session '{session_name}' does not exist at path: {session_path}")
        return jsonify({"success": False, "message": f"Session '{session_name}' does not exist."}), 404

    # Try to rename the session directory
    try:

        changeAgent =  brad.chatname != os.path.join(session_path, 'log.json')
        logger.info(f"Change Agent for rename: {changeAgent}")

        # Activate session being renamed if it is not currently active
        if changeAgent:
            # Delete the old agent
            brad.save_state()
            logger.info(f"Saving state of agent")

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

@bp.route("/llm/get", methods=['GET'])
def ep_llm_get():
    return llm_get()

def llm_get():
    """
    Get the available OpenAI LLM models.

    This endpoint returns a list of possible LLM models that can be used.

    Request Structure:
    The request must contain a JSON body with the following fields:

        >>> GET /llm/get

    Successful response example:
    
        >>> {
        >>>     "success": true,
        >>>     "models": [
        >>>         "model name 1",
        >>>         "model name 1",
        >>>         ...
        >>>     ]
        >>> }

    :return: A JSON response indicating success and the name of the active LLM.
    :rtype: dict
    """
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: October 20, 2024
    try:
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        models = []
        for model in client.models.list():
            models.append(model.id)
        response = jsonify({"success": True, "models": models})
        return response, 200
    except:
        response = jsonify({
            "success": False, 
            "message": "Unknown error incountered by /llms/get/"
        })
        return response, 500


@bp.route("/llm/set", methods=['POST'])
def ep_llm_set():
    return llm_set(request)

def llm_set(request):
    """
    Set the language model (LLM) for the BRAD agent.

    This endpoint allows users to specify which language model should be used by the BRAD agent.
    It updates the BRAD agent's configuration and responds with the current LLM setting.

    Request Structure:
    The request must contain a JSON body with the following fields:

        >>> {
        >>>   "llm": "str"  # The name of the LLM to set (e.g., "gpt-4", "bloom")
        >>> }

    - llm (str): The name of the LLM to be used (Required).

    Successful response example:
    
        >>> {
        >>>   "success": true,
        >>>   "message": "LLM set to <llm_choice>"
        >>> }

    Error response example:
    
        >>> {
        >>>   "success": false,
        >>>   "message": "Error message describing the failure if the LLM is missing or invalid"
        >>> }

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

@bp.route("/llm/apikey", methods=['POST'])
def ep_llm_apikey():
    return llm_apikey(request)

def llm_apikey(request):
    """
    Set the NVIDIA API key for the BRAD agent.

    This endpoint allows users to provide an NVIDIA API key, which will be stored securely
    for use by the BRAD agent. The key can be used for authentication when accessing NVIDIA services.
    The function currently supports only the NVIDIA API key but may be extended to process other API keys in the future.

    **Request Structure**:
    The request must contain a JSON body with the following fields:

        >>> {
        >>>   "nvidia-api-key": "str"  # The NVIDIA API key to be set
        >>> }

    - nvidia-api-key (str): The NVIDIA API key to be set (Required).

    **Response Structure**:
    On success, the response will contain:

        >>> {
        >>>   "message": "NVIDIA API key set successfully."
        >>> }

    On failure (missing API key), the response will contain:

        >>> {
        >>>   "message": "NVIDIA API key is required."
        >>> }

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

