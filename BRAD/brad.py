# Standard
import pandas as pd
import numpy as np
from copy import deepcopy
import os
import sys
from importlib import reload
import textwrap
from scipy import sparse
import importlib
from itertools import product
from datetime import datetime as dt
from IPython.display import display # displaying dataframes
import string
import warnings
import re
import json
import logging

# Bioinformatics
import gget

# Plotting
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# RAG
import chromadb
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.output_parsers import CommaSeparatedListOutputParser
from semantic_router.layer import RouteLayer
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain.memory import ConversationBufferMemory

# Put your modules here:
from BRAD.enrichr import *
from BRAD.scraper import *
from BRAD.router import *
from BRAD.tables import *
from BRAD.rag import *
from BRAD.gene_ontology import *
from BRAD.seabornCaller import *
# from matlabCaller import *
from BRAD.snakemakeCaller import *
from BRAD.llms import *
from BRAD.geneDatabaseCaller import *

def getModules():
    """
    Retrieve a dictionary mapping module names to their corresponding functions.

    This function provides a central place to define and access different modules 
    by their identifiers. The dictionary returned maps module identifiers (keys) 
    to their respective functions (values).

    Returns:
        dict: A dictionary where the keys are module identifiers (str) and the 
              values are function references.
    """
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: June 4, 2024
    
    # this + imports should be the only code someone needs to write to add a new module
    module_functions = {
        'GGET'   : geneDBRetriever,     # gget
#        'GGET'   : queryEnrichr,     # gget
#        'DATA'   : manipulateTable,  #
        'SCRAPE' : webScraping,      # webscrapping
        'SNS'    : callSnsV3,        # seaborn
        'RAG'    : queryDocs,        # standard rag
#        'MATLAB' : callMatlab,       # matlab
        'SNAKE'  : callSnakemake     # snakemake
    }
    return module_functions

def load_config():
    """
    Load configuration from a JSON file.

    This function reads the configuration settings from a JSON file located at 
    'config/config.json' and returns the data as a dictionary.

    Returns:
        dict: The configuration data loaded from the JSON file.
    """
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: June 4, 2024
    
    current_script_path = os.path.abspath(__file__)
    current_script_dir = os.path.dirname(current_script_path)
    file_path = os.path.join(current_script_dir, 'config', 'config.json')
    with open(file_path, 'r') as f:
        return json.load(f)
    
def save_config(config):
    """
    Save configuration to a JSON file.

    This function writes the provided configuration data to a JSON file named 
    'config.json', ensuring the data is formatted with an indentation of 4 spaces.

    Args:
        config (dict): The configuration data to be saved.
    """
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: June 4, 2024
    
    current_script_path = os.path.abspath(__file__)
    current_script_dir = os.path.dirname(current_script_path)
    file_path = os.path.join(current_script_dir, 'config', 'config.json')
    with open(file_path, 'w') as f:
        json.dump(config, f, indent=4)
        
def reconfig(chat_status):
    """
    Update a specific configuration in the chat status based on the provided prompt.

    This function parses a prompt string from the chat status to extract a key-value 
    pair and updates the corresponding configuration in the chat status. The function 
    attempts to convert the value to an integer or float if possible. The updated 
    configuration is then saved to a JSON file.

    Args:
        chat_status (dict): A dictionary containing the current chat status, including
                            the prompt and configuration.

    Returns:
        dict: The updated chat status.
    """
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: June 4, 2024
    
    prompt = chat_status['prompt']
    _, key, value = prompt.split(maxsplit=2)
    try:
        value = int(value)
    except ValueError:
        try:
            value = float(value)
        except ValueError:
            value = str(value)
    if key in chat_status['config']:
        chat_status['config'][key] = value
        save_config(chat_status['config'])
        print("Configuration " + str(key) + " updated to " + str(value))
    else:
        print("Configuration " + str(key) + " not found")
    return chat_status

def loadChatStatus():
    """
    Load the initial chat status.

    This function initializes the chat status with default values, including loading 
    the configuration from a JSON file. The chat status dictionary contains various 
    fields to manage the chat session.

    Returns:
        dict: A dictionary representing the initial chat status with the following keys:
            - config (dict): The configuration settings loaded from the JSON file.
            - prompt (str or None): The current prompt string, initially set to None.
            - output (str or None): The current output string, initially set to None.
            - process (dict): A dictionary to manage ongoing processes, initially empty.
            - current table (dict): A dictionary with keys 'key' and 'tab' representing
                                    the current table key and table object, both initially None.
            - current documents (None): The current documents, initially set to None.
            - tables (dict): A dictionary to store table data, initially empty.
            - documents (dict): A dictionary to store document data, initially empty.
            - plottingParams (dict): A dictionary to store parameters for plotting, initially empty.
    """
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: June 4, 2024
    
    chatstatus = {
        'config'            : load_config(),
        'prompt'            : None,
        'output'            : None,
        'process'           : {},
        'current table'     : {'key':None, 'tab':None},
        'current documents' : None,
        'tables'            : {},
        'documents'         : {},
        'plottingParams'    : {}
    }
    return chatstatus

def load_literature_db(
    persist_directory = "/nfs/turbo/umms-indikar/shared/projects/RAG/databases/Transcription-Factors-5-10-2024/"
):
    """
    Load a literature database with embedding models and persistent client settings.

    This function loads a literature database from a specified directory, using a 
    HuggingFace embedding model and Chroma for vector database management. It checks 
    if the database contains any articles and issues a warning if it is empty.

    Args:
        persist_directory (str): The directory path where the database is stored. 
                                 Defaults to a predefined path for transcription factors.

    Returns:
        tuple: A tuple containing:
            - vectordb (Chroma): The loaded vector database.
            - embeddings_model (HuggingFaceEmbeddings): The embeddings model used 
                                                        for generating embeddings.

    Notes:
        - The function uses the 'BAAI/bge-base-en-v1.5' model from HuggingFace for 
          embedding text.
        - The database is named dynamically based on collection size and overlap 
          parameters.

    Example:
        # Load the literature database from the default directory
        vectordb, embeddings_model = load_literature_db()

        # Load the literature database from a custom directory
        custom_directory = "/path/to/custom/directory/"
        vectordb, embeddings_model = load_literature_db(persist_directory=custom_directory)

    Warnings:
        If the database contains no articles, a warning is issued.

    """
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: June 4, 2024
    
    # load database
    embeddings_model = HuggingFaceEmbeddings(model_name='BAAI/bge-base-en-v1.5')        # Embedding model
    db_name = "DB_cosine_cSize_%d_cOver_%d" % (700, 200)
    _client_settings = chromadb.PersistentClient(path=(persist_directory + db_name))
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings_model, client=_client_settings, collection_name=db_name)
    if len(vectordb.get()['ids']) == 0:
        warnings.warn('The loaded database contains no articles. See the database: ' + str(persist_directory) + ' for details')
    return vectordb, embeddings_model

def is_json_serializable(value):
    """
    Check if a value is JSON serializable.

    This function attempts to serialize a given value to a JSON-formatted string. 
    If the serialization is successful, the function returns True, indicating that 
    the value is JSON serializable. If a TypeError or OverflowError is raised during 
    serialization, the function returns False, indicating that the value is not JSON 
    serializable.

    Args:
        value: The value to be checked for JSON serializability. This can be of any 
               data type.

    Returns:
        bool: True if the value is JSON serializable, False otherwise.

    Example:
        # Check if a dictionary is JSON serializable
        data = {"key": "value"}
        is_serializable = is_json_serializable(data)  # Returns True

        # Check if a complex number is JSON serializable
        complex_num = 1 + 2j
        is_serializable = is_json_serializable(complex_num)  # Returns False
    """
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: June 4, 2024
    
    try:
        json.dumps(value)
        return True
    except (TypeError, OverflowError):
        return False

def logger(chatlog, chatstatus, chatname):
    """
    Log the current chat status and process information to a log file.

    This function updates the chat log with the current status and process 
    information from `chatstatus`. It ensures that all process information is 
    JSON serializable. The updated log is then saved to a specified file.

    Args:
        chatlog (dict): The current log of the chat, containing previous chat 
                        entries.
        chatstatus (dict): The current status of the chat, including prompt, 
                           output, process, and other metadata.
        chatname (str): The name of the file where the updated chat log will be 
                        saved.

    Returns:
        tuple: A tuple containing:
            - chatlog (dict): The updated chat log.
            - chatstatus (dict): The updated chat status with the process 
                                 information reset to None.

    Notes:
        - The `process` information in `chatstatus` is checked for JSON 
          serializability. Any non-serializable values are converted to strings.
        - The current table information is converted to JSON if it is not None.
    """
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: June 4, 2024
    
    process_serializable = {
            key: value if is_json_serializable(value) else str(value)
            for key, value in chatstatus['process'].items()
        }

    chatlog[len(chatlog)] = {
        'prompt' : chatstatus['prompt'],  # the input to the user
        'output' : chatstatus['output'],  # the output to the user
        'process': process_serializable,  # chatstatus['process'], # information about the process that ran
        'status' : {                      # information about the chat at that time
            'databases'         : str(chatstatus['databases']),
            'current table'     : {
                    'key'       : chatstatus['current table']['key'],
                    'tab'       : chatstatus['current table']['tab'].to_json() if chatstatus['current table']['tab'] is not None else None,
                },
            'current documents' : chatstatus['current documents'],
        }
    }
    with open(chatname, 'w') as fp:
        json.dump(chatlog, fp, indent=4)
    chatstatus['process'] = None
    return chatlog, chatstatus

def chatbotHelp():
    """
    Display the help message for the RAG chatbot.

    This function prints a help message that provides an overview of the capabilities 
    of the RAG chatbot, including the various databases it can interact with and 
    special commands available to the user.
    """
    help_message = """
    Welcome to our RAG chatbot Help!
    
    You can chat with the llama-2 llm and several scientifici databases including:
    - literature augmented generation
    - spreadsheet manipulation
    - web scraping
    - enrichr
    - gene ontology

    Special commands:
    /set   - Allows the user to change configuration variables.
    /force - Allows the user to specify which database to use.

    For example:
    /set config_variable_name new_value
    --force option_name

    Enjoy chatting with the chatbot!
    """
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: June 4, 2024
    
    print(help_message)
    return

def chat(
        model_path = '/nfs/turbo/umms-indikar/shared/projects/RAG/models/llama-2-7b-chat.Q8_0.gguf',
        persist_directory = "/nfs/turbo/umms-indikar/shared/projects/RAG/databases/Transcription-Factors-5-10-2024/",
        llm=None,
        ragvectordb=None,
        embeddings_model=None
    ):
    """
    Main function for the RAG chatbot.

    This function initializes and manages the RAG chatbot session. It loads the 
    necessary models and databases, sets up routing for user queries, handles 
    special commands (/set, /force), interacts with the selected modules, logs 
    the chat interactions, and continues the conversation until the user exits.

    Args:
        model_path (str, optional): File path to the llama model. Defaults to 
                                    '/nfs/turbo/umms-indikar/shared/projects/RAG/models/llama-2-7b-chat.Q8_0.gguf'.
        persist_directory (str, optional): Directory path where the literature 
                                           database is stored. Defaults to 
                                           '/nfs/turbo/umms-indikar/shared/projects/RAG/databases/Transcription-Factors-5-10-2024/'.
        llm (LlamaCpp, optional): Preloaded llama model instance. Defaults to None.
        ragvectordb (Chroma, optional): Preloaded literature database instance. 
                                        Defaults to None.
        embeddings_model (HuggingFaceEmbeddings, optional): Preloaded embeddings 
                                                            model instance. Defaults to None.

    Returns:
        None

    Notes:
        - The chat log is saved to a JSON file named based on the current date and 
          time.
        - The function initializes the llama model, the literature database, and 
          necessary configurations.
        - It handles user commands like 'help', '/set', and '/force' to provide 
          assistance, change configurations, or force database usage.
        - The function continues to prompt the user for input until an exit command 
          is received ('exit', 'quit', 'q').

    Example:
        # Run the RAG chatbot session
        main()
    """
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: June 4, 2024
    
    config = load_config()
    base_dir = os.path.expanduser('~')
    log_dir = os.path.join(base_dir, config['log_path'])
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    chatname = os.path.join(log_dir, str(dt.now()) + '.json')
    chatname = '-'.join(chatname.split())
    print('Welcome to RAG! The chat log from this conversation will be saved to ' + chatname + '. How can I help?')
    
    # Initialize the dictionaries of tables and databases accessible to BRAD
    databases = {} # a dictionary to look up databases
    tables = {}    # a dictionary to look up tables
    
    # Initialize the RAG database
    if llm is None:
        llm = load_nvidia()
        # llm = load_llama(model_path) # load the llama
    if ragvectordb is None:
        print('\nWould you like to use a database with BRAD [Y/N]?')
        loadDB = input().strip().upper()
        if loadDB == 'Y':
            ragvectordb, embeddings_model = load_literature_db(persist_directory) # load the literature database
        else:
            ragvectordb, embeddings_model = None, None
    
    databases['RAG'] = ragvectordb
    memory = ConversationBufferMemory(ai_prefix="BRAD")
    # externalDatabases = ['docs', 'GO', 'GGET']
    # retriever = ragvectordb.as_retriever(search_kwargs={"k": 4}) ## Pick top 4 results from the search
    # template = """At the end of each question, print all the genes named in the answer separated by commas.\n Context: {context}\n Question: {question}\n Answer:"""
    # template = """Context: {context}\n Question: {question}\n Answer:"""
    # QA_CHAIN_PROMPT = PromptTemplate(template=template, input_variables=["context" ,"question"],)

    # Initialize the routers from router.py
    router = getRouter()
    
    # Initialize the chatlog
    chatstatus        = loadChatStatus()
    chatstatus['llm'] = llm
    chatstatus['memory'] = memory
    chatstatus['databases'] = databases
    chatlog           = {
        'llm'           : str(chatstatus['llm'])
    }
    
    # Initialize all modules
    module_functions = getModules()
    
    while True:
        print('==================================================')
        chatstatus['prompt'] = input('Input >> ')                 # get query from user
        
        # Handle explicit commands and routing
        if chatstatus['prompt'] in ['exit', 'quit', 'q', 'bye']:         # check to exit
            break
        elif chatstatus['prompt'].lower() == 'help':              # print man to the screen
            chatbotHelp()
            continue
        elif chatstatus['prompt'].startswith('/set'):             # set a configuration variable
            chatstatus = reconfig(chatstatus)
            continue
        elif '/force' not in chatstatus['prompt'].split(' '):     # use the router
            route = router(chatstatus['prompt']).name
            if route is None:
                route = 'RAG'
        else:
            route = chatstatus['prompt'].split(' ')[1]            # use the forced router
            buildRoutes(chatstatus['prompt'])

        print('==================================================')
        print('RAG >> ' + str(len(chatlog)) + ': ', end='')

        # select appropriate routeing function
        if route.upper() in module_functions.keys():
            routeName = route.upper()
        else:
            routeName = 'RAG'
        logging.info(routeName) if chatstatus['config']['debug'] else None

        # get the specific module to use
        module = module_functions[routeName]
        
        # Query database
        chatstatus = module(chatstatus)

        # Log and reset these values
        chatlog, chatstatus = logger(chatlog, chatstatus, chatname)
    print("Thanks for chatting today! I hope to talk soon, and don't forget that a record of this conversation is available at: " + chatname)

