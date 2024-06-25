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
from BRAD.planner import *
from BRAD.enrichr import *
from BRAD.scraper import *
from BRAD.router import *
from BRAD.tables import *
from BRAD.rag import *
from BRAD.gene_ontology import *
from BRAD.seabornCaller import *
from BRAD.matlabCaller import *
from BRAD.pythonCaller import *
from BRAD.snakemakeCaller import *
from BRAD.llms import *
from BRAD.geneDatabaseCaller import *
from BRAD.planner import planner
from BRAD.coder import codeCaller
from BRAD.writer import summarizeSteps
from BRAD import log

def getModules():
    """
    Returns a dictionary mapping module names to their corresponding function handles for various tasks.

    :param None: This function does not take any parameters.

    :raises None: This function does not raise any specific errors.

    :return: A dictionary where the keys are module names and the values are function handles for tasks such as querying Enrichr, web scraping, generating seaborn plots, querying documents, and calling Snakemake.
    :rtype: dict

    """
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: June 4, 2024
    
    # this + imports should be the only code someone needs to write to add a new module
    module_functions = {
        'DATABASE'   : geneDBRetriever,     # gget
#        'GGET'   : queryEnrichr,     # gget
#        'DATA'   : manipulateTable,  #
        'SCRAPE' : webScraping,      # webscrapping
        'SNS'    : callSnsV3,        # seaborn
        'RAG'    : queryDocs,        # standard rag
        'MATLAB' : callMatlab,       # matlab
        'PYTHON' : callPython,
        'SNAKE'  : callSnakemake,    # snakemake,
        'PLANNER': planner,
        'CODE'   : codeCaller,
        'WRITE'  : summarizeSteps,
    }
    return module_functions

def load_config():
    """
    Loads the configuration settings from a JSON file.

    :param None: This function does not take any parameters.

    :raises FileNotFoundError: If the configuration file is not found.
    :raises json.JSONDecodeError: If the configuration file contains invalid JSON.

    :return: A dictionary containing the configuration settings.
    :rtype: dict
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
    Saves the configuration settings to a JSON file.

    :param config: A dictionary containing the configuration settings to be saved.
    :type config: dict

    :raises FileNotFoundError: If the directory for the configuration file is not found.
    :raises TypeError: If the configuration dictionary contains non-serializable values.

    :return: None
    :rtype: None

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
    Updates a specific configuration setting based on the given chat status and saves the updated configuration.

    :param chat_status: A dictionary containing the current chat status, including the prompt and configuration.
    :type chat_status: dict

    :raises KeyError: If the specified configuration key is not found in the chat status.
    :raises ValueError: If the value cannot be converted to an integer or float when applicable.

    :return: The updated chat status dictionary.
    :rtype: dict

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
        log.debugLog("Configuration " + str(key) + " updated to " + str(value), chatstatus=chatstatus)
    else:
        log.debugLog("Configuration " + str(key) + " not found", chatstatus=chatstatus)
    return chat_status

def loadChatStatus():
    """
    Initializes and loads the chat status with default values and configuration settings from a file.

    :param None: This function does not take any parameters.

    :raises FileNotFoundError: If the configuration file is not found.
    :raises json.JSONDecodeError: If the configuration file contains invalid JSON.

    :return: A dictionary representing the chat status with initial values and loaded configuration.
    :rtype: dict

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
        'plottingParams'    : {},
        'matlabEng'         : None,
        'experiment'        : False,
        'planned'           : []
    }
    return chatstatus

def load_literature_db(
    persist_directory = "/nfs/turbo/umms-indikar/shared/projects/RAG/databases/Transcription-Factors-5-10-2024/"
):
    """
    Loads a literature database using specified embedding model and settings.

    :param persist_directory: The directory where the database is stored, defaults to "/nfs/turbo/umms-indikar/shared/projects/RAG/databases/Transcription-Factors-5-10-2024/"
    :type persist_directory: str, optional

    :raises FileNotFoundError: If the specified directory does not exist or is inaccessible.
    :raises Warning: If the loaded database contains no articles.

    :return: A tuple containing the vector database and the embeddings model.
    :rtype: tuple

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
    vectordb = remove_repeats(vectordb)
    return vectordb, embeddings_model

def chatbotHelp():
    """
    Displays a help message with information about the RAG chatbot's capabilities and special commands.

    :param None: This function does not take any parameters.

    :raises None: This function does not raise any specific errors.

    :return: None
    :rtype: None

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
    
    chatstatus = log.userOutput(help_message, chatstatus=chatstatus)
    return

def chat(
        model_path = '/nfs/turbo/umms-indikar/shared/projects/RAG/models/llama-2-7b-chat.Q8_0.gguf',
        persist_directory = "/nfs/turbo/umms-indikar/shared/projects/RAG/databases/Transcription-Factors-5-10-2024/",
        llm=None,
        ragvectordb=None,
        embeddings_model=None
    ):
    """
    Initializes and runs the RAG chatbot, allowing interaction with various models and databases, and logs the conversation.

    :param model_path: The path to the Llama model file, defaults to '/nfs/turbo/umms-indikar/shared/projects/RAG/models/llama-2-7b-chat.Q8_0.gguf'.
    :type model_path: str, optional
    :param persist_directory: The directory where the literature database is stored, defaults to "/nfs/turbo/umms-indikar/shared/projects/RAG/databases/Transcription-Factors-5-10-2024/".
    :type persist_directory: str, optional
    :param llm: The language model to be used. If None, it will be loaded within the function.
    :type llm: PreTrainedModel, optional
    :param ragvectordb: The RAG vector database to be used. If None, it will prompt the user to load it.
    :type ragvectordb: Chroma, optional
    :param embeddings_model: The embeddings model to be used. If None, it will be loaded within the function.
    :type embeddings_model: HuggingFaceEmbeddings, optional

    :raises FileNotFoundError: If the specified model or database directories do not exist.
    :raises json.JSONDecodeError: If the configuration file contains invalid JSON.
    :raises KeyError: If required keys are missing from the configuration or chat status.

    :return: None
    :rtype: None

    """
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: June 4, 2024

    config = load_config()
    base_dir = os.path.expanduser('~')
    log_dir = os.path.join(base_dir, config['log_path'])
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    new_dir_name = dt.now().strftime("%Y-%m-%d_%H-%M-%S")
    new_log_dir = os.path.join(log_dir, new_dir_name)
    os.makedirs(new_log_dir)
    chatname = os.path.join(new_log_dir, 'log.json')

    chatstatus = log.userOutput('Welcome to RAG! The chat log from this conversation will be saved to ' + chatname + '. How can I help?', chatstatus=chatstatus)

    # Initialize the dictionaries of tables and databases accessible to BRAD
    databases = {} # a dictionary to look up databases
    tables = {}    # a dictionary to look up tables

    # Initialize the RAG database
    if llm is None:
        llm = load_nvidia()
        # llm = load_llama(model_path) # load the llama
    if ragvectordb is None:
        chatstatus = log.userOutput('\nWould you like to use a database with BRAD [Y/N]?', chatstatus=chatstatus)
        loadDB = input().strip().upper()
        if loadDB == 'Y':
            ragvectordb, embeddings_model = load_literature_db(persist_directory) # load the literature database
        else:
            ragvectordb, embeddings_model = None, None
    
    databases['RAG'] = ragvectordb
    memory = ConversationBufferMemory(ai_prefix="BRAD")

    # Initialize the routers from router.py
    router = getRouter()

    # Initialize the chatlog
    chatstatus        = loadChatStatus()
    chatstatus['llm'] = llm
    chatstatus['memory'] = memory
    chatstatus['databases'] = databases
    chatstatus['output-directory'] = new_log_dir
    chatlog           = {
        'llm'           : str(chatstatus['llm'])
    }
    if chatstatus['config']['experiment']:
        experimentName = os.path.join(log_dir, 'EXP-out-' + str(dt.now()) + '.csv')
        chatstatus['experiment-output'] = '-'.join(experimentName.split())

    # Initialize all modules
    module_functions = getModules()
    
    while True:
        chatstatus = log.userOutput('==================================================', chatstatus=chatstatus)
        if len(chatstatus['planned']) == 0:
            chatstatus['prompt'] = input('Input >> ')                 # get query from user
        else:
            chatstatus['prompt'] = chatstatus['planned'][0]['prompt']
            output_files         = utils.outputFiles(chatstatus)
        
        # Handle explicit commands and routing
        if chatstatus['prompt'] in ['exit', 'quit', 'q', 'bye']:         # check to exit
            break
        elif chatstatus['prompt'].lower() == 'help':              # print man to the screen
            chatbotHelp()
            continue
        # Routing
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
            chatstatus['prompt'] = " ".join(chatstatus['prompt'].split(' ')[2:]).strip()

        # Outputs
        chatstatus = log.userOutput('==================================================', chatstatus=chatstatus)
        chatstatus = log.userOutput('RAG >> ' + str(len(chatlog)) + ': ', chatstatus=chatstatus)
        
        # I wasn't able to add the {end = ''} info to the userOutput function

        # select appropriate routeing function
        if route.upper() in module_functions.keys():
            routeName = route.upper()
        else:
            routeName = 'RAG'
        logging.info(routeName) if chatstatus['config']['debug'] else None

        # get the specific module to use
        module = module_functions[routeName]

        # Query module
        chatstatus['process'] = {'module' : routeName,
                                 'steps'  : []
                            }
        chatstatus = module(chatstatus)

        # Remove the item that was executed. We need must do it after running it for the current file naming system.
        log.debugLog('\n\n\nroute\n\n\n', chatstatus=chatstatus)
        log.debugLog(route, chatstatus=chatstatus)
        if len(chatstatus['planned']) != 0 and route != 'PLANNER':
            log.debugLog(chatstatus['planned'], chatstatus=chatstatus)
            new_output_files = utils.outputFiles(chatstatus)
            new_output_files = list(set(new_output_files).difference(set(output_files)))
            chatstatus = utils.makeNamesConsistent(chatstatus, new_output_files)
            chatstatus['planned'].pop(0)
        
        # Log and reset these values
        chatlog, chatstatus = log.logger(chatlog, chatstatus, chatname)
        
    chatstatus = log.userOutput("Thanks for chatting today! I hope to talk soon, and don't forget that a record of this conversation is available at: " + chatname, chatstatus=chatstatus)
#removes repeat chunks in vectordb
def remove_repeats(vectordb):
    df = pd.DataFrame({'id' :vectordb.get()['ids'] , 'documents' : vectordb.get()['documents']})
    repeated_ids = df[df.duplicated(subset='documents', keep='last')]['id'].tolist()
    if len(repeated_ids) > 0:
        vectordb.delete(repeated_ids)
    return vectordb