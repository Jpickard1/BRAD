"""
This module provides a framework for a chatbot that interacts with various language models, databases, and software. These include literature-augmented generation, spreadsheet manipulation, web scraping, and more. The chatbot uses a combination of natural language processing models and specific task-oriented modules to handle user queries and provide relevant responses.

To integrate various capabilities with `brad.py`, three features are used:

    1. Semantic Routing: fast routing is used determine which module should process each user query.

    2. Unified chat status: the `chatstatus` variable is used to manage language models, memory, debugging, and other configuration settings for different modules.

    3. Standardized file sharing: the `utils.py` module standardizes how all data and new files are managed.

"""
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

from typing import Optional, List

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
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM

# Put your modules here:
from BRAD.planner import *
from BRAD.enrichr import *
from BRAD.scraper import *
from BRAD.router import *
from BRAD.tables import *
from BRAD.rag import queryDocs, remove_repeats
from BRAD.gene_ontology import *
from BRAD.seabornCaller import *
# from BRAD.matlabCaller import *
from BRAD.pythonCaller import *
from BRAD.snakemakeCaller import *
from BRAD.llms import *
from BRAD.geneDatabaseCaller import *
from BRAD.planner import planner
from BRAD.coder import codeCaller
from BRAD.writer import summarizeSteps, chatReport
from BRAD import log
from BRAD.bradllm import BradLLM

class chatbot():

    def __init__(self,
        model_path = '/nfs/turbo/umms-indikar/shared/projects/RAG/models/llama-2-7b-chat.Q8_0.gguf',
        persist_directory = "/nfs/turbo/umms-indikar/shared/projects/RAG/databases/Transcription-Factors-5-10-2024/",
        llm=None,
        ragvectordb=None,
        embeddings_model=None,
        restart=None,
        name='BRAD'
    ):
        """
        Initializes and runs the chatbot.
    
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

        # Dev. Comments:
        # -------------------
        # This function initializes a chat session and the chatstatus variable
        #
        # History:
        # - 2024-06-04: wrote 1st draft of this code in the brad.chat() method
        # - 2024-07-10: refactored brad.py file to a class and converted the code
        #               used to initialize the chat session to initialize this class
        # Issues:
        # - We should change the structure of the classes/modules. In this 1st iteration
        #   chatstatus was packed as a class variable and used similar to before, but it
        #   is likely reasonable to split this variable up into separate class variables.
        # super().__init__()
        self.chatstatus = self.loadChatStatus()
        self.name       = name.strip()
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
        base_dir = os.path.expanduser('~')
        log_dir = os.path.join(base_dir, self.chatstatus['config']['log_path'])
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        if restart is None:
            new_dir_name = dt.now().strftime("%Y-%m-%d_%H-%M-%S")
            new_log_dir = os.path.join(log_dir, new_dir_name)
            os.makedirs(new_log_dir)
            self.chatname = os.path.join(new_log_dir, 'log.json')
            self.chatlog  = {}
        else:
            new_log_dir = restart
            self.chatname = os.path.join(restart, 'log.json')
            self.chatlog  = json.load(open(chatname))
    
        # Initialize the dictionaries of tables and databases accessible to BRAD
        databases = {} # a dictionary to look up databases
        tables = {}    # a dictionary to look up tables
    
        # Initialize the RAG database
        if llm is None:
            llm = load_nvidia()
            # llm = load_llama(model_path) # load the llama
            #llm = load_openai()
        if ragvectordb is None:
            chatstatus = log.userOutput('\nWould you like to use a database with ' + self.name + ' [Y/N]?', chatstatus=self.chatstatus)
            loadDB = input().strip().upper()
            if loadDB == 'Y':
                ragvectordb, embeddings_model = self.load_literature_db(persist_directory) # load the literature database
            else:
                ragvectordb, embeddings_model = None, None
        
        databases['RAG'] = ragvectordb
        memory = ConversationBufferMemory(ai_prefix="BRAD")
    
        # Initialize the routers from router.py
        self.router = getRouter()
    
        # Add other information to chatstatus
        self.chatstatus['llm'] = llm
        self.chatstatus['memory'] = memory
        self.chatstatus['databases'] = databases
        self.chatstatus['output-directory'] = new_log_dir
        if self.chatstatus['config']['experiment']:
            self.experimentName = os.path.join(log_dir, 'EXP-out-' + str(dt.now()) + '.csv')
            self.chatstatus['experiment-output'] = '-'.join(experimentName.split())
    
        # Initialize all modules
        self.module_functions = self.getModules()
    
        # Start loop
        self.chatstatus = log.userOutput('Welcome to RAG! The chat log from this conversation will be saved to ' + self.chatname + '. How can I help?', chatstatus=self.chatstatus)

    def invoke(self, query):
        """
        This function executes a single query with the chatbot. It is named so that it is used similar to how we invoke or call llms.
        """
        # Auth: Joshua Pickard
        #       jpic@umich.edu
        # Date: June 4, 2024

        # Dev. Comments:
        # -------------------
        # This function executes a single user prompt with BRAD
        #
        # History:
        # - 2024-06-04: wrote 1st draft of this code in the brad.chat() method
        # - 2024-07-10: refactored brad.py file to a class and converted the code
        #               used to execute a single prompt into the invoke method
        
        # This line packs a query into the chatstatus variable we were using previously
        self.chatstatus['prompt'] = query
        
        # Handle explicit commands and routing
        if self.chatstatus['prompt'].lower() in ['exit', 'quit', 'q', 'bye']:         # check to exit
            return False
        elif self.chatstatus['prompt'].lower() == 'help':              # print man to the screen
            self.chatbotHelp()
            return True
            # continue
        # Routing
        elif self.chatstatus['prompt'].startswith('/set'):             # set a configuration variable
            self.chatstatus = self.reconfig()
            return True
            # continue
        elif '/force' not in self.chatstatus['prompt'].split(' '):     # use the router
            route = self.router(self.chatstatus['prompt']).name
            if route is None:
                route = 'RAG'
        else:
            route = self.chatstatus['prompt'].split(' ')[1]            # use the forced router
            buildRoutes(self.chatstatus['prompt'])
            self.chatstatus['prompt'] = " ".join(self.chatstatus['prompt'].split(' ')[2:]).strip()

        # Outputs
        self.chatstatus = log.userOutput('==================================================', chatstatus=self.chatstatus)
        self.chatstatus = log.userOutput(self.name + ' >> ' + str(len(self.chatlog)) + ': ', chatstatus=self.chatstatus)

        # select appropriate routeing function
        if route.upper() in self.module_functions.keys():
            routeName = route.upper()
        else:
            routeName = 'RAG'

        log.debugLog(routeName, chatstatus=self.chatstatus)

        # get the specific module to use
        module = self.module_functions[routeName]

        # Standardize chatstatus/logging schema for all modules
        #     see: https://github.com/Jpickard1/RAG-DEV/wiki/Logging
        self.chatstatus['process'] = {'module' : routeName,
                                 'steps'  : []
                            }
        self.chatstatus['output'] = ""
        
        # Query module
        self.chatstatus = module(self.chatstatus)

        # Remove the item that was executed. We need must do it after running it for the current file naming system.
        log.debugLog('\n\n\nroute\n\n\n', chatstatus=self.chatstatus)
        log.debugLog(route, chatstatus=self.chatstatus)
        if len(self.chatstatus['queue']) != 0 and route != 'PLANNER':
            log.debugLog(self.chatstatus['queue'], chatstatus=self.chatstatus)
            new_output_files = utils.outputFiles(self.chatstatus)
            new_output_files = list(set(new_output_files).difference(set(output_files)))
            self.chatstatus = utils.makeNamesConsistent(chatstatus, new_output_files)
            if self.chatstatus['process']['module'] != 'ROUTER':
                self.chatstatus['queue pointer'] += 1
        
        # Log and reset these values
        self.chatlog, self.chatstatus = log.logger(self.chatlog, self.chatstatus, self.chatname)
        return True

    def chat(self):
        """This opens a chat session where a user can execute a series of prompts"""
        # Auth: Joshua Pickard
        #       jpic@umich.edu
        # Date: June 4, 2024

        # Dev. Comments:
        # -------------------
        # This function executes a series of prompts in an interactive mannor with BRAD
        #
        # History:
        # - 2024-06-04: wrote 1st draft of this code in the brad.chat() method
        # - 2024-07-10: refactored brad.py file to a class and converted the code
        #               used to execute consecutive prompts into this function
        while True:
            print('==================================================')
            if len(self.chatstatus['queue']) != 0 and self.chatstatus['queue pointer'] < len(self.chatstatus['queue']):
                query = self.chatstatus['queue'][self.chatstatus['queue pointer']]['prompt']
            else:
                query = input('Input >> ')  # get query from user
            
            if not self.invoke(query):
                break
        self.chatstatus = log.userOutput("Thanks for chatting today! I hope to talk soon, and don't forget that a record of this conversation is available at: " + self.chatname, chatstatus=self.chatstatus)

    def getModules(self):
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
            'WRITE'  : chatReport, # summarizeSteps,
            'ROUTER' : reroute,
        }
        return module_functions

    def load_config(self):
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
    
    def save_config(self):
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
            json.dump(self.chatstatus['config'], f, indent=4)

    def reconfig(self):
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
        
        prompt = self.chatstatus['prompt']
        _, key, value = prompt.split(maxsplit=2)
        if key == 'debug':
            value = (value.lower() == 'true')
        try:
            value = int(value)
        except ValueError:
            try:
                value = float(value)
            except ValueError:
                value = str(value)
        if key in self.chatstatus['config']:
            chatstatus['config'][key] = value
            self.save_config()
            chatstatus = log.userOutput("Configuration " + str(key) + " updated to " + str(value), chatstatus=self.chatstatus)
        else:
            chatstatus = log.userOutput("Configuration " + str(key) + " not found", chatstatus=self.chatstatus)
        return chatstatus

    def loadChatStatus(self):
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
            'config'            : self.load_config(),
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
            'queue'             : [],
            'queue pointer'     : 0,
        }
        return chatstatus

    def load_literature_db(
        self,
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

    def chatbotHelp(self):
        """
        Displays a help message with information about the RAG chatbot's capabilities and special commands.
    
        :param None: This function does not take any parameters.
    
        :raises None: This function does not raise any specific errors.
    
        :return: None
        :rtype: None
        """
        # Auth: Joshua Pickard
        #       jpic@umich.edu
        # Date: May 19, 2024
        help_message = """
        Welcome to our RAG chatbot Help!
        
        You can chat with the llama-2 llm and several scientifici databases including:
        - retrieval augmented generation
        - web scraping
        - search enrichr and gene ontology databases
        - run codes
        - and more!
    
        Special commands:
        /set   - Allows the user to change configuration variables.
        /force - Allows the user to specify which database to use.
    
        For example:
        /set config_variable_name new_value
        --force option_name
    
        Enjoy chatting with the chatbot!
        """
        self.chatstatus = log.userOutput(help_message, chatstatus=self.chatstatus)
        return

    def to_langchain(self):
        """
        This function constructs an object that may be used as an llm in langchain.
        """
        # Auth: Joshua Pickard
        #       jpic@umich.edu
        # Date: July 10, 2024
        llm = BradLLM(bot=self)
        return llm

def chat(
        model_path = '/nfs/turbo/umms-indikar/shared/projects/RAG/models/llama-2-7b-chat.Q8_0.gguf',
        persist_directory = "/nfs/turbo/umms-indikar/shared/projects/RAG/databases/Transcription-Factors-5-10-2024/",
        llm=None,
        ragvectordb=None,
        embeddings_model=None,
        restart=None,
        name='BRAD'
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

    # Dev. Comments:
    # -------------------
    # This function executes a chat session
    #
    # History:
    # - 2024-06-04: wrote 1st draft of this code to execute a series of prompts
    #               or a conversation between the user and BRAD
    #
    #       ...     modified this funciton adding new features, logging, routing
    #               and more
    #
    # - 2024-07-10: refactored brad.py file to a class and split this method
    #               into the chatbot.__init__(), chatbot.invoke(), and
    #               chatbot.chat() methods. This method was maintained so that
    #               a user can still fun `brad.chat()` easily and without knowledge
    #               of the class structure

    bot = chatbot(
        model_path = '/nfs/turbo/umms-indikar/shared/projects/RAG/models/llama-2-7b-chat.Q8_0.gguf',
        persist_directory = "/nfs/turbo/umms-indikar/shared/projects/RAG/databases/Transcription-Factors-5-10-2024/",
        llm=None,
        ragvectordb=None,
        embeddings_model=None,
        restart=None,
        name='BRAD'
    )
    bot.chat()
