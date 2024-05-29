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

# Put your modules here:
from enrichr import *
from scraper import *
from router import *
from tables import *
from RAG import *
from gene_ontology import *

def helloWorld():
    print('hi')

def load_config():
    file_path = 'config.json'
    with open(file_path, 'r') as f:
        return json.load(f)

def save_config(config):
    file_path = 'config.json'
    with open(file_path, 'w') as f:
        json.dump(config, f, indent=4)

def reconfig(chat_status):
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

def load_llama(model_path = '/nfs/turbo/umms-indikar/shared/projects/RAG/models/llama-2-7b-chat.Q8_0.gguf'):
    # load llama model
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    llm = LlamaCpp(model_path=model_path, n_ctx = 4098, max_tokens = 1000, callback_manager=callback_manager, verbose=True)
    return llm, callback_manager

def load_literature_db(persist_directory = "/nfs/turbo/umms-indikar/shared/projects/RAG/databases/Transcription-Factors-5-10-2024/"):
    # load database
    embeddings_model = HuggingFaceEmbeddings(model_name='BAAI/bge-base-en-v1.5')        # Embedding model
    db_name = "DB_cosine_cSize_%d_cOver_%d" % (700, 200)
    _client_settings = chromadb.PersistentClient(path=(persist_directory + db_name))
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings_model, client=_client_settings, collection_name=db_name)
    if len(vectordb.get()['ids']) == 0:
        warnings.warn('The loaded database contains no articles. See the database: ' + str(persist_directory) + ' for details')
    return vectordb, embeddings_model

def is_json_serializable(value):
    try:
        json.dumps(value)
        return True
    except (TypeError, OverflowError):
        return False

def logger(chatlog, chatstatus, chatname):
    # print(chatname)
    # print(chatlog)
    process_serializable = {
            key: value if is_json_serializable(value) else str(value)
            for key, value in chatstatus['process'].items()
        }

    chatlog[len(chatlog)] = {
        'prompt' : chatstatus['prompt'],  # the input to the user
        'output' : chatstatus['output'],  # the output to the user
        'process': process_serializable,  #chatstatus['process'], # information about the process that ran
        'status' : {                      # information about the chat at that time
            'databases'         : str(chatstatus['databases']),
            'current table'     : {
                    'key'       : chatstatus['current table']['key'],
                    'tab'       : chatstatus['current table']['tab'].to_json() if chatstatus['current table']['tab'] is not None else None,
                },
            'current documents' : chatstatus['current documents'],
        }
    }
    # print(chatlog)
    with open(chatname, 'w') as fp:
        json.dump(chatlog, fp, indent=4)
    return chatlog

def chatbotHelp():
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
    print(help_message)
    return

def main(model_path = '/nfs/turbo/umms-indikar/shared/projects/RAG/models/llama-2-7b-chat.Q8_0.gguf', persist_directory = "/nfs/turbo/umms-indikar/shared/projects/RAG/databases/Transcription-Factors-5-10-2024/", llm=None, ragvectordb=None, embeddings_model=None):
    chatname = 'logs/RAG' + str(dt.now()) + '.json'
    chatname = '-'.join(chatname.split())
    print('Welcome to RAG! The chat log from this conversation will be saved to ' + chatname + '. How can I help?')
    
    databases = {} # a dictionary to look up databases
    tables = {}    # a dictionary to look up tables
    
    if llm is None:
        llm, callback_manager = load_llama(model_path) # load the llama
    if ragvectordb is None:
        ragvectordb, embeddings_model = load_literature_db(persist_directory) # load the literature database
    
    databases['RAG'] = ragvectordb
    externalDatabases = ['docs', 'GO', 'GGET']
    retriever = ragvectordb.as_retriever(search_kwargs={"k": 4}) ## Pick top 4 results from the search
    template = """At the end of each question, print all the genes named in the answer separated by commas.\n Context: {context}\n Question: {question}\n Answer:"""
    template = """Context: {context}\n Question: {question}\n Answer:"""
    QA_CHAIN_PROMPT = PromptTemplate(template=template, input_variables=["context" ,"question"],)

    router = getRouter()
    
    chatstatus = {
        'config'            : load_config(),
        'prompt'            : None,
        'output'            : None,
        'process'           : {},
        'llm'               : llm,
        'databases'         : databases,
        'current table'     : {'key':None, 'tab':None},
        'current documents' : None,
        'tables'            : {},
        'documents'         : {}
    }
    chatlog = {
        'llm'           : str(chatstatus['llm'])
    }
    while True:
        print('==================================================')
        chatstatus['prompt'] = input('Input >> ') # get query from user
        if chatstatus['prompt'].lower() == 'help':
            chatbotHelp()
            continue

        if '/force' not in chatstatus['prompt'].split(' '):
            route = router(chatstatus['prompt']).name # determine which path to use
        else:
            route = chatstatus['prompt'].split(' ')[1]
            chatstatus['prompt'] = buildRoutes(chatstatus['prompt'])
            

        print('==================================================')
        print('RAG >> ' + str(len(chatlog)) + ': ', end='')
        if chatstatus['prompt'] in ['exit', 'quit', 'q']:
            break
        elif chatstatus['prompt'].startswith('/set'):
            chatstatus = reconfig(chatstatus)
        # Query database
        elif route == 'GGET':
            logging.info('GGET')
            chatstatus = queryEnrichr(chatstatus)
        elif route == 'DATA':
            logging.info('DATA')
            chatstatus = manipulateTable(chatstatus)
        elif route == 'SCRAPE':
            logging.info('SCRAPE')
            chatstatus = webScraping(chatstatus)
        elif route == 'GENE ONTOLOGY':
            logging.info('GENE ONTOLOGY')
            chatstatus = goSearch(chatstatus)
        else:
            logging.info('RAG')
            chatstatus = queryDocs(chatstatus)
        chatlog = logger(chatlog, chatstatus, chatname)

        
# log output
#chatlog[len(chatlog)] = {
#    'prompt' : prompt,
#    'output' : output,
#    'process': loggedOutput,
#    'status' : chatstatus,
#}