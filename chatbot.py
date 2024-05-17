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
from RAG import *

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

def logger(chatlog, chatstatus, chatname):
    print(chatname)
    print(chatlog)
    chatlog[len(chatlog)] = {
        'prompt' : chatstatus['prompt'],  # the input to the user
        'output' : chatstatus['output'],  # the output to the user
        'process': chatstatus['process'], # information about the process that ran
        'status' : {                      # information about the chat at that time
            'databases'     : str(chatstatus['databases']),
#             'llm'           : str(chatstatus['llm']),
            'current table' : chatstatus['current table'],
            'current table' : chatstatus['current documents'],
        }
    }
#    print(chatlog)
    with open(chatname, 'w') as fp:
        json.dump(chatlog, fp, indent=4)
    return chatlog
    

def main(model_path = '/nfs/turbo/umms-indikar/shared/projects/RAG/models/llama-2-7b-chat.Q8_0.gguf', persist_directory = "/nfs/turbo/umms-indikar/shared/projects/RAG/databases/Transcription-Factors-5-10-2024/", ):
    chatname = 'RAG' + str(dt.now()) + '.json'
    chatname = '-'.join(chatname.split())
    print('Welcome to RAG! The chat log from this conversation will be saved to ' + chatname + '. How can I help?')
    
    databases = {} # a dictionary to look up databases
    tables = {}    # a dictionary to look up tables
    
    llm, callback_manager = load_llama(model_path) # load the llama
    ragvectordb, embeddings_model = load_literature_db(persist_directory) # load the literature database
    
    databases['RAG'] = ragvectordb
    externalDatabases = ['docs', 'GO', 'GGET']
    retriever = ragvectordb.as_retriever(search_kwargs={"k": 4}) ## Pick top 4 results from the search
    template = """At the end of each question, print all the genes named in the answer separated by commas.\n Context: {context}\n Question: {question}\n Answer:"""
    template = """Context: {context}\n Question: {question}\n Answer:"""
    QA_CHAIN_PROMPT = PromptTemplate(template=template, input_variables=["context" ,"question"],)

    router = getRouter()
    
    chatstatus = {
        'prompt'            : None,
        'output'            : None,
        'process'           : None,
        'llm'               : llm,
        'databases'         : databases,
        'current table'     : None,
        'current documents' : None,
    }
    chatlog = {
        'llm'           : str(chatstatus['llm'])
    }
    #    'llm'               : llm,
    #}
    while True:
        print('=========================')
        print('I'+str(len(chatlog))+':') # get query from user
        chatstatus['prompt'] = input()

        if chatstatus['prompt'] in ['exit', 'quit', 'q']: break  # parse prompt for user commands

        if '--force' not in chatstatus['prompt'].split(' '):
            route = router(chatstatus['prompt']).name # determine which path to use
        else:
            route = chatstatus['prompt'].split(' ')[1]

        # Code to handle recursive inputs
        # prompt += getPreviousInput(log, prompt.split(' ')[2].upper()) if len(prompt.split(' ')) >= 2 and prompt.split(' ')[1].upper() == 'R' else ''

        print('=========================')
        print('O' + str(len(chatlog)) + ':')
        # Query database
        if route == 'GGET':
            print('GGET')
            output, loggedOutput = queryGGET(chatstatus['prompt'])
        elif route == 'LOAD':
            print('LOAD')
            output, loggedOutput, docsearch = loadFile(chatstatus['prompt'])
            tableNum += 1
            tables[tableNum] = docsearch
            # output, loggedOutput, datadb = queryData(prompt)
        elif route == 'SCRAPE':
        #    Marc's code
            print('SCRAPE')
            loggedOutput = webScraping(chatstatus['prompt'])
        else:
            print('RAG')
            chatstatus['output'], loggedOutput = queryDocs(chatstatus['prompt'], chatstatus, llm)

        chatlog = logger(chatlog, chatstatus, chatname)

        
# log output
#chatlog[len(chatlog)] = {
#    'prompt' : prompt,
#    'output' : output,
#    'process': loggedOutput,
#    'status' : chatstatus,
#}