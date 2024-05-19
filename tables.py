import pandas as pd
import numpy as np
from copy import deepcopy
import os
import sys
from importlib import reload
from scipy.stats import zscore
from scipy.stats import entropy
import scipy.io
import scipy
import textwrap
from scipy import sparse
import importlib
from itertools import product
from datetime import datetime
from IPython.display import display # displaying dataframes
import string
import warnings
import re

from langchain.llms import Replicate
from langchain.vectorstores import Pinecone
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS


def extract_csv_word(text, file_types):
    words = text.split(' ')
    for word in words:
        for file_type in file_types:
            if word.endswith(file_type):
                return word
    return False

def loadFile(chatstatus):
    '''
    implemented for csv files only (tsv should work as well)
    '''
    prompt              = chatstatus['prompt']
    file_types          = chatstatus['config']['acceptable_data_files']
    num_df_rows_display = chatstatus['config']['num_df_rows_display']
    
    file = extract_csv_word(prompt, file_types)
    if not file:
        output = 'no file found'
        print(output)
        return output, {'Load': 'Failed'}
    output = 'loading: ' + file
    print(output)
    process = {'filename' : file}
    
    # load the file
    if file.endswith('.csv'):
        sep = ','
    elif file.endswith('.tsv'):
        sep = '\t'
    df = pd.read_csv(file, sep=sep)
    process['table'] = df.to_json()
    display(df[:num_df_rows_display].style)
    loader = CSVLoader(file)  # I am not sure how this line works with .tsv data
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks = text_splitter.split_documents(data)
    embeddings_model = HuggingFaceEmbeddings(model_name='BAAI/bge-base-en-v1.5')
    process['database'] = FAISS.from_documents(text_chunks, embeddings_model)
    return output, process