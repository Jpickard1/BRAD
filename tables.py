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


def manipulateTable(chatstatus):
    process = {'name' : 'table'}
    prompt  = chatstatus['prompt']

    # select operation
    operation = selectOperation(prompt)
    # select operation
    if operation == 'unsure':
        print('the operation is not clear')

    if operation == 'load':
        chatstatus = loadFile(chatstatus)
    else:
        # select table
        selectTable = chatstatus['current table']          # select the most recent table
        if selectTable is not None:
            for word in prompt.split(' '):
                for table in chatstatus['tables'].keys():
                    if str(word).upper() == str(table).upper():      # look for a table with a similar name
                        selectTable = word
        if selectTable is None:                            # return if no table
            print('No table selected')
            return
        df = chatstatus['tables'][selectTable]             # select the specific table    
        if operation == 'save':
            saveTable(df, chatstatus)
        elif operation == 'summarize':
            summarizeTable(df, chatstatus)
        elif operation == 'na':
            handleMissingData(df, chatstatus)
        elif operation == 'plot':
            visualizeTable(df, chatstatus)
    return chatstatus

def selectOperation(prompt):
    tokens = set(prompt.lower().split(' '))
    if set(['save']).intersection(tokens):                   # save
        return 'save'
    if set(['load', 'read', 'open']).intersection(tokens):                   # load
        return 'load'
    elif set(['describe', 'head', 'info', 'tail', 'columns']).intersection(tokens):# sumamrize the data
        return 'summarize'
    elif set(['na', 'n/a', 'fill', 'missing']).intersection(tokens):    # handle missing data
        return 'na'
    elif set(['plot', 'illustrate', 'vis', 'visualize']).intersection(tokens):    # plot data
        return 'plot'
    return 'unsure'
        
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
    output = 'loading: ' + file + ' as Table ' + str(len(chatstatus['tables']) + 1)
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
    chatstatus['process']       = process
    chatstatus['output']        = output
    chatstatus['current table'] = df
    chatstatus['tables'][str(len(chatstatus['tables']) + 1)] = df
    # print(chatstatus['current table'])
    return chatstatus
        
def saveTable(df, chatstatus):
    prompt              = chatstatus['prompt']
    file_types          = chatstatus['config']['acceptable_data_files']
    file = extract_csv_word(prompt, file_types)
    if not file:
        output = 'no file found, so default naming used'
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        file = f'save-table-{timestamp}.csv'
    print('saving the table to ' + file)
    df.to_csv(file, index=False)
    return f'Table saved as {file}'
    
def extract_summary_command(prompt):
    commands = {
        'info': ['info', 'information', 'details'],
        'describe': ['describe', 'summary', 'stats'],
        'head': ['head', 'top rows'],
        'tail': ['tail', 'bottom rows'],
        'shape': ['shape', 'dimensions'],
        'columns': ['columns', 'fields', 'column names']
    }
    prompt = prompt.lower()
    for command, keywords in commands.items():
        if any(keyword in prompt for keyword in keywords):
            return command
    return 'info'  # Default command if none match

def summarizeTable(df, chatstatus):
    prompt  = chatstatus['prompt']
    command = extract_summary_command(prompt)
    output  = ""
    if command == 'info':
        buffer = io.StringIO()
        df.info(buf=buffer)
        info_str = buffer.getvalue()
        output = f"DataFrame Info:\n\n{info_str}"
    elif command == 'describe':
        describe_str = df.describe().to_string()
        output = f"DataFrame Description:\n\n{describe_str}"
    elif command == 'head':
        head_str = df.head().to_string()
        output = f"Top Rows of DataFrame:\n\n{head_str}"
    elif command == 'tail':
        tail_str = df.tail().to_string()
        output = f"Bottom Rows of DataFrame:\n\n{tail_str}"
    elif command == 'shape':
        output = f"The DataFrame has {df.shape[0]} rows and {df.shape[1]} columns."
    elif command == 'columns':
        columns_str = ", ".join(df.columns)
        output = f"The columns of the DataFrame are:\n\n{columns_str}"
    else:
        output = 'No valid command found in the prompt.'
    return output

def handleMissingData(df, chatstatus):
    if "drop" in prompt:
        df_cleaned = df.dropna()
        output = f"Rows with missing data have been dropped. The DataFrame now has {df_cleaned.shape[0]} rows and {df_cleaned.shape[1]} columns."
    elif "fill" in prompt:
        fill_value = 0
        if "with mean" in prompt:
            df_cleaned = df.fillna(df.mean())
            fill_value = "mean values"
        elif "with median" in prompt:
            df_cleaned = df.fillna(df.median())
            fill_value = "median values"
        else:
            df_cleaned = df.fillna(0)
            fill_value = 0
        output = f"Missing data has been filled with {fill_value}. The DataFrame now has {df_cleaned.shape[0]} rows and {df_cleaned.shape[1]} columns."
    else:
        missing_data_summary = df.isnull().sum()
        output = f"Missing Data Summary:\n\n{missing_data_summary.to_string()}"
    return output

def visualizeTable(df, chatstatus):
    prompt = chatstatus['prompt'].lower()
    output = "Visualization created."
    if "histogram" in prompt:
        column = [col for col in df.columns if col in prompt]
        if column:
            column = column[0]
            plt.figure(figsize=(10, 6))
            sns.histplot(df[column].dropna(), kde=True)
            plt.title(f'Histogram of {column}')
            plt.xlabel(column)
            plt.ylabel('Frequency')
            plt.show()
        else:
            output = "No specific column found in the prompt for histogram."
    elif "scatter plot" in prompt:
        columns = [col for col in df.columns if col in prompt]
        if len(columns) == 2:
            plt.figure(figsize=(10, 6))
            sns.scatterplot(x=df[columns[0]], y=df[columns[1]])
            plt.title(f'Scatter Plot between {columns[0]} and {columns[1]}')
            plt.xlabel(columns[0])
            plt.ylabel(columns[1])
            plt.show()
        else:
            output = "Scatter plot requires exactly two columns mentioned in the prompt."
    elif "box plot" in prompt or "boxplot" in prompt:
        column = [col for col in df.columns if col in prompt]
        if column:
            column = column[0]
            plt.figure(figsize=(10, 6))
            sns.boxplot(x=df[column].dropna())
            plt.title(f'Box Plot of {column}')
            plt.xlabel(column)
            plt.show()
        else:
            output = "No specific column found in the prompt for box plot."
    else:
        output = "No valid visualization command found in the prompt."
    return output