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
import io
import logging
from matplotlib import pyplot as plt
import seaborn as sns
import random

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
    logging.info('manipulateTable')
    process = {'name' : 'table'}
    prompt  = chatstatus['prompt']

    # select operation
    operation = selectOperation(prompt)

    # select operation
    if operation == 'unsure':
        print('the operation is not clear')
        
    if operation == 'load':
        chatstatus = loadFile(chatstatus)
    elif operation != 'merge':
        # select table
        df = chatstatus['current table']['tab']          # select the most recent table
        if df is None:
            selectTable = None
            for word in prompt.split(' '):
                for table in chatstatus['tables'].keys():
                    if str(word).upper() == str(table).upper():      # look for a table with a similar name
                        selectTable = word
            if selectTable is None:                            # return if no table
                print('No table selected')
                return
            df = chatstatus['tables'][selectTable]             # select the specific table    
        if operation == 'save':
            chatstatus = saveTable(df, chatstatus)
        elif operation == 'summarize':
            chatstatus = summarizeTable(df, chatstatus)
        elif operation == 'na':
            chatstatus = handleMissingData(df, chatstatus)
        elif operation == 'plot':
            chatstatus = visualizeTable(df, chatstatus)
    #elif operation == 'merge':
    #    mergeTables(chatstatus)
    return chatstatus

def selectOperation(prompt):
    logging.info('selectOperation')
    tokens = set(prompt.lower().split(' '))
    if set(['save']).intersection(tokens):                   # save
        return 'save'
    if set(['load', 'read', 'open', 'export']).intersection(tokens):                   # load
        return 'load'
    elif set(['summarize', 'describe', 'head', 'info', 'tail',
              'columns', 'first few', 'last few', 'structure',
              'show stats', 'list columns', 'details']).intersection(tokens):# sumamrize the data
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

def getColumns(df, prompt):
    cols = df.columns
    colsKeep = []
    for col in cols:
        if col in prompt:
            colsKeep.append(col)
    if len(colsKeep) == 0:
        colsKeep = df.columns
    return colsKeep

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
    chatstatus['current table']['tab'] = df
    chatstatus['current table']['key'] = str(len(chatstatus['tables']) + 1)
    chatstatus['tables'][chatstatus['current table']['key']] = chatstatus['current table']['tab']
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
    return chatstatus
    
def extract_summary_command(prompt):
    logging.info('extract_summary_command')
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
    logging.info('summarizeTable')
    prompt  = chatstatus['prompt']
    command = extract_summary_command(prompt)
    print(command)
    output  = ""
    if command == 'info':
        buffer = io.StringIO()
        df.info(buf=buffer)
        info_str = buffer.getvalue()
        output = f"DataFrame Info:\n\n{info_str}"
        print(output)  # Print to terminal
    elif command == 'describe':
        describe_str = df.describe().to_string()
        output = f"DataFrame Description:\n\n{describe_str}"
        print(output)  # Print to terminal
    elif command == 'head':
        head_str = df.head().to_string()
        output = f"Top Rows of DataFrame:\n\n{head_str}"
        print(output)  # Print to terminal
    elif command == 'tail':
        tail_str = df.tail().to_string()
        output = f"Bottom Rows of DataFrame:\n\n{tail_str}"
        print(output)  # Print to terminal
    elif command == 'shape':
        output = f"The DataFrame has {df.shape[0]} rows and {df.shape[1]} columns."
        print(output)  # Print to terminal
    elif command == 'columns':
        columns_str = ", ".join(df.columns)
        output = f"The columns of the DataFrame are:\n\n{columns_str}"
        print(output)  # Print to terminal
    else:
        output = 'No valid command found in the prompt.'
        print(output)  # Print to terminal
    return chatstatus

def handleMissingData(df, chatstatus):
    logging.info('handleMissingData')
    prompt  = chatstatus['prompt']
    if "drop" in prompt:
        chatstatus['current table']['tab'] = df.dropna()
        chatstatus['output'] = f"Rows with missing data have been dropped. The DataFrame now has {df_cleaned.shape[0]} rows and {df_cleaned.shape[1]} columns."
    elif "fill" in prompt:
        fill_value = 0
        if "mean" in prompt:
            chatstatus['current table']['tab'] = df.fillna(df.mean())
            fill_value = "mean values"
        elif "median" in prompt:
            chatstatus['current table']['tab'] = df.fillna(df.median())
            fill_value = "median values"
        else:
            chatstatus['current table']['tab'] = df.fillna(0)
            fill_value = 0
        chatstatus['output'] = f"Missing data has been filled with {fill_value}. The DataFrame now has {chatstatus['current table']['tab'].shape[0]} rows and {chatstatus['current table']['tab'].shape[1]} columns."
    else:
        missing_data_summary = df.isnull().sum()
        chatstatus['output'] = f"Missing Data Summary:\n\n{missing_data_summary.to_string()}"
    chatstatus['tables'][chatstatus['current table']['key']] = chatstatus['current table']['tab']
    print(chatstatus['output'])
    return chatstatus

def visualizeTable(df, chatstatus):
    prompt = chatstatus['prompt'].lower()
    output = "Visualization created."
    plot_functions = plottingMethods()
    plt.figure(figsize = chatstatus['config']['display']['figsize'],
               dpi     = chatstatus['config']['display']['dpi'],
              )
    ax = None
    for key in plot_functions:
        if key in prompt:
            plot_func = plot_functions[key]
            break
    else:
        plot_func = random.choice(list(plot_functions.values()))
        logging.info("No matching plot type found in the prompt.")
    ax = plot_func(df, prompt)

    if ax is not None:
        plt.show()
    print(output)
    return chatstatus

def plottingMethods():
    plot_functions = {
        "line"         : line_plot   ,
        "bar"          : bar_plot    ,
        "histogram"    : histogram   ,
        "box"          : box_plot    ,
        "boxplot"      : box_plot    ,
        "violin"       : violin_plot ,
        "scatter"      : scatter_plot,
        "pair"         : pair_plot   ,
        "heatmap"      : heatmap     ,
        "pie"          : pie_chart   ,
        "area"         : area_plot   ,
        "hexbin"       : hexbin_plot ,
        "kde"          : kde_plot    ,
        "facet grid"   : facet_grid  ,
        "joint"        : joint_plot  ,
        "strip"        : strip_plot  ,
        "swarm"        : swarm_plot  ,
        "count"        : count_plot  ,
        "cat"          : cat_plot    ,
        "reg"          : reg_plot    ,
        "dist"         : dist_plot   ,
        "distribution" : dist_plot
    }
    return plot_functions

def line_plot(df, prompt):
    column = [col for col in df.columns if col.lower() in prompt.lower()]
    if column:
        # column = column[0]
        ax = df[column].plot(kind='line', title=f'Line Plot of {column}')
        plt.xlabel(column)
        plt.ylabel('Value')
        return ax

def bar_plot(df, prompt):
    column = [col for col in df.columns if col.lower() in prompt.lower()]
    kind = 'bar'
    if 'horizontal' in prompt.lower() or 'flip' in prompt.lower():
        kind = 'barh'
    if column:
        column = column[0]
        ax = df[column].value_counts().plot(kind=kind, title=f'Bar Plot of {column}')
        if kind == 'bar':
            plt.xlabel(column)
            plt.ylabel('Frequency')
        else:
            plt.xlabel('Frequency')
            plt.ylabel(column)
        return ax

def histogram(df, prompt):
    column = [col for col in df.columns if col.lower() in prompt.lower()]
    if column:
        # column = column[0]
        ax = sns.histplot(df[column].dropna(), kde=True)
        plt.title(f'Histogram of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        return ax

def box_plot(df, prompt):
    column = [col for col in df.columns if col.lower() in prompt.lower()]
    if column:
        # column = column[0]
        ax = sns.boxplot(x=df[column].dropna())
        plt.title(f'Box Plot of {column}')
        plt.xlabel(column)
        return ax

def violin_plot(df, prompt):
    column = [col for col in df.columns if col.lower() in prompt.lower()]
    if column:
        # column = column[0]
        ax = sns.violinplot(data=df[column].dropna())
        plt.title(f'Violin Plot of {column}')
        plt.xlabel(column)
        return ax

def scatter_plot(df, prompt):
    columns = [col for col in df.columns if col.lower() in prompt.lower()]
    if len(columns) == 2:
        ax = sns.scatterplot(x=df[columns[0]], y=df[columns[1]])
        plt.title(f'Scatter Plot between {columns[0]} and {columns[1]}')
        plt.xlabel(columns[0])
        plt.ylabel(columns[1])
        return ax

def pair_plot(df, prompt):
    ax = sns.pairplot(df)
    plt.title('Pair Plot')
    return ax

def heatmap(df, prompt):
    ax = sns.heatmap(df.corr(), annot=True)
    plt.title('Heatmap')
    return ax

def pie_chart(df, prompt):
    column = [col for col in df.columns if col.lower() in prompt.lower()]
    if column:
        column = column[0]
        ax = df[column].value_counts().plot(kind='pie', autopct='%1.1f%%', title=f'Pie Chart of {column}')
        plt.ylabel('')
        return ax

def area_plot(df, prompt):
    column = [col for col in df.columns if col.lower() in prompt.lower()]
    if column:
        column = column[0]
        ax = df[column].plot(kind='area', title=f'Area Plot of {column}')
        plt.xlabel(column)
        plt.ylabel('Value')
        return ax

def hexbin_plot(df, prompt):
    column = [col for col in df.columns if col.lower() in prompt.lower()]
    if len(columns) == 2:
        ax = df.plot.hexbin(x=columns[0], y=columns[1], gridsize=25, cmap='Blues')
        plt.title(f'Hexbin Plot between {columns[0]} and {columns[1]}')
        plt.xlabel(columns[0])
        plt.ylabel(columns[1])
        return ax

def kde_plot(df, prompt):
    column = [col for col in df.columns if col.lower() in prompt.lower()]
    if column:
        column = column[0]
        ax = sns.kdeplot(df[column].dropna(), shade=True)
        plt.title(f'KDE Plot of {column}')
        plt.xlabel(column)
        plt.ylabel('Density')
        return ax

def facet_grid(df, prompt):
    column = [col for col in df.columns if col.lower() in prompt.lower()]
    if len(columns) >= 2:
        ax = sns.FacetGrid(df, col=columns[0], row=columns[1]).map(plt.hist, columns[2] if len(columns) > 2 else df.columns[0])
        return ax

def joint_plot(df, prompt):
    column = [col for col in df.columns if col.lower() in prompt.lower()]
    if len(columns) == 2:
        ax = sns.jointplot(x=column[0], y=column[1], data=df, kind="scatter")
        return ax

def strip_plot(df, prompt):
    column = [col for col in df.columns if col.lower() in prompt.lower()]
    if len(columns) == 2:
        ax = sns.stripplot(x=column[0], y=column[1], data=df)
        plt.title(f'Strip Plot of {columns[1]} by {columns[0]}')
        plt.xlabel(columns[0])
        plt.ylabel(columns[1])
        return ax

def swarm_plot(df, prompt):
    column = [col for col in df.columns if col.lower() in prompt.lower()]
    if len(columns) == 2:
        ax = sns.swarmplot(x=column[0], y=column[1], data=df)
        plt.title(f'Swarm Plot of {columns[1]} by {columns[0]}')
        plt.xlabel(columns[0])
        plt.ylabel(columns[1])
        return ax

def count_plot(df, prompt):
    column = [col for col in df.columns if col.lower() in prompt.lower()]
    if column:
        column = column[0]
        ax = sns.countplot(x=column, data=df)
        plt.title(f'Count Plot of {column}')
        plt.xlabel(column)
        plt.ylabel('Count')
        return ax

def cat_plot(df, prompt):
    column = [col for col in df.columns if col.lower() in prompt.lower()]
    if len(columns) >= 2:
        ax = sns.catplot(x=column[0], y=column[1], kind="box", data=df)
        plt.title(f'Cat Plot of {column[1]} by {column[0]}')
        plt.xlabel(column[0])
        plt.ylabel(column[1])
        return ax

def reg_plot(df, prompt):
    column = [col for col in df.columns if col.lower() in prompt.lower()]
    if len(columns) == 2:
        ax = sns.regplot(x=column[0], y=column[1], data=df)
        plt.title(f'Regression Plot of {column[1]} by {column[0]}')
        plt.xlabel(columns[0])
        plt.ylabel(columns[1])
        return ax

def dist_plot(df, prompt):
    column = [col for col in df.columns if col.lower() in prompt.lower()]
    if column:
        column = column[0]
        ax = sns.distplot(df[column].dropna(), kde=True)
        plt.title(f'Distribution Plot of {column}')
        plt.xlabel(column)
        plt.ylabel('Density')
        return ax

'''
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
    elif "scatter" in prompt:
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
    elif "box" in prompt or "boxplot" in prompt:
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
    print(output)
    return chatstatus
    
    if "line" in prompt:
        ax = line_plot(df, prompt)
    elif "bar" in prompt:
        ax = bar_plot(df, prompt)
    elif "histogram" in prompt:
        ax = histogram(df, prompt)
    elif "box" in prompt or "boxplot" in prompt:
        ax = box_plot(df, prompt)
    elif "violin" in prompt:
        ax = violin_plot(df, prompt)
    elif "scatter" in prompt:
        ax = scatter_plot(df, prompt)
    elif "pair" in prompt:
        ax = pair_plot(df, prompt)
    elif "heatmap" in prompt:
        ax = heatmap(df, prompt)
    elif "pie" in prompt:
        ax = pie_chart(df, prompt)
    elif "area" in prompt:
        ax = area_plot(df, prompt)
    elif "hexbin" in prompt:
        ax = hexbin_plot(df, prompt)
    elif "kde" in prompt:
        ax = kde_plot(df, prompt)
    elif "facet grid" in prompt:
        ax = facet_grid(df, prompt)
    elif "joint" in prompt:
        ax = joint_plot(df, prompt)
    elif "strip" in prompt:
        ax = strip_plot(df, prompt)
    elif "swarm" in prompt:
        ax = swarm_plot(df, prompt)
    elif "count" in prompt:
        ax = count_plot(df, prompt)
    elif "cat" in prompt:
        ax = cat_plot(df, prompt)
    elif "reg" in prompt:
        ax = reg_plot(df, prompt)
    elif "dist" in prompt or "distribution" in prompt:
        ax = dist_plot(df, prompt)
    else:
        output = "No valid visualization command found in the prompt."
    
'''