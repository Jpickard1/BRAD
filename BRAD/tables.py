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
import string
import matplotlib.colors as mcolors

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
    """
    Call manipulateTable with parameters extracted from chatstatus.

    :param chatstatus: A dictionary containing chat status information.
    :type chatstatus: dict

    The function sets up the process for manipulating tables by initializing various parameters related to table manipulation. It then determines the desired operation from the user prompt and proceeds accordingly. Supported operations include loading, saving, summarizing, handling missing data, and visualizing tables.

    If no specific table is selected, it attempts to find a table with a similar name in the chatstatus. If the operation is 'merge', it handles merging tables (this part is commented out).

    :return: Updated chatstatus with the results of the table manipulation.
    :rtype: dict

    """
    logging.info('manipulateTable')
    chatstatus['process'] : {
        'name'            : 'table',
        'edgecolor'       : None,
        'markersize'      : None,
        'linewidth'       : None,
        'xticks'          : None,
        'yticks'          : None,
        'markerfacecolor' : None,
        'legend'          : True,
        'hue'             : None,
        'bins'            : 10,
        'kde'             : True,
        'xlim'            : None,
        'ylim'            : None,
        'title'           : None,
        'xlabel'          : None,
        'ylabel'          : None,
        'linestyle'       : '-',
        'despine'         : False,
    }
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
    """
    Determine the operation to perform based on the user prompt.

    :param prompt: The user prompt containing instructions.
    :type prompt: str

    The function analyzes the user prompt to identify the desired operation. It checks for keywords related to different operations such as saving, loading, summarizing, handling missing data, and plotting.

    :return: The identified operation as a string. If the operation is unclear, it returns 'unsure'.
    :rtype: str

    """
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
    """
    Extract the first word from the text that ends with a specified file type.

    :param text: The input text to search through.
    :type text: str
    :param file_types: A list of file types/extensions to look for.
    :type file_types: list

    The function splits the input text into words and checks each word to see if it ends with any of the specified file types. If a match is found, it returns the word.

    :return: The first word that ends with a specified file type, or False if no such word is found.
    :rtype: str or bool

    """
    words = text.split(' ')
    for word in words:
        for file_type in file_types:
            if word.endswith(file_type):
                return word
    return False

def getColumns(df, prompt):
    """
    Extract columns from a DataFrame based on the user's prompt.

    :param df: The DataFrame to extract columns from.
    :type df: pandas.DataFrame
    :param prompt: The user's prompt containing column names to keep.
    :type prompt: str

    :return: A list of column names to keep.
    :rtype: list

    """
    cols = df.columns
    colsKeep = []
    for col in cols:
        if col in prompt:
            colsKeep.append(col)
    if len(colsKeep) == 0:
        colsKeep = df.columns
    return colsKeep

def loadFile(chatstatus):
    """
    Load a CSV or TSV file based on the user's prompt and update the chat status.

    :param chatstatus: A dictionary containing chat status information.
    :type chatstatus: dict

    The function reads the file name from the user's prompt, loads the file into a DataFrame,
    and updates the chat status with the loaded data and relevant process information.

    :return: The updated chat status.
    :rtype: dict
    """
    
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
    """
    Save the DataFrame to a CSV file based on the user's prompt and update the chat status.

    :param df: The DataFrame to save.
    :type df: pandas.DataFrame
    :param chatstatus: A dictionary containing chat status information.
    :type chatstatus: dict

    The function reads the file name from the user's prompt, saves the DataFrame to the file,
    and updates the chat status with relevant information.

    :return: The updated chat status.
    :rtype: dict

    """
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
    """
    Extract a summary command based on keywords in the user prompt.

    :param prompt: The user prompt containing the command.
    :type prompt: str

    The function checks the prompt for specific keywords to determine
    which summary command to return. It searches for keywords that match
    specific commands such as 'info', 'describe', 'head', 'tail', 'shape', and 'columns'.

    :return: The summary command corresponding to the keywords in the prompt.
    :rtype: str

    """
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
    """
    Summarize a DataFrame based on a command extracted from the user prompt.

    :param df: The DataFrame to be summarized.
    :type df: pandas.DataFrame
    :param chatstatus: A dictionary containing chat status information.
    :type chatstatus: dict

    The function extracts a summary command from the user prompt and
    generates the appropriate summary of the DataFrame. It supports commands
    such as 'info', 'describe', 'head', 'tail', 'shape', and 'columns'.

    :return: The updated chatstatus dictionary with the summary output.
    :rtype: dict

    """
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
    """
    Handle missing data in a DataFrame based on the user prompt.

    :param df: The DataFrame with missing data.
    :type df: pandas.DataFrame
    :param chatstatus: A dictionary containing chat status information.
    :type chatstatus: dict

    The function processes the user prompt to determine how to handle
    missing data in the DataFrame. It supports dropping rows with missing
    data and filling missing data with specific values (mean, median, or zero).
    It updates the chatstatus dictionary with the cleaned DataFrame and
    the corresponding output message.

    :return: The updated chatstatus dictionary with the missing data handling output.
    :rtype: dict

    """
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
    """
    Visualize a DataFrame based on the user prompt and chat status configuration.

    :param df: The DataFrame to be visualized.
    :type df: pandas.DataFrame
    :param chatstatus: A dictionary containing chat status information.
    :type chatstatus: dict

    The function determines the appropriate visualization method based
    on the user prompt and chat status configuration. It applies the
    specified plotting configurations and generates the corresponding
    visualization. If no specific plot type is found in the prompt, a
    random plot type is chosen.

    :return: The updated chatstatus dictionary with the visualization output.
    :rtype: dict

    """
    prompt = chatstatus['prompt'].lower()
    output = "Visualization created."
    plot_functions = plottingMethods()
    chatstatus = checkPlottingConfigurations(chatstatus, df)
    sns.set_palette(chatstatus['config']['display']['colormap'])
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
    ax = plot_func(df, chatstatus)
    plt.xlim(chatstatus['process']['xlim'])
    plt.ylim(chatstatus['process']['ylim'])
    if chatstatus['process']['despine']:
        sns.despine()
    if chatstatus['process']['title'] is not None:
        plt.title(chatstatus['process']['title'])
    if chatstatus['process']['xlabel'] is not None:
        plt.xlabel(chatstatus['process']['xlabel'])
    if chatstatus['process']['ylabel'] is not None:
        plt.ylabel(chatstatus['process']['ylabel'])
    if ax is not None:
        plt.show()
    print(output)
    return chatstatus

def plottingMethods():
    """
    Return a dictionary of plot function mappings.

    :return: A dictionary where the keys are plot types and the values are corresponding plot functions.
    :rtype: dict
    
    """
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

def line_plot(df, chatstatus):
    """
    Create a line plot based on the user prompt and DataFrame.

    :param df: The DataFrame containing the data to plot.
    :type df: pandas.DataFrame
    :param chatstatus: A dictionary containing chat status information.
    :type chatstatus: dict

    :return: A matplotlib Axes object representing the line plot.
    :rtype: matplotlib.axes.Axes

    """
    prompt = chatstatus['prompt']
    column = [col for col in df.columns if col.lower() in prompt.lower()]
    if column == []:
        return
    ax = df[column].plot(kind='line',
                         title=f'Line Plot of {column}',
                         colormap=chatstatus['config']['display']['colormap'],
                        )
    plt.xlabel(column)
    plt.ylabel('Value')
    return ax

def bar_plot(df, chatstatus):
    """
    Create a bar plot based on the user prompt and DataFrame.

    :param df: The DataFrame containing the data to plot.
    :type df: pandas.DataFrame
    :param chatstatus: A dictionary containing chat status information.
    :type chatstatus: dict

    :return: A seaborn Axes object representing the bar plot.
    :rtype: seaborn.axisgrid.FacetGrid

    """
    prompt = chatstatus['prompt']
    cols = [col.lower() for col in df.columns]
    column = None
    for word in prompt.split():
        if word.lower() in cols:
            column = word
            break
    if column is None:
        return
    # Make dataframe
    hue = chatstatus['process']['hue']
    if hue is not None:
        category_counts = df.groupby([column, hue]).size().reset_index(name='Count')
    else:
        category_counts = df.groupby([column]).size().reset_index(name='Count')
    # make plot
    kind = 'bar'
    if 'horizontal' in prompt.lower() or 'flip' in prompt.lower():
        kind = 'barh'
    if kind == 'bar':
        ax = sns.barplot(x=column,
                         y='Count',
                         hue=hue,
                         data=category_counts,
                         palette=chatstatus['config']['display']['colormap'],
                         edgecolor=chatstatus['process']['edgecolor']
                        )
    else:
        ax = sns.barplot(x='Count',
                         y=column,
                         hue=hue,
                         data=category_counts,
                         palette=chatstatus['config']['display']['colormap'],
                         edgecolor=chatstatus['process']['edgecolor']
                        )
    # Add labels
    if kind == 'bar':
        plt.xlabel(column)
        plt.ylabel('Frequency')
    else:
        plt.xlabel('Frequency')
        plt.ylabel(column)
    return ax

def histogram(df, chatstatus):
    """
    Create a histogram based on the user prompt and DataFrame.

    :param df: The DataFrame containing the data to plot.
    :type df: pandas.DataFrame
    :param chatstatus: A dictionary containing chat status information.
    :type chatstatus: dict

    :return: A seaborn Axes object representing the histogram.
    :rtype: seaborn.axisgrid.FacetGrid

    """
    prompt = chatstatus['prompt']
    columns = [col for col in df.columns if col.lower() in prompt.lower()]
    if columns:
        # column = column[0]
        ax = sns.histplot(df[columns].dropna(),
                          kde       = chatstatus['process']['kde'],
                          bins      = chatstatus['process']['bins'],
                          edgecolor = chatstatus['process']['edgecolor'],
                          color     = chatstatus['config']['display']['colormap']
                         )
        plt.title(f'Histogram of {columns}')
        plt.xlabel(columns)
        plt.ylabel('Frequency')
        return ax

def box_plot(df, chatstatus):
    """
    Create a box plot based on the user prompt and DataFrame.

    :param df: The DataFrame containing the data to plot.
    :type df: pandas.DataFrame
    :param chatstatus: A dictionary containing chat status information.
    :type chatstatus: dict

    :return: A seaborn Axes object representing the box plot.
    :rtype: seaborn.axisgrid.FacetGrid

    """
    prompt = chatstatus['prompt']
    columns = [col for col in df.columns if col.lower() in prompt.lower()]
    if columns:
        scalar = list(set(columns).intersection(set(df.select_dtypes(include=['number']).columns.tolist())))[0]
        catego = list(set(columns).intersection(set(df.select_dtypes(include=['object', 'category']).columns.tolist())))[0]
        if 'horizontal' in prompt.lower() or 'flip' in prompt.lower():
            scalar, catego = catego, scalar
        ax = sns.boxplot(data      = df,
                         x         = scalar,
                         y         = catego,
                         # linecolor = chatstatus['process']['edgecolor']
                         #kde       = chatstatus['process']['kde'],
                         #bins      = chatstatus['process']['bins'],
                         #edgecolor = chatstatus['process']['edgecolor'],
                         #color     = chatstatus['config']['display']['colormap']
                        )
        plt.title(f'Box Plot of {columns}')
        plt.xlabel(catego)
        plt.xlabel(scalar)
        return ax

def violin_plot(df, prompt):
    """
    Create a violin plot based on the user prompt and DataFrame.

    :param df: The DataFrame containing the data to plot.
    :type df: pandas.DataFrame
    :param prompt: The user prompt containing plot specifications.
    :type prompt: str

    :return: A seaborn Axes object representing the violin plot.
    :rtype: seaborn.axisgrid.FacetGrid

    """
    column = [col for col in df.columns if col.lower() in prompt.lower()]
    if column:
        # column = column[0]
        ax = sns.violinplot(data=df[column].dropna())
        plt.title(f'Violin Plot of {column}')
        plt.xlabel(column)
        return ax

def scatter_plot(df, chatstatus):
    """
    Create a scatter plot based on the user prompt and DataFrame.

    :param df: The DataFrame containing the data to plot.
    :type df: pandas.DataFrame
    :param chatstatus: A dictionary containing chat status information.
    :type chatstatus: dict

    :return: A seaborn Axes object representing the scatter plot.
    :rtype: seaborn.axisgrid.FacetGrid

    """
    prompt = chatstatus['prompt']
    columns = [col for col in df.columns if (col.lower() in prompt.lower() and col.lower() != chatstatus['process']['hue'].lower())]
    if chatstatus['debug']: print(columns)
    if len(columns) == 2:
        ax = sns.scatterplot(data = df,
                             x    = columns[0],
                             y    = columns[1],
                             size = chatstatus['process']['markersize'],
                             hue  = chatstatus['process']['hue'],
                             # markerfacecolor = chatstatus['config']['display']['markerfacecolor']
                            )
        plt.title(f'Scatter Plot between {columns[0]} and {columns[1]}')
        plt.xlabel(columns[0])
        plt.ylabel(columns[1])
        return ax

def pair_plot(df, prompt):
    """
    Create a pair plot for the given DataFrame.

    :param df: The DataFrame containing the data.
    :type df: pandas.DataFrame
    :param prompt: The user prompt containing additional information.
    :type prompt: str

    :return: The Seaborn pair plot axis object.
    :rtype: matplotlib.axes._subplots.AxesSubplot

    """
    ax = sns.pairplot(df)
    plt.title('Pair Plot')
    return ax

def heatmap(df, prompt):
    """
    Create a heatmap for the given DataFrame's correlation matrix.

    :param df: The DataFrame containing the data.
    :type df: pandas.DataFrame
    :param prompt: The user prompt containing additional information.
    :type prompt: str

    :return: The Seaborn heatmap axis object.
    :rtype: matplotlib.axes._subplots.AxesSubplot

    """
    ax = sns.heatmap(df.corr(), annot=True)
    plt.title('Heatmap')
    return ax

def pie_chart(df, prompt):
    """
    Create a pie chart for a specified column in the DataFrame.

    :param df: The DataFrame containing the data.
    :type df: pandas.DataFrame
    :param prompt: The user prompt specifying the column to plot.
    :type prompt: str

    :return: The pie chart axis object.
    :rtype: matplotlib.axes._subplots.AxesSubplot

    """
    column = [col for col in df.columns if col.lower() in prompt.lower()]
    if column:
        column = column[0]
        ax = df[column].value_counts().plot(kind='pie', autopct='%1.1f%%', title=f'Pie Chart of {column}')
        plt.ylabel('')
        return ax

def area_plot(df, prompt):
    """
    Create an area plot for a specified column in the DataFrame.

    :param df: The DataFrame containing the data.
    :type df: pandas.DataFrame
    :param prompt: The user prompt specifying the column to plot.
    :type prompt: str

    :return: The area plot axis object.
    :rtype: matplotlib.axes._subplots.AxesSubplot

    """
    column = [col for col in df.columns if col.lower() in prompt.lower()]
    if column:
        column = column[0]
        ax = df[column].plot(kind='area', title=f'Area Plot of {column}')
        plt.xlabel(column)
        plt.ylabel('Value')
        return ax

def hexbin_plot(df, prompt):
    """
    Create a hexbin plot for two specified columns in the DataFrame.

    :param df: The DataFrame containing the data.
    :type df: pandas.DataFrame
    :param prompt: The user prompt specifying the columns to plot.
    :type prompt: str

    :return: The hexbin plot axis object.
    :rtype: matplotlib.axes._subplots.AxesSubplot

    """
    column = [col for col in df.columns if col.lower() in prompt.lower()]
    if len(columns) == 2:
        ax = df.plot.hexbin(x=columns[0], y=columns[1], gridsize=25, cmap='Blues')
        plt.title(f'Hexbin Plot between {columns[0]} and {columns[1]}')
        plt.xlabel(columns[0])
        plt.ylabel(columns[1])
        return ax

def kde_plot(df, prompt):
    """
    Create a KDE plot for a specified column in the DataFrame.

    :param df: The DataFrame containing the data.
    :type df: pandas.DataFrame
    :param prompt: The user prompt specifying the column to plot.
    :type prompt: str

    :return: The KDE plot axis object.
    :rtype: matplotlib.axes._subplots.AxesSubplot

    """
    column = [col for col in df.columns if col.lower() in prompt.lower()]
    if column:
        column = column[0]
        ax = sns.kdeplot(df[column].dropna(), shade=True)
        plt.title(f'KDE Plot of {column}')
        plt.xlabel(column)
        plt.ylabel('Density')
        return ax

def facet_grid(df, prompt):
    """
    Create a facet grid for the specified columns in the DataFrame.

    :param df: The DataFrame containing the data.
    :type df: pandas.DataFrame
    :param prompt: The user prompt specifying the columns to use for the grid.
    :type prompt: str

    :return: The Seaborn facet grid object.
    :rtype: seaborn.axisgrid.FacetGrid

    """
    column = [col for col in df.columns if col.lower() in prompt.lower()]
    if len(columns) >= 2:
        ax = sns.FacetGrid(df, col=columns[0], row=columns[1]).map(plt.hist, columns[2] if len(columns) > 2 else df.columns[0])
        return ax

def joint_plot(df, prompt):
    """
    This function creates a joint plot based on the provided DataFrame and prompt.

    :param df: The DataFrame containing the data.
    :type df: pandas.DataFrame
    :param prompt: The user prompt.
    :type prompt: str

    The function searches for columns in the DataFrame that match words in the prompt.
    If exactly two matching columns are found, it creates a joint plot using seaborn's jointplot function.
    The joint plot displays the relationship between these two columns.

    :return: The seaborn jointplot object.
    :rtype: seaborn.axisgrid.JointGrid or None

    """
    column = [col for col in df.columns if col.lower() in prompt.lower()]
    if len(columns) == 2:
        ax = sns.jointplot(x=column[0], y=column[1], data=df, kind="scatter")
        return ax

def strip_plot(df, prompt):
    """
    This function creates a strip plot based on the provided DataFrame and prompt.

    :param df: The DataFrame containing the data.
    :type df: pandas.DataFrame
    :param prompt: The user prompt.
    :type prompt: str

    The function searches for columns in the DataFrame that match words in the prompt.
    If exactly two matching columns are found, it creates a strip plot using seaborn's stripplot function.
    The strip plot displays the distribution of one variable across levels of another variable.

    :return: The seaborn stripplot object.
    :rtype: matplotlib.axes._subplots.AxesSubplot or None

    """
    column = [col for col in df.columns if col.lower() in prompt.lower()]
    if len(columns) == 2:
        ax = sns.stripplot(x=column[0], y=column[1], data=df)
        plt.title(f'Strip Plot of {columns[1]} by {columns[0]}')
        plt.xlabel(columns[0])
        plt.ylabel(columns[1])
        return ax

def swarm_plot(df, prompt):
    """
    This function creates a swarm plot based on the provided DataFrame and prompt.

    :param df: The DataFrame containing the data.
    :type df: pandas.DataFrame
    :param prompt: The user prompt.
    :type prompt: str

    The function searches for columns in the DataFrame that match words in the prompt.
    If exactly two matching columns are found, it creates a swarm plot using seaborn's swarmplot function.
    The swarm plot displays the distribution of one variable across levels of another variable.

    :return: The seaborn swarmplot object.
    :rtype: matplotlib.axes._subplots.AxesSubplot or None

    """
    column = [col for col in df.columns if col.lower() in prompt.lower()]
    if len(columns) == 2:
        ax = sns.swarmplot(x=column[0], y=column[1], data=df)
        plt.title(f'Swarm Plot of {columns[1]} by {columns[0]}')
        plt.xlabel(columns[0])
        plt.ylabel(columns[1])
        return ax

def count_plot(df, prompt):
    """
    This function creates a count plot based on the provided DataFrame and prompt.

    :param df: The DataFrame containing the data.
    :type df: pandas.DataFrame
    :param prompt: The user prompt.
    :type prompt: str

    The function searches for columns in the DataFrame that match words in the prompt.
    If a matching column is found, it creates a count plot using seaborn's countplot function.
    The count plot displays the count of observations in each category.

    :return: The seaborn countplot object.
    :rtype: matplotlib.axes._subplots.AxesSubplot or None

    """
    column = [col for col in df.columns if col.lower() in prompt.lower()]
    if column:
        column = column[0]
        ax = sns.countplot(x=column, data=df)
        plt.title(f'Count Plot of {column}')
        plt.xlabel(column)
        plt.ylabel('Count')
        return ax

def cat_plot(df, prompt):
    """
    This function creates a categorical plot based on the provided DataFrame and prompt.

    :param df: The DataFrame containing the data.
    :type df: pandas.DataFrame
    :param prompt: The user prompt.
    :type prompt: str

    The function searches for columns in the DataFrame that match words in the prompt.
    If at least two matching columns are found, it creates a categorical plot using seaborn's catplot function.
    The categorical plot displays the distribution of one variable across levels of another variable using boxes.

    :return: The seaborn catplot object.
    :rtype: seaborn.axisgrid.FacetGrid or None

    """
    column = [col for col in df.columns if col.lower() in prompt.lower()]
    if len(columns) >= 2:
        ax = sns.catplot(x=column[0], y=column[1], kind="box", data=df)
        plt.title(f'Cat Plot of {column[1]} by {column[0]}')
        plt.xlabel(column[0])
        plt.ylabel(column[1])
        return ax

def reg_plot(df, prompt):
    """
    This function creates a regression plot based on the provided DataFrame and prompt.

    :param df: The DataFrame containing the data.
    :type df: pandas.DataFrame
    :param prompt: The user prompt.
    :type prompt: str

    The function searches for columns in the DataFrame that match words in the prompt.
    If exactly two matching columns are found, it creates a regression plot using seaborn's regplot function.
    The regression plot displays the relationship between two variables with a regression line.

    :return: The seaborn regplot object.
    :rtype: matplotlib.axes._subplots.AxesSubplot or None

    """
    column = [col for col in df.columns if col.lower() in prompt.lower()]
    if len(columns) == 2:
        ax = sns.regplot(x=column[0], y=column[1], data=df)
        plt.title(f'Regression Plot of {column[1]} by {column[0]}')
        plt.xlabel(columns[0])
        plt.ylabel(columns[1])
        return ax

def dist_plot(df, prompt):
    """
    This function creates a distribution plot based on the provided DataFrame and prompt.

    :param df: The DataFrame containing the data.
    :type df: pandas.DataFrame
    :param prompt: The user prompt.
    :type prompt: str

    The function searches for columns in the DataFrame that match words in the prompt.
    If a matching column is found, it creates a distribution plot using seaborn's distplot function.
    The distribution plot displays the distribution of a single variable.

    :return: The seaborn distplot object.
    :rtype: matplotlib.axes._subplots.AxesSubplot or None

    """
    column = [col for col in df.columns if col.lower() in prompt.lower()]
    if column:
        column = column[0]
        ax = sns.distplot(df[column].dropna(), kde=True)
        plt.title(f'Distribution Plot of {column}')
        plt.xlabel(column)
        plt.ylabel('Density')
        return ax

def separate_punctuation_with_spaces(text):
    """
    This function separates punctuation marks in a text with spaces.

    :param text: The input text.
    :type text: str

    The function uses regular expressions to replace each punctuation mark with ' <punctuation> '.

    :return: The text with separated punctuation.
    :rtype: str

    """
    # Use regular expression to replace each punctuation mark with ' <punctuation> '
    return re.sub(r'([.,!?;:"(){}\[\]])', r' \1 ', text)

def is_valid_colormap(colormap_name):
    """
    Check if a colormap name is valid.

    :param colormap_name: The name of the colormap.
    :type colormap_name: str

    :return: True if the colormap name is valid, False otherwise.
    :rtype: bool

    """
    return colormap_name in plt.colormaps()

def is_valid_color(color_string):
    """
    Check if a color string is valid.

    :param color_string: The color string to check.
    :type color_string: str

    :return: True if the color string is valid, False otherwise.
    :rtype: bool

    """
    try:
        mcolors.to_rgba(color_string)
        return True
    except ValueError:
        return False    

def should_apply_hue(promptWords, data, max_categories=15):
    """
    Detect if hue should be applied and determine the hue variable based on the prompt and data.

    :param promptWords: List of words parsed from the prompt.
    :type promptWords: list
    :param data: Pandas DataFrame containing the data.
    :type data: pandas.DataFrame
    :param max_categories: Maximum number of categories for a variable to be considered as hue.
    :type max_categories: int

    :return: The hue variable to use, or None if hue is not needed.
    :rtype: str or None

    """

    if 'hue' in promptWords:
        loc = promptWords.index('hue')
        hue_var = promptWords[loc + 1]
        if hue_var in data.columns:
            return hue_var
    
    # Check if there is a categorical column that could be used as hue
    categorical_columns = [col for col in data.columns if pd.api.types.is_categorical_dtype(data[col]) or data[col].dtype == 'object']
    categorical_columns = [col for col in categorical_columns if data[col].nunique() <= max_categories]

    if categorical_columns:
        return random.choice(categorical_columns)  # Return the first categorical column
    
    return None  # No hue variable detected

def checkPlottingConfigurations(chatstatus, df):
    """
    Check if any plotting configurations need to be changed based on the user prompt.

    :param chatstatus: A dictionary containing chat status information.
    :type chatstatus: dict
    :param df: The DataFrame containing the data.
    :type df: pandas.DataFrame

    :return: The updated chat status information.
    :rtype: dict

    """

    prompt = separate_punctuation_with_spaces(chatstatus['prompt']) # we don't mess with the '-' character
    promptWords = prompt.split(' ')
    # unwanted_strings = {'', ' ', 'of', 'as', 'use', 'is', 'to', 'by', 'the', ';', '(', '[', '.', ',', '!', '?', ';', ':', '"', '(', ')', '{', '}', '\[', '\]' ']', ')' } # we don't mess with the '-' character
    promptWords = [s for s in promptWords if s not in unwanted_strings]
    # if chatstatus['debug']: logging.info(promptWords)

    if 'dpi' in promptWords:
        loc = promptWords.index('dpi')
        try:
            chatstatus['config']['display']['dpi'] = int(promptWords[loc+1])
        except ValueError:
            print('The dpi value is unclear')
    if 'figsize' in promptWords:
        loc = promptWords.index('figsize')
        try:
            chatstatus['config']['display']['figsize'] = (int(promptWords[loc+1]), int(promptWords[loc+2]))
        except ValueError:
            print('The figsize value is unclear')
    if 'cm' in prompt or 'colormap' in prompt:
        loc = promptWords.index('cm') if 'cm' in promptWords else promptWords.index('colormap')
        if is_valid_colormap(promptWords[loc + 1]):
            chatstatus['config']['display']['colormap'] = promptWords[loc + 1]
        else:
            print('The colormap is unclear')
    if 'ec' in promptWords or 'edgecolor' in prompt or 'edge-color' in prompt:
        if 'ec' in promptWords:
            loc = promptWords.index('ec')
        elif 'edge-color' in prompt:
            loc = promptWords.index('edge-color')
        else:
            loc = promptWords.index('edgecolor')
        if is_valid_color(promptWords[loc + 1]):
            chatstatus['process']['edgecolor'] = promptWords[loc + 1]
        else:
            print('The colormap is unclear')
    if 'markersize' in promptWords:
        loc = promptWords.index('markersize')
        try:
            chatstatus['process']['markersize'] = int(promptWords[loc + 1])
        except ValueError:
            print('The markersize value is unclear')
    if 'linewidth' in promptWords:
        loc = promptWords.index('linewidth')
        try:
            chatstatus['process']['linewidth'] = int(promptWords[loc + 1])
        except ValueError:
            print('The linewidth value is unclear')
    #if 'grid' in promptWords:
    #    chatstatus['process']['grid'] = not chatstatus['config']['display']['grid']
    if 'xlim' in promptWords:
        loc = promptWords.index('xlim')
        try:
            chatstatus['process']['xlim'] = (int(promptWords[loc + 1]), int(promptWords[loc + 2]))
        except ValueError:
            print('The xlim value is unclear')
    if 'ylim' in promptWords:
        loc = promptWords.index('ylim')
        try:
            chatstatus['process']['ylim'] = (int(promptWords[loc + 1]), int(promptWords[loc + 2]))
        except ValueError:
            print('The ylim value is unclear')
    if 'markerfacecolor' in promptWords:
        loc = promptWords.index('markerfacecolor')
        if is_valid_color(promptWords[loc + 1]):
            chatstatus['process']['markerfacecolor'] = promptWords[loc + 1]
        else:
            print('The markerfacecolor is unclear')
    
    if 'legend' in promptWords:
        loc = promptWords.index('legend')
        chatstatus['process']['legend'] = promptWords[loc + 1].lower() == 'true'
    
    if 'fontsize' in promptWords:
        loc = promptWords.index('fontsize')
        try:
            chatstatus['process']['fontsize'] = int(promptWords[loc + 1])
        except ValueError:
            print('The fontsize value is unclear')
            
    if 'bins' in promptWords:
        loc = promptWords.index('bins')
        try:
            chatstatus['process']['bins'] = int(promptWords[loc + 1])
        except ValueError:
            try:
                chatstatus['process']['bins'] = int(promptWords[loc - 1])
            except ValueError:
                print('The bins value is unclear')
    
    if "spine" in prompt.lower():
        chatstatus['process']['despine'] = not chatstatus['process']['despine']

    if "kde" in prompt.lower():
        chatstatus['process']['kde'] = not chatstatus['process']['kde']
            
    hue_var = should_apply_hue(promptWords, df)
    if hue_var:
        chatstatus['process']['hue'] = hue_var
        
    # get string based parameters such as title, xlabel, and ylabel
    chatstatus = checkPlotLabels(chatstatus)

    return chatstatus

def checkPlotLabels(chatstatus):
    """
    Parses title, xlabel, and ylabel from the prompt.

    :param chatstatus: A dictionary containing chat status information.
    :type chatstatus: dict

    :return: The updated chat status information.
    :rtype: dict

    """
    prompt = chatstatus['prompt']
    patterns = {
        'title': r"title\s*'([^']*)'",
        'xlabel': r"x\s*label\s*'([^']*)'",
        'ylabel': r"y\s*label\s*'([^']*)'"
    }
    for label, pattern in patterns.items():
        match = re.search(pattern, prompt, re.IGNORECASE)
        if match:
            chatstatus['process'][label] = match.group(1)
    return chatstatus
    
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
'''