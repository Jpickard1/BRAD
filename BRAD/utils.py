"""
This module provides a set of utility functions designed to streamline common tasks related to file management, 
data handling, directory operations, and more across the core and tool modules.

Scope
=====

The goal of this module is to offer a reusable, general-purpose utilities that simplify routine tasks that interface the LLM with other aspects of the
code. These tasks include saving and loading files, ensuring directories  exist, generating standardized file paths,
and more. Each function is designed to abstract repetitive operations and enhance code clarity, maintainability,
and reliability. The functions in this module can be imported as needed when building different aspects of the BRAD framework.




Available Methods
=================


This module contains the following methods:

"""

import re
import os
import time
import numpy as np
import pandas as pd
import subprocess
import difflib
import matplotlib.pyplot as plt
import shutil

from langchain import PromptTemplate, LLMChain
from langchain_community.callbacks import get_openai_callback

from BRAD import log
from BRAD.promptTemplates import fileChooserTemplate, fieldChooserTemplate

def save(state, data, name):
    """
    Save data to a specified output directory, with optional stage number prefix.

    This function saves the provided data to a specified output directory within 
    the `state` configuration. If the `state` is part of a pipeline, 
    it prefixes the filename with the current stage number.

    Args:
        state (dict): A dictionary containing the current chat status, 
                           including queued pipeline stages and output directory.
        data (pd.DataFrame or str): The data to be saved. It can be either a 
                                    pandas DataFrame (for CSV output) or a string (for .tex output).
        name (str): The name of the output file.

    Returns:
        dict: The updated `state` dictionary with information about the saved file.

    Raises:
        ValueError: If the data type is not a DataFrame for CSV or a string for .tex files.

    """
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: June 19, 2024

    # If this is part of a pipeline, then add the stage number to the printed output
    if len(state['queue']) != 0:
        stageNum = state['queue pointer'] + 1#[0]['order']
        name = 'S' + str(stageNum) + '-' + name
    output_path = os.path.join(state['output-directory'], name)

    if isinstance(data, pd.DataFrame):
        data.to_csv(output_path, index=False)
    elif output_path.endswith('.tex'):
        with open(output_path, 'w') as file:
            file.write(data)
    else:
        raise ValueError("Unsupported data type or file extension. Use a DataFrame for CSV or a string for .tex files.")
    
    log.debugLog('The information has been saved to: ' + output_path, state=state)
    state['process']['steps'].append(
        {
            'func'     : 'utils.save',
            'new file' : output_path
        }
    )
    return state

def savefig(state, ax, name):
    """
    Save a matplotlib figure to a specified output directory, with optional stage number prefix.

    This function saves the provided matplotlib axis (`ax`) as a figure to a specified 
    output directory within the `state` configuration. If the `state` is part 
    of a pipeline, it prefixes the filename with the current stage number.

    Args:
        state (dict): A dictionary containing the current chat status, including 
                           queued pipeline stages and output directory.
        ax (matplotlib.axes.Axes): The matplotlib axis object containing the figure to be saved.
        name (str): The name of the output file.

    Returns:
        dict: The updated `state` dictionary with information about the saved file.

    """
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: June 19, 2024
    log.debugLog("SAVEFIG", state=state)
    if len(state['queue']) != 0:
        stageNum = state['queue pointer'] + 1 # [0]['order']
        name = 'S' + str(stageNum) + '-' + name
    output_path = os.path.join(state['output-directory'], state['config']['image-path-extension'], name)
    ensure_directory_exists(output_path, state)
    plt.savefig(output_path)
    log.debugLog('The image was saved to: ' + output_path, state=state)
    state['process']['steps'].append(
        {
            'func'     : 'utils.savefig',
            'new file' : output_path
        }
    )
    return state

def ensure_directory_exists(file_path, state):
    """
    Ensure that the directory for a given file path exists, creating it if necessary.

    This function checks if the directory path for the provided `file_path` exists.
    If the directory does not exist, it creates the directory. It prints a message
    indicating whether the directory was created or if it already existed.

    Args:
        file_path (str): The full file path for which the directory needs to be checked/created.

    """
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: June 23, 2024
    directory_path = os.path.dirname(file_path)
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        log.debugLog(f"Directory '{directory_path}' created.", state=state)
    else:
        log.debugLog(f"Directory '{directory_path}' already exists.", state=state)


def pdfDownloadPath(state):
    """
    Generate the file path for downloading PDF files.

    This function constructs the file path for downloading PDF files based on the 
    `output-directory` specified in the `state` dictionary. It appends 'pdf'
    to the output directory path to indicate the location where PDF files should be saved.

    Args:
        state (dict): A dictionary containing the chat status and configuration details.
                           It must include the key 'output-directory'.

    Returns:
        str: The complete file path for downloading PDF files.

    """
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: June 19, 2024
    path = os.path.join(state['output-directory'], 'pdf')
    return path

def outputFiles(state):
    """
    Retrieve a list of all files in the output directory.

    This function lists all files present in the `output-directory` specified in the 
    `state` dictionary and returns them as a list.

    Args:
        state (dict): A dictionary containing the chat status and configuration details.
                           It must include the key 'output-directory'.

    Returns:
        list: A list of filenames present in the output directory.

    """
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: June 19, 2024
    output_files = []
    for filename in os.listdir(state['output-directory']):
        output_files.append(filename)
    return output_files

def makeNamesConsistent(state, files):
    """
    Ensure filenames in the output directory are consistent with the pipeline stage numbering.

    This function renames files in the output directory to include the current stage number
    from the pipeline. If a file's name does not start with 'S', it will be prefixed with the 
    stage number. Additionally, it removes any '/' or '\\' characters from filenames.

    Args:
        state (dict): A dictionary containing the chat status and configuration details.
                           It must include the keys 'queue' and 'output-directory'.
        files (list): A list of filenames to be processed.

    Returns:
        dict: Updated state with renamed files logged in 'process' steps.

    """
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: June 19, 2024

    # Dev. Comments:
    # -------------------
    # This function executes a single user prompt with BRAD
    #
    # Issues:
    # - It is not clear why there are 2 for loops that renamd files
    #
    if len(state['queue']) != 0:
        log.debugLog('Finding Stage Number of Pipeline', state=state)
        log.debugLog(state['queue'], state=state)
        IP = state['queue pointer'] # [0]['order'] + 1
        IP = int(IP)
    else:
        return
    renamedFiles = []
    log.debugLog(f"{IP=}", state=state)
    log.debugLog(f"{type(IP)=}", state=state)
    for file in files:
        if file[0] != 'S':
            old_path = os.path.join(state['output-directory'], file)
            if os.path.isdir(old_path):
                continue
            new_path = os.path.join(state['output-directory'], 'S' + str(IP) + '-' + file)
            renamedFiles.append(
                {
                    'old-name' : old_path,
                    'new-name' : new_path
                }
            )
            os.rename(old_path, new_path)
            if 'output' not in state['queue'][IP].keys():
                state['queue'][IP] = []
            state['queue'][IP]['output'].append(new_path)
    for file in outputFiles(state):
        old_path = os.path.join(state['output-directory'], file)
        new_path = os.path.join(state['output-directory'], file.replace('/', '').replace('\\', ''))
        if old_path != new_path:
            renamedFiles.append(
                {
                    'old-name' : old_path,
                    'new-name' : new_path
                }
            )
            os.rename(old_path, new_path)
            if 'output' not in state['queue'][IP].keys():
                state['queue'][IP] = []
            state['queue'][IP]['output'].append(new_path)
    state['process']['steps'].append(
        {
            'func'  : 'utils.makeNamesConsistent',
            'files' : renamedFiles
        }
    )
    return state

def loadFromFile(state):
    """
    Loads data from a file selected by an LLM prompt based on user input.

    This function interacts with a language model to select a file from available files
    in the output directory. It extracts the specified fields from the selected file 
    and returns the data along with updated chat status.

    Args:
        state (dict): A dictionary containing the chat status and configuration details.
                           It must include the keys 'prompt', 'llm', and 'output-directory'.

    Returns:
        tuple: Updated state dictionary and a list of values from the specified fields in the file.

    """
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: June 19, 2024
    prompt = state['prompt']
    llm    = state['llm']
    # Get files to choose from
    availableFilesList = outputFiles(state)
    availableFiles = '\n'.join(availableFilesList)
    log.debugLog(availableFiles, state=state)
    
    # Build lang chain
    template = fileChooserTemplate()
    template = template.format(files=availableFiles)
    log.debugLog(template, state=state)
    PROMPT   = PromptTemplate(input_variables=["user_query"], template=template)
    chain    = PROMPT | llm

    # Call chain
    state   = log.userOutput(prompt, state=state)
    start_time = time.time()
    with get_openai_callback() as cb:
        responseFull = chain.invoke(prompt)
    response = responseFull.content.strip()
    responseFull = {'content': responseFull}
    responseFull['time'] = time.time() - start_time
    responseFull['call back'] = {
            "Total Tokens": cb.total_tokens,
            "Prompt Tokens": cb.prompt_tokens,
            "Completion Tokens": cb.completion_tokens,
            "Total Cost (USD)": cb.total_cost
    }
    
    # Regular expressions to extract file and fields
    file_pattern = r"File:\s*(\S+)"
    fields_pattern = r"Fields:\s*(.+)"

    # Search for patterns in the response
    file_match = re.search(file_pattern, response)
    fields_match = re.search(fields_pattern, response)

    # Extract the matched values
    file = file_match.group(1) if file_match else None
    fields = fields_match.group(1) if fields_match else None

    # Find the file that is most similar to the extracted file
    scores = []
    for availableFile in availableFilesList:
        scores.append(word_similarity(file, availableFile))
    file = availableFilesList[np.argmax(scores)]
    
    log.debugLog('File=' + str(file) + '\n' + 'Fields=' + str(fields), state=state)
    state['process']['steps'].append(
        log.llmCallLog(
            llm          = llm,
            prompt       = PROMPT,
            input        = prompt,
            output       = responseFull,
            parsedOutput = {
                'File'   : file,
                'Fields' : fields
            },
            purpose      = 'Select File'
        )
    )
    
    # Determine the delimiter based on the file extension
    delimiter = ',' if not file.endswith('.tsv') else '\t'
    
    # Read the file into a DataFrame
    loadfile = os.path.join(state['output-directory'], file)
    df = pd.read_csv(loadfile, delimiter=delimiter)
    state['process']['steps'].append(log.loadFileLog(file      = loadfile,
                                                          delimiter = delimiter)
                                         )

    if fields not in df.columns:
        state, fields = fieldSelectorFromDataFrame(state, df)

    return state, list(df[fields].values)

def fieldSelectorFromDataFrame(state, df):
    """
    Selects a field from a DataFrame using a language model prompt.

    This function uses a language model to select a specific field from the columns of a given DataFrame.
    It builds a prompt with the available columns, invokes the language model, and parses the response to
    determine the selected field.

    Args:
        state (dict): A dictionary containing the chat status and configuration details.
                           It must include the keys 'llm', 'prompt', and 'process'.
        df (pandas.DataFrame): The DataFrame from which a field will be selected.

    Returns:
        tuple: Updated state dictionary and the selected field as a string.

    """
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: June 19, 2024
    llm      = state['llm']
    prompt   = state['prompt']
    template = fieldChooserTemplate()
    template = template.format(columns=', '.join(list(df.columns)))
    PROMPT   = PromptTemplate(input_variables=["user_query"], template=template)
    chain    = PROMPT | llm

    # Call chain
    response = chain.invoke(prompt).content.strip()
    fields = response.split('=')[1].strip()
    state['process']['steps'].append(log.llmCallLog(llm          = llm,
                                                         prompt       = PROMPT,
                                                         input        = prompt,
                                                         output       = response,
                                                         parsedOutput = {
                                                             'Fields' : fields
                                                         },
                                                         purpose      = 'Select Field'
                                                        )
                                        )

    log.debugLog('field identifier response=\n'+fields, state=state)
    return state, fields

def word_similarity(word1, word2):
    """
    Calculate the similarity ratio between two words using SequenceMatcher.

    This function computes the similarity ratio between two input words. The ratio is calculated
    based on the longest contiguous matching subsequence between the two words using the
    `difflib.SequenceMatcher` from the Python standard library.


    :param word1: The first word to compare.
    :type word1: str
    :param word2: The second word to compare.
    :type word2: str
    
    :return: A float value between 0 and 1 representing the similarity ratio. A value of 1.0 means the words
             are identical, while 0.0 means they are completely different.
    :rtype: (float)

    """
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: June 23, 2024
    return difflib.SequenceMatcher(None, word1, word2).ratio()

def outputFromPriorStep(state, step, values=None):
    """
    Retrieve the output from a prior step in the pipeline.

    .. warning:: We may be removing this function soon.

    This function searches for and loads the output file corresponding to a specified step in the pipeline.
    If the file is a CSV, it loads the data into a DataFrame. Optionally, specific columns can be selected from the DataFrame.

    Args:
        state (dict): The dictionary containing the current status and configuration of the chat, including the output directory.
        step (str): The step number as a string to identify the specific output file.
        values (list, optional): A list of column names to select from the DataFrame. If None, all columns are returned.

    Returns:
        pandas.DataFrame: The DataFrame containing the data from the output file of the specified step. If specific columns are provided, only those columns are included.

    """
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: June 19, 2024
    log.debugLog(state, state=state)
    log.debugLog(step, state=state)
    step_output_files = []
    file = None
    for filename in os.listdir(state['output-directory']):
        if filename.startswith('S'):
            step_output_files.append(filename)
        if filename.startswith('S' + step):
            file = filename
    state = log.userOutput(file, state=state)
    if file.endswith('.csv'):
        file_path = os.path.join(state['output-directory'], file)
        df = pd.read_csv(file_path)
        state = log.userOutput(df, state=state)
        if values is not None:
            df = df[values]
    return df

def compile_latex_to_pdf(state, tex_file):
    """
    Compile a LaTeX (.tex) file into a PDF using pdflatex.

    This function compiles a LaTeX file into a PDF by running pdflatex command with the specified output directory.

    Args:
        state (dict): The dictionary containing the current status and configuration of the chat, including the output directory.
        tex_file (str): The filename of the LaTeX file (including the .tex extension) to compile.

    Returns:
        dict: Updated state dictionary after attempting to compile the LaTeX file.

    Raises:
        FileNotFoundError: If the specified LaTeX file does not exist.

    """
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: June 23, 2024
    tex_file_path = os.path.join(state['output-directory'], tex_file)
    
    # Ensure the file exists
    if not os.path.isfile(tex_file_path):
        raise FileNotFoundError(f"The file {tex_file_path} does not exist.")
    
    # Run the pdflatex command with the specified output directory
    try:
        subprocess.run(
            ['pdflatex', '-output-directory', state['output-directory'], tex_file_path], 
            check=True
        )
        log.debugLog(f"PDF generated successfully in {state['output-directory']}.", state=state)
        state['process']['steps'].append(
            {
                'func' : 'utils.compile_latex_to_pdf',
                'what' : 'tried to compile latex to a pdf'
            }
        )
    except subprocess.CalledProcessError as e:
        log.debugLog(f"An error occurred: {e}", state=state)
        state['process']['steps'].append(
            {
                'func' : 'utils.compile_latex_to_pdf',
                'what' : 'failed to compile latex to a pdf'
            }
        )        
    return state

def add_output_file_path_to_string(string, state):
    """
    Modifies the given string to include the appropriate file paths for any files 
    previously generated by BRAD. If a file from the generated files list is found 
    in the string, and it is not immediately preceded by the append path, the 
    function inserts the append path before the file name.

    Parameters:
        string (str): The input string to be modified.
        state (dict): A dictionary containing chat status information, including 
                           'output-path' and a function outputFiles that returns a list 
                           of generated file names.

    Returns:
        str: The modified string with appropriate file paths included.
    """
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: June 30, 2024
    
    # Retrieve the list of generated files and the output path
    generated_files = outputFiles(state)  # Returns a list of strings each indicating a file name
    append_path = state['output-directory']

    # Check and modify the string if necessary
    for file in generated_files:
        if file in string:
            fileWpath = os.path.join(append_path, file)
            if fileWpath not in string:
                string = string.replace(file, fileWpath)
                log.debugLog("Replacing: " + file + ' with ' + fileWpath, state=state)
                log.debugLog("New String: " + str(string), state=state)
    return string

def load_file_to_dataframe(filename):
    """
    Load a file into a Pandas DataFrame based on its extension.

    This function reads a CSV or TSV file into a Pandas DataFrame based on the file extension.

    Parameters
    ----------
        filename (str): The path to the file to load.

    Returns
    -------
        pd.DataFrame or None: The loaded DataFrame if successful, or None if the file extension is not supported.

    """
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: June 23, 2024

    # Determine the file extension
    _, file_extension = os.path.splitext(filename)
    
    if file_extension.lower() == '.csv':
        df = pd.read_csv(filename)
    elif file_extension.lower() == '.tsv':
        df = pd.read_csv(filename, delimiter='\t')
    else:
        return None
    
    return df

def find_integer_in_string(text):
    # Find all sequences of digits in the text
    match = re.search(r'\d+', text)
    
    if match:
        # Convert the found string to an integer
        return int(match.group(0))
    else:
        # Return None if no integer is found
        return None

def delete_dirs_without_log(agent):
    directory = agent.state['config'].get('log_path')
    # List only first-level subdirectories
    for subdir in os.listdir(directory):
        subdir_path = os.path.join(directory, subdir)
        
        # Check if it's a directory
        if os.path.isdir(subdir_path):
            log_file_path = os.path.join(subdir_path, 'log.json')
            
            # If log.json does not exist in the subdirectory, delete the subdirectory
            if not os.path.exists(log_file_path):
                shutil.rmtree(subdir_path)  # Recursively delete directory and its contents
                print(f"Deleted directory: {subdir_path}")
