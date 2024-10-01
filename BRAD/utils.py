"""
Module: utils

This module provides a collection of utility functions that facilitate common tasks for large language models (LLMs) 
across various modules. These functions are designed to simplify file management, data handling, and directory 
operations, enabling LLMs to perform essential tasks efficiently.

Key Functions:
--------------
1. save(data, path): Saves the provided data to a specified file path.
2. savefig(fig, path): Saves a given figure to a specified file path in a suitable format.
3. ensure_directory_exists(directory): Checks if a directory exists; if not, it creates the directory.
4. pdfDownloadPath(base_path): Generates a standard download path for PDF files based on the base path.
5. getOutputFiles(directory): Retrieves a list of output files from the specified directory.
6. makeNamesConsistent(names): Normalizes and standardizes a list of names for consistency.
7. loadFile(path): Loads data from a specified file path and returns the content.
8. fileFieldSelector(data, field): Selects specific fields from a data structure based on the provided field name.
9. (additional utility functions as needed...)

Usage:
------
These utility functions are intended to be imported and used in various modules that require basic functionalities 
related to file and data handling. By centralizing these common tasks, the codebase remains organized and efficient.
"""

import re
import os
import time
import numpy as np
import pandas as pd
import subprocess
import difflib
import matplotlib.pyplot as plt

from langchain import PromptTemplate, LLMChain
from langchain_community.callbacks import get_openai_callback

from BRAD import log
from BRAD.promptTemplates import fileChooserTemplate, fieldChooserTemplate

def save(chatstatus, data, name):
    """
    Save data to a specified output directory, with optional stage number prefix.

    This function saves the provided data to a specified output directory within 
    the `chatstatus` configuration. If the `chatstatus` is part of a pipeline, 
    it prefixes the filename with the current stage number.

    Args:
        chatstatus (dict): A dictionary containing the current chat status, 
                           including queued pipeline stages and output directory.
        data (pd.DataFrame or str): The data to be saved. It can be either a 
                                    pandas DataFrame (for CSV output) or a string (for .tex output).
        name (str): The name of the output file.

    Returns:
        dict: The updated `chatstatus` dictionary with information about the saved file.

    Raises:
        ValueError: If the data type is not a DataFrame for CSV or a string for .tex files.

    Example
    -------
    >>> chatstatus = {
    >>>     'queue': [{'order': 1}],
    >>>     'output-directory': '/path/to/output',
    >>>     'process': {'steps': []}
    >>> }
    >>> data = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    >>> save(chatstatus, data, 'results.csv')
    """
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: June 19, 2024

    # If this is part of a pipeline, then add the stage number to the printed output
    if len(chatstatus['queue']) != 0:
        stageNum = chatstatus['queue pointer'] + 1#[0]['order']
        name = 'S' + str(stageNum) + '-' + name
    output_path = os.path.join(chatstatus['output-directory'], name)

    if isinstance(data, pd.DataFrame):
        data.to_csv(output_path, index=False)
    elif output_path.endswith('.tex'):
        with open(output_path, 'w') as file:
            file.write(data)
    else:
        raise ValueError("Unsupported data type or file extension. Use a DataFrame for CSV or a string for .tex files.")
    
    log.debugLog('The information has been saved to: ' + output_path, chatstatus=chatstatus)
    chatstatus['process']['steps'].append(
        {
            'func'     : 'utils.save',
            'new file' : output_path
        }
    )
    return chatstatus

def savefig(chatstatus, ax, name):
    """
    Save a matplotlib figure to a specified output directory, with optional stage number prefix.

    This function saves the provided matplotlib axis (`ax`) as a figure to a specified 
    output directory within the `chatstatus` configuration. If the `chatstatus` is part 
    of a pipeline, it prefixes the filename with the current stage number.

    Args:
        chatstatus (dict): A dictionary containing the current chat status, including 
                           queued pipeline stages and output directory.
        ax (matplotlib.axes.Axes): The matplotlib axis object containing the figure to be saved.
        name (str): The name of the output file.

    Returns:
        dict: The updated `chatstatus` dictionary with information about the saved file.

    Example
    -------
    >>> chatstatus = {
    >>>     'queue': [{'order': 1}],
    >>>     'output-directory': '/path/to/output',
    >>>     'config': {'image-path-extension': 'images'},
    >>>     'process': {'steps': []}
    >>> }
    >>> fig, ax = plt.subplots()
    >>> savefig(chatstatus, ax, 'figure.png')
    """
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: June 19, 2024
    log.debugLog("SAVEFIG", chatstatus=chatstatus)
    if len(chatstatus['queue']) != 0:
        stageNum = chatstatus['queue pointer'] + 1 # [0]['order']
        name = 'S' + str(stageNum) + '-' + name
    output_path = os.path.join(chatstatus['output-directory'], chatstatus['config']['image-path-extension'], name)
    ensure_directory_exists(output_path, chatstatus)
    plt.savefig(output_path)
    log.debugLog('The image was saved to: ' + output_path, chatstatus=chatstatus)
    chatstatus['process']['steps'].append(
        {
            'func'     : 'utils.savefig',
            'new file' : output_path
        }
    )
    return chatstatus

def ensure_directory_exists(file_path, chatstatus):
    """
    Ensure that the directory for a given file path exists, creating it if necessary.

    This function checks if the directory path for the provided `file_path` exists.
    If the directory does not exist, it creates the directory. It prints a message
    indicating whether the directory was created or if it already existed.

    Args:
        file_path (str): The full file path for which the directory needs to be checked/created.

    Example
    -------
    >>> ensure_directory_exists('/path/to/output/figure.png', chatstatus)
    >>> # If the directory '/path/to/output' does not exist, it will be created.
    """
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: June 23, 2024
    directory_path = os.path.dirname(file_path)
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        log.debugLog(f"Directory '{directory_path}' created.", chatstatus=chatstatus)
    else:
        log.debugLog(f"Directory '{directory_path}' already exists.", chatstatus=chatstatus)


def pdfDownloadPath(chatstatus):
    """
    Generate the file path for downloading PDF files.

    This function constructs the file path for downloading PDF files based on the 
    `output-directory` specified in the `chatstatus` dictionary. It appends 'pdf'
    to the output directory path to indicate the location where PDF files should be saved.

    Args:
        chatstatus (dict): A dictionary containing the chat status and configuration details.
                           It must include the key 'output-directory'.

    Returns:
        str: The complete file path for downloading PDF files.

    Example
    -------
    >>> chatstatus = {'output-directory': '/path/to/output'}
    >>> pdf_path = pdfDownloadPath(chatstatus)
    >>> # pdf_path will be '/path/to/output/pdf'
    """
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: June 19, 2024
    path = os.path.join(chatstatus['output-directory'], 'pdf')
    return path

def outputFiles(chatstatus):
    """
    Retrieve a list of all files in the output directory.

    This function lists all files present in the `output-directory` specified in the 
    `chatstatus` dictionary and returns them as a list.

    Args:
        chatstatus (dict): A dictionary containing the chat status and configuration details.
                           It must include the key 'output-directory'.

    Returns:
        list: A list of filenames present in the output directory.

    Example
    -------
    >>> chatstatus = {
    >>>     'output-directory': '/path/to/output'
    >>> }
    >>> files = outputFiles(chatstatus)
    >>> # files will be a list of filenames in '/path/to/output'
    """
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: June 19, 2024
    output_files = []
    for filename in os.listdir(chatstatus['output-directory']):
        output_files.append(filename)
    return output_files

def makeNamesConsistent(chatstatus, files):
    """
    Ensure filenames in the output directory are consistent with the pipeline stage numbering.

    This function renames files in the output directory to include the current stage number
    from the pipeline. If a file's name does not start with 'S', it will be prefixed with the 
    stage number. Additionally, it removes any '/' or '\\' characters from filenames.

    Args:
        chatstatus (dict): A dictionary containing the chat status and configuration details.
                           It must include the keys 'queue' and 'output-directory'.
        files (list): A list of filenames to be processed.

    Returns:
        dict: Updated chatstatus with renamed files logged in 'process' steps.

    Example
    -------
    >>> chatstatus = {
    >>>     'queue': [{'order': 1}],
    >>>     'output-directory': '/path/to/output',
    >>>     'process': {'steps': []}
    >>> }
    >>> files = ['file1.txt', 'file2.txt']
    >>> updated_chatstatus = makeNamesConsistent(chatstatus, files)
    >>> # Files will be renamed to include the stage number and logged in chatstatus['process']['steps']
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
    if len(chatstatus['queue']) != 0:
        log.debugLog('Finding Stage Number of Pipeline', chatstatus=chatstatus)
        log.debugLog(chatstatus['queue'], chatstatus=chatstatus)
        IP = chatstatus['queue pointer'] # [0]['order'] + 1
        IP = int(IP)
    else:
        return
    renamedFiles = []
    log.debugLog(f"{IP=}", chatstatus=chatstatus)
    log.debugLog(f"{type(IP)=}", chatstatus=chatstatus)
    for file in files:
        if file[0] != 'S':
            old_path = os.path.join(chatstatus['output-directory'], file)
            if os.path.isdir(old_path):
                continue
            new_path = os.path.join(chatstatus['output-directory'], 'S' + str(IP) + '-' + file)
            renamedFiles.append(
                {
                    'old-name' : old_path,
                    'new-name' : new_path
                }
            )
            os.rename(old_path, new_path)
            if 'output' not in chatstatus['queue'][IP].keys():
                chatstatus['queue'][IP] = []
            chatstatus['queue'][IP]['output'].append(new_path)
    for file in outputFiles(chatstatus):
        old_path = os.path.join(chatstatus['output-directory'], file)
        new_path = os.path.join(chatstatus['output-directory'], file.replace('/', '').replace('\\', ''))
        if old_path != new_path:
            renamedFiles.append(
                {
                    'old-name' : old_path,
                    'new-name' : new_path
                }
            )
            os.rename(old_path, new_path)
            if 'output' not in chatstatus['queue'][IP].keys():
                chatstatus['queue'][IP] = []
            chatstatus['queue'][IP]['output'].append(new_path)
    chatstatus['process']['steps'].append(
        {
            'func'  : 'utils.makeNamesConsistent',
            'files' : renamedFiles
        }
    )
    return chatstatus

def loadFromFile(chatstatus):
    """
    Loads data from a file selected by an LLM prompt based on user input.

    This function interacts with a language model to select a file from available files
    in the output directory. It extracts the specified fields from the selected file 
    and returns the data along with updated chat status.

    Args:
        chatstatus (dict): A dictionary containing the chat status and configuration details.
                           It must include the keys 'prompt', 'llm', and 'output-directory'.

    Returns:
        tuple: Updated chatstatus dictionary and a list of values from the specified fields in the file.

    Example
    -------
    >>> chatstatus = {
    >>>     'prompt': 'Choose a file containing gene expression data.',
    >>>     'llm': your_language_model_instance,
    >>>     'output-directory': '/path/to/output',
    >>>     'process': {'steps': []}
    >>> }
    >>> updated_chatstatus, field_values = loadFromFile(chatstatus)
    """
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: June 19, 2024
    prompt = chatstatus['prompt']
    llm    = chatstatus['llm']
    # Get files to choose from
    availableFilesList = outputFiles(chatstatus)
    availableFiles = '\n'.join(availableFilesList)
    log.debugLog(availableFiles, chatstatus=chatstatus)
    
    # Build lang chain
    template = fileChooserTemplate()
    template = template.format(files=availableFiles)
    log.debugLog(template, chatstatus=chatstatus)
    PROMPT   = PromptTemplate(input_variables=["user_query"], template=template)
    chain    = PROMPT | llm

    # Call chain
    chatstatus   = log.userOutput(prompt, chatstatus=chatstatus)
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
    
    log.debugLog('File=' + str(file) + '\n' + 'Fields=' + str(fields), chatstatus=chatstatus)
    chatstatus['process']['steps'].append(
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
    loadfile = os.path.join(chatstatus['output-directory'], file)
    df = pd.read_csv(loadfile, delimiter=delimiter)
    chatstatus['process']['steps'].append(log.loadFileLog(file      = loadfile,
                                                          delimiter = delimiter)
                                         )

    if fields not in df.columns:
        chatstatus, fields = fieldSelectorFromDataFrame(chatstatus, df)

    return chatstatus, list(df[fields].values)

def fieldSelectorFromDataFrame(chatstatus, df):
    """
    Selects a field from a DataFrame using a language model prompt.

    This function uses a language model to select a specific field from the columns of a given DataFrame.
    It builds a prompt with the available columns, invokes the language model, and parses the response to
    determine the selected field.

    Args:
        chatstatus (dict): A dictionary containing the chat status and configuration details.
                           It must include the keys 'llm', 'prompt', and 'process'.
        df (pandas.DataFrame): The DataFrame from which a field will be selected.

    Returns:
        tuple: Updated chatstatus dictionary and the selected field as a string.

    Example
    -------
    >>> chatstatus = {
    >>>     'prompt': 'Choose a column for analysis.',
    >>>     'llm': your_language_model_instance,
    >>>     'process': {'steps': []}
    >>> }
    >>> df = pd.DataFrame({'Gene': ['BRCA1', 'TP53'], 'Expression': [5.6, 8.2]})
    >>> updated_chatstatus, selected_field = fieldSelectorFromDataFrame(chatstatus, df)
    """
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: June 19, 2024
    llm      = chatstatus['llm']
    prompt   = chatstatus['prompt']
    template = fieldChooserTemplate()
    template = template.format(columns=', '.join(list(df.columns)))
    PROMPT   = PromptTemplate(input_variables=["user_query"], template=template)
    chain    = PROMPT | llm

    # Call chain
    response = chain.invoke(prompt).content.strip()
    fields = response.split('=')[1].strip()
    chatstatus['process']['steps'].append(log.llmCallLog(llm          = llm,
                                                         prompt       = PROMPT,
                                                         input        = prompt,
                                                         output       = response,
                                                         parsedOutput = {
                                                             'Fields' : fields
                                                         },
                                                         purpose      = 'Select Field'
                                                        )
                                        )

    log.debugLog('field identifier response=\n'+fields, chatstatus=chatstatus)
    return chatstatus, fields

def word_similarity(word1, word2):
    """
    Calculate the similarity ratio between two words using SequenceMatcher.

    This function computes the similarity ratio between two input words. The ratio is calculated
    based on the longest contiguous matching subsequence between the two words using the
    `difflib.SequenceMatcher` from the Python standard library.

    Args:
        word1 (str): The first word to compare.
        word2 (str): The second word to compare.

    Returns:
        float: A float value between 0 and 1 representing the similarity ratio. A value of 1.0 means the words
               are identical, while 0.0 means they are completely different.

    Example
    -------
    >>> similarity = word_similarity("apple", "apples")
    >>> print(similarity)  # Output might be a high value close to 1.0
    """
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: June 23, 2024
    return difflib.SequenceMatcher(None, word1, word2).ratio()

def outputFromPriorStep(chatstatus, step, values=None):
    """
    Retrieve the output from a prior step in the pipeline.

    .. warning:: We may be removing this function soon.

    This function searches for and loads the output file corresponding to a specified step in the pipeline.
    If the file is a CSV, it loads the data into a DataFrame. Optionally, specific columns can be selected from the DataFrame.

    Args:
        chatstatus (dict): The dictionary containing the current status and configuration of the chat, including the output directory.
        step (str): The step number as a string to identify the specific output file.
        values (list, optional): A list of column names to select from the DataFrame. If None, all columns are returned.

    Returns:
        pandas.DataFrame: The DataFrame containing the data from the output file of the specified step. If specific columns are provided, only those columns are included.

    Example
    -------
    >>> df = outputFromPriorStep(chatstatus, "2", values=["gene", "expression"])
    >>> print(df)
    """
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: June 19, 2024
    log.debugLog(chatstatus, chatstatus=chatstatus)
    log.debugLog(step, chatstatus=chatstatus)
    step_output_files = []
    file = None
    for filename in os.listdir(chatstatus['output-directory']):
        if filename.startswith('S'):
            step_output_files.append(filename)
        if filename.startswith('S' + step):
            file = filename
    chatstatus = log.userOutput(file, chatstatus=chatstatus)
    if file.endswith('.csv'):
        file_path = os.path.join(chatstatus['output-directory'], file)
        df = pd.read_csv(file_path)
        chatstatus = log.userOutput(df, chatstatus=chatstatus)
        if values is not None:
            df = df[values]
    return df

def compile_latex_to_pdf(chatstatus, tex_file):
    """
    Compile a LaTeX (.tex) file into a PDF using pdflatex.

    This function compiles a LaTeX file into a PDF by running pdflatex command with the specified output directory.

    Args:
        chatstatus (dict): The dictionary containing the current status and configuration of the chat, including the output directory.
        tex_file (str): The filename of the LaTeX file (including the .tex extension) to compile.

    Returns:
        dict: Updated chatstatus dictionary after attempting to compile the LaTeX file.

    Raises:
        FileNotFoundError: If the specified LaTeX file does not exist.

    Example
    -------
    >>> chatstatus = {'output-directory': '/path/to/output'}
    >>> chatstatus = compile_latex_to_pdf(chatstatus, 'report.tex')
    """
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: June 23, 2024
    tex_file_path = os.path.join(chatstatus['output-directory'], tex_file)
    
    # Ensure the file exists
    if not os.path.isfile(tex_file_path):
        raise FileNotFoundError(f"The file {tex_file_path} does not exist.")
    
    # Run the pdflatex command with the specified output directory
    try:
        subprocess.run(
            ['pdflatex', '-output-directory', chatstatus['output-directory'], tex_file_path], 
            check=True
        )
        log.debugLog(f"PDF generated successfully in {chatstatus['output-directory']}.", chatstatus=chatstatus)
        chatstatus['process']['steps'].append(
            {
                'func' : 'utils.compile_latex_to_pdf',
                'what' : 'tried to compile latex to a pdf'
            }
        )
    except subprocess.CalledProcessError as e:
        log.debugLog(f"An error occurred: {e}", chatstatus=chatstatus)
        chatstatus['process']['steps'].append(
            {
                'func' : 'utils.compile_latex_to_pdf',
                'what' : 'failed to compile latex to a pdf'
            }
        )        
    return chatstatus

def add_output_file_path_to_string(string, chatstatus):
    """
    Modifies the given string to include the appropriate file paths for any files 
    previously generated by BRAD. If a file from the generated files list is found 
    in the string, and it is not immediately preceded by the append path, the 
    function inserts the append path before the file name.

    Parameters:
    string (str): The input string to be modified.
    chatstatus (dict): A dictionary containing chat status information, including 
                       'output-path' and a function outputFiles that returns a list 
                       of generated file names.

    Returns:
    str: The modified string with appropriate file paths included.
    """
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: June 30, 2024
    
    # Retrieve the list of generated files and the output path
    generated_files = outputFiles(chatstatus)  # Returns a list of strings each indicating a file name
    append_path = chatstatus['output-directory']

    # Check and modify the string if necessary
    for file in generated_files:
        if file in string:
            fileWpath = os.path.join(append_path, file)
            if fileWpath not in string:
                string = string.replace(file, fileWpath)
                log.debugLog("Replacing: " + file + ' with ' + fileWpath, chatstatus=chatstatus)
                log.debugLog("New String: " + str(string), chatstatus=chatstatus)
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

    Example
    -------
    >>> df = load_file_to_dataframe('data.csv')
    >>> if df is not None:
    >>>     print(df.head())
    >>> else:
    >>>     print("Unsupported file format.")
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

