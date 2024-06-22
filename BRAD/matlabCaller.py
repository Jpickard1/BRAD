"""
Module responsible for integrating MATLAB scripts execution into the BRAD system.

This module provides functions to execute MATLAB scripts based on user prompts and configuration settings. It interacts with the MATLAB engine, identifies available MATLAB functions, selects the appropriate function based on user input using a large language model (LLM), and executes the selected MATLAB code.

Notes:
- The module integrates with BRAD's chat functionality to execute MATLAB scripts based on user queries.
- It leverages the MATLAB engine through `matlab.engine` for script execution and path management.

**MATLAB Documentation Requirements**
1. they must have full docstrings at the top of the file
2. they must have a one line description at the top of the docstring used for selecting which code to run
"""

from langchain import PromptTemplate, LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
import difflib
import matlab.engine
import os
import glob
import re
from BRAD.promptTemplates import matlabPromptTemplate
from BRAD import log

def callMatlab(chatstatus):
    """
    Executes a MATLAB function based on the user's prompt and chat status configuration.

    This function performs the following steps:
    1. Activates the MATLAB engine and extends the MATLAB path.
    2. Identifies available MATLAB functions in the specified directory.
    3. Uses an llm to selects the MATLAB function that best matches the user's prompt and format code to execute the script based upon the users input.
    4. Executes the selected MATLAB function and updates the chat status.

    Parameters
    ----------
    chatstatus : dict
        A dictionary containing the chat status, including user prompt, language model (llm),
        memory, and configuration settings.

    Returns
    -------
    dict
        Updated chat status after executing the selected MATLAB function.

    Notes
    -----
    This function is called from `brad.chat()` when the `MATLAB` module is selected by the router. Additionally, MATLAB code can be executed from `coder.codeCaller()`, which follows a similar process to identify and run the appropriate scripts. However, `coder.codeCaller()` also has the capability to execute Python scripts.
    """
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: June 22, 2024
    print('Matlab Caller Start')
    prompt = chatstatus['prompt']                                        # Get the user prompt
    llm = chatstatus['llm']                                              # Get the llm
    memory = chatstatus['memory']
    chatstatus['process'] = {}                                           # Begin saving plotting arguments
    chatstatus['process']['name'] = 'Matlab'

    # Turn on matlab engine just before running the code
    chatstatus, mpath = activateMatlabEngine()
    print('Matlab PATH Extended') if chatstatus['config']['debug'] else None
    
    # Identify matlab files we are adding to the path of BRAD
    matlabFunctions = find_matlab_files(matlabPath)
    print(matlabFunctions) if chatstatus['config']['debug'] else None
    # Identify which file we should use
    matlabFunction = find_closest_function(prompt, matlabFunctions)
    print(matlabFunction) if chatstatus['config']['debug'] else None
    # Identify matlab arguments
    matlabFunctionPath = os.path.join(matlabPath, matlabFunction + '.m')
    print(matlabFunctionPath) if chatstatus['config']['debug'] else None
    # Get matlab docstrings
    matlabDocStrings = read_matlab_docstrings(matlabFunctionPath)
    print(matlabDocStrings) if chatstatus['config']['debug'] else None
    template = matlabPromptTemplate()
    print(template) if chatstatus['config']['debug'] else None
    # matlabDocStrings = callMatlab(chatstatus)
    filled_template = template.format(scriptName=matlabFunction, scriptDocumentation=matlabDocStrings) #, history=None, input=None)
    print(filled_template) if chatstatus['config']['debug'] else None
    PROMPT = PromptTemplate(input_variables=["history", "input"], template=filled_template)
    print(PROMPT) if chatstatus['config']['debug'] else None
    conversation = ConversationChain(prompt  = PROMPT,
                                     llm     =    llm,
                                     verbose =   chatstatus['config']['debug'],
                                     memory  = memory,
                                    )
    response = conversation.predict(input=chatstatus['prompt'] )
    print(response) if chatstatus['config']['debug'] else None
    matlabCode = extract_matlab_code(response, chatstatus)
    print(matlabCode) if chatstatus['config']['debug'] else None
    execute_matlab_code(matlabCode, chatstatus)
    return chatstatus

def activateMatlabEngine(chatstatus):
    """
    Activates the MATLAB engine and adds the specified MATLAB path to the engine.

    This function checks if the MATLAB engine is already active in the chat status. 
    If not, it starts the MATLAB engine. It then adds the specified MATLAB path 
    to the engine's search path.

    Parameters
    ----------
    chatstatus : dict
        A dictionary containing the chat status, including configuration settings 
        and the MATLAB engine instance.

    Returns
    -------
    tuple
        A tuple containing the updated chat status and the MATLAB path that was added 
        to the engine's search path.

    Notes
    -----
    - If the MATLAB engine is not active (`chatstatus['matlabEng']` is None), the 
      function starts it using `matlab.engine.start_matlab()`.
    - The MATLAB path is added using the `addpath` and `genpath` methods of the 
      MATLAB engine instance.
    """
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: June 22, 2024
    if chatstatus['matlabEng'] is None:
        chatstatus['matlabEng'] = matlab.engine.start_matlab()
    print('Matlab Engine On') if chatstatus['config']['debug'] else None
    # Add matlab files to the path
    if chatstatus['config']['matlab-path'] == 'matlab-tutorial/':
        matlabPath = chatstatus['config']['matlab-path']
    else:
        base_dir = os.path.expanduser('~')
        matlabPath = os.path.join(base_dir, chatstatus['config']['matlab-path'])
    mpath = chatstatus['matlabEng'].addpath(chatstatus['matlabEng'].genpath(matlabPath))
    return chatstatus, mpath

def execute_matlab_code(matlab_code, chatstatus):
    """
    Executes the given MATLAB code within the MATLAB engine context.

    This function attempts to evaluate and execute the provided MATLAB code
    using the active MATLAB engine. It logs debug information and handles
    potential exceptions that may occur during code execution.

    Parameters
    ----------
    matlab_code : str
        The MATLAB code to be executed.
    chatstatus : dict
        A dictionary containing the chat status, including configuration settings
        and the MATLAB engine instance.

    Returns
    -------
    None

    Notes
    -----
    - The function logs debug information about the MATLAB code execution process.
    - It uses the `eval` function to execute the MATLAB code and handles common exceptions
      such as `SyntaxError` and `NameError`, as well as any other generic exceptions.
    - The function assumes that the MATLAB engine instance is stored in `chatstatus['matlabEng']`.
    """
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: June 22, 2024
    log.debugLog('execute_matlab_code', chatstatus=chatstatus)
    log.debugLog(matlab_code, chatstatus=chatstatus)
    eng = chatstatus['matlabEng']
    if matlab_code:
        try:
            # Attempt to evaluate the MATLAB code
            eval(matlab_code)
            print("Debug: MATLAB code executed successfully.")
        except SyntaxError as se:
            print(f"Debug: Syntax error in the MATLAB code: {se}")
        except NameError as ne:
            print(f"Debug: Name error, possibly undefined function or variable: {ne}")
        except Exception as e:
            print(f"Debug: An error occurred during MATLAB code execution: {e}")
    else:
        print("Debug: No MATLAB code to execute.")


def find_matlab_files(path):
    """
    Recursively finds all MATLAB (.m) files in the specified directory path.

    This function searches for all MATLAB files in the given directory and its subdirectories,
    returning a list of the file names without their extensions.

    Parameters
    ----------
    path : str
        The directory path where the search for MATLAB files should begin.

    Returns
    -------
    list of str
        A list of MATLAB file names (without the .m extension) found in the specified directory.

    Notes
    -----
    - The function constructs a search pattern for .m files and uses the `glob` module to find
      all matching files recursively.
    - The file names are extracted from the full paths and the .m extension is removed.

    Example
    -------
    >>> path = '/path/to/matlab/scripts'
    >>> matlab_files = find_matlab_files(path)
    >>> print(matlab_files)
    ['script1', 'script2', 'subdir/script3']
    """
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: June 22, 2024
    
    # Construct the search pattern for .m files
    search_pattern = os.path.join(path, '**', '*.m')
    
    # Find all .m files recursively
    matlab_files = glob.glob(search_pattern, recursive=True)

    # Extract only the file names from the full paths
    file_names = [os.path.basename(file)[:-2] for file in matlab_files]
    
    return file_names

def find_closest_function(query, functions):
    """
    Finds the closest matching function name to the query from a list of functions.

    This function uses the `difflib.get_close_matches` method to find the closest match
    to the provided query within the list of function names. It returns the best match
    if any matches are found, otherwise it returns None.

    Parameters
    ----------
    query : str
        The query string to match against the list of functions.
    functions : list of str
        The list of function names to search within.

    Returns
    -------
    str or None
        The closest matching function name if a match is found; otherwise, None.

    Notes
    -----
    - The function uses `difflib.get_close_matches` with `n=1` to find the single closest match
      and `cutoff=0.0` to include all possible matches regardless of similarity.

    Example
    -------
    >>> functions = ['analyzeData', 'processImage', 'generateReport']
    >>> query = 'analyze'
    >>> closest_function = find_closest_function(query, functions)
    >>> print(closest_function)
    'analyzeData'
    """
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: June 22, 2024
    
    # Find the closest matches to the query in the list of functions
    closest_matches = difflib.get_close_matches(query, functions, n=1, cutoff=0.0)
    
    if closest_matches:
        return closest_matches[0]
    else:
        return None

def read_matlab_docstrings(file_path):
    """
    Reads the docstrings from a MATLAB (.m) file.

    This function extracts lines that start with the '%' character from the beginning 
    of the MATLAB file, which are typically used as comments or docstrings in MATLAB. 
    The reading stops when a line that doesn't start with '%' is encountered.

    Parameters
    ----------
    file_path : str
        The path to the MATLAB file from which to read the docstrings.

    Returns
    -------
    str
        A string containing the extracted docstrings, with each line separated by a newline character.

    Notes
    -----
    - The function assumes that docstrings are located at the beginning of the MATLAB file.
    - It stops reading further lines once a line that does not start with '%' is encountered.

    Example
    -------
    >>> docstrings = read_matlab_docstrings('/path/to/matlab/script.m')
    >>> print(docstrings)
    % This is a sample MATLAB script
    % It demonstrates how to read docstrings
    """
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: June 22, 2024
    docstrings = []
    first = True
    with open(file_path, 'r') as file:
        for line in file:
            # print(line)
            stripped_line = line.strip()
            if stripped_line.startswith('%') or first:
                # Remove the '%' character and any leading/trailing whitespace
                docstrings.append(stripped_line)
                first = False
            else:
                # Stop reading if we encounter a line that doesn't start with '%'
                # assuming docstrings are at the beginning of the file
                break
    return "\n".join(docstrings)

def get_matlab_description(file_path):
    """
    Extracts a brief description from the docstrings of a MATLAB (.m) file.

    This function reads the docstrings from the given MATLAB file and returns a one-liner 
    description extracted from the second line of the docstrings.

    Parameters
    ----------
    file_path : str
        The path to the MATLAB file from which to extract the description.

    Returns
    -------
    str
        A one-liner description extracted from the second line of the docstrings.

    Notes
    -----
    - The function assumes that the second line of the docstrings contains a brief description 
      of the MATLAB script.
    - The `read_matlab_docstrings` function is used to read the docstrings from the file.

    Example
    -------
    >>> description = get_matlab_description('/path/to/matlab/script.m')
    >>> print(description)
    ' This script demonstrates how to read docstrings from a MATLAB file.'
    """
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: June 22, 2024
    docstrings = read_matlab_docstrings(file_path)
    oneliner = docstrings.split('\n')[1][1:]
    return oneliner

def extract_matlab_code(llm_output, chatstatus):
    """
    Extracts the MATLAB code to be executed from the LLM (Large Language Model) output.

    This function processes the output from the LLM to extract the MATLAB code that needs 
    to be executed. It handles debugging output and adjusts the function call format as needed.

    Parameters
    ----------
    llm_output : str
        The output from the Large Language Model (LLM) containing the MATLAB code to be executed.
    chatstatus : dict
        A dictionary containing the chat status, including configuration settings.

    Returns
    -------
    str
        The extracted MATLAB function call or script to be executed.

    Notes
    -----
    - The function assumes that the LLM output contains the MATLAB code prefixed with 'Execute:'.
    - It adjusts the function call format if necessary to ensure compatibility with the MATLAB engine.
    - Debug messages are printed if `chatstatus['config']['debug']` is True.

    Example
    -------
    >>> llm_output = "Execute: myFunction(param1, param2)"
    >>> chatstatus = {'config': {'debug': True}}
    >>> matlab_code = extract_matlab_code(llm_output, chatstatus)
    LLM OUTPUT PARSER
    Execute: myFunction(param1, param2)
    ['Execute:', ' myFunction(param1, param2)']
    'eng myFunction(param1, param2)'
    """
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: June 22, 2024
    print('LLM OUTPUT PARSER') if chatstatus['config']['debug'] else None
    print(llm_output) if chatstatus['config']['debug'] else None
    funcCall = llm_output.split('Execute:')
    print(funcCall) if chatstatus['config']['debug'] else None
    funcCall = funcCall[len(funcCall)-1].strip()
    if funcCall[:3] != 'eng':
        print(funcCall) if chatstatus['config']['debug'] else None
        funcCall = 'eng' + funcCall[3:]
        print(funcCall) if chatstatus['config']['debug'] else None
    return funcCall


def callMatlab_depricated(chatstatus, chatlog):
    """
    .. note:: Deprecated soon
    
    THIS IS THE INCORRECT DOCUMENTATION FOR THIS FUNCTION IT IS ONLY A TEST DELETE LATER
    IT DOES NOT WORK... WHY?
    Performs a search on Gene Ontology (GO) based on the provided query and allows downloading associated charts and papers.

    :param query: The query list containing gene names or terms for GO search.
    :type query: list

    :return: A dictionary containing the GO search process details.
    :rtype: dict
    """
    prompt = chatstatus['prompt']                                        # Get the user prompt
    chatstatus['process'] = {}                                           # Begin saving plotting arguments
    chatstatus['process']['name'] = 'Matlab'    
    config_file_path = 'configMatlab.json' # we could use this to add matlab files to path
    eng = matlab.engine.start_matlab()


