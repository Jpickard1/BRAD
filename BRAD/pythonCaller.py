"""
Python Codes
------------

This module is responsible for integrating python scripts execution into the BRAD system. It selects the appropriate function based on user input using a LLM and runs the selected python code.

Python Documentation Requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In order to run a python script, the following documentation requirements must be satisfied.

1. doctrings:
    complete docstrings detailing both (1) what the purpose of the script is and (2) how to call or run the script from the command line or in python must be provided at the top of each file

2. one-liners:
    one line descriptions must be present at the top of each file for the purpose of selecting which script best suits a users request.

These docstings found within the python files will be used by the LLM of an `Agent` to select and formulate the required code to call the appropriate script. Since these documentation are input to an LLM, classic prompting techniques such as chain of thought prompting and few-shot learning can be implemented by writing longer and more detailed documentation. The following example illustrates what these docstrings may look like for a script that allows BRAD to use the `scanpy` library.

>>> \"\"\"
... This script executes a series of scanpy commands on an AnnData object loaded from a .h5ad file and saves the resulting AnnData object back to disk. 
... 
... Arguments (four arguments):
...     1. output directory: state['output-directory']
...     2. output file: <name of output file>
...     3. input file: <file created in previous step>
...     4. scanpy commands: a list of scanpy commands to be executed on the AnnData object (provided as a single string with commands separated by a delimiter, e.g., ';')
... 
... Based on the arguments, the input file will be loaded, then your commands will be executed, and finally, the output or resulting ann data object will be saved to the corrrect output file and directory. Your code is not responsible for loading the .h5ad object, that will happen automatically, and when loaded, the object will be called adata. Your scanpy commands can operate directly on the adata object that will be loaded for you.
... 
... Additionally, the following imports are already provided for you and can be used in your code:
... ```
... import scanpy as sc
... import seaborn as sns
... import matplotlib.pyplot as plt
... import os
... ```
... 
... **Usage**
... BRAD Line:
... ```
... subprocess.run([sys.executable, "<path/to/script/>/scanpy_brad.py", state['output-directory'], <output file>, <input file>, "<scanpy commands>"], capture_output=True, text=True)
... ```
... 
... **Examples**
... Use the below examples to help generate your code.
... 
... *Example 1*
... User Prompt: Run scanpy preprocessing and UMAP visualization on XXX.h5ad and save the UMAP plot
... Response Code:
... ```
... response = subprocess.run([sys.executable, "<path/to/script/>/scanpy_brad.py", state['output-directory'], "XXX-modified.h5ad", "<path/to/data>/XXX.h5ad", "sc.pp.neighbors(adata); sc.tl.umap(adata); sc.pl.umap(adata, save='umap.png')"], capture_output=True, text=True)
... ```
... Explination: the adata object will be loaded in memory already. The command "sc.pp.neighbors(adata)" will preprocess the data, then the command "sc.tl.umap(adata)" will perform UMAP, and finally the command "sc.pl.umap(adata, 'umap.png')" will save the UMAP to a well named file.
... 
... *Example 2*
... User Prompt: Perform PCA and clustering on the dataset YYY.h5ad and save the PCA plot
... Response Code:
... ```
... response = subprocess.run([sys.executable, "<path/to/script/>/scanpy_brad.py", state['output-directory'], "YYY-modified.h5ad", "<path/to/data>/YYY.h5ad", "sc.pp.pca(adata); sc.tl.leiden(adata); sc.pl.pca(adata, save='pca.png')"], capture_output=True, text=True)
... ```
... Explination: the adata object will be loaded in memory already. The command "sc.pp.pca(adata)" will preprocess the data, then the command "sc.tl.leiden(adata)" will perform the leiden algorithm, and finally the command "sc.pl.pca(adata, save='pca.png')" will save the PCA to a well named file.
... 
... **OUTPUT FILE NAME INSTRUCTIONS**
... 1. Output path should be state['output-directory']
... 2. Output file name should be `<descriptive name>.h5ad`
<<< \"\"\"

The above example uses the following methods to help the LLM select and call this script:

(1) a concise, detailed description of the purpose of the file is provided
(2) a list of the 4 arguments required by the script is clearly stated
(3) a paragraph describing what the script does along with a list of the available imports to use
(4) a template example is provided for how to call the script
(5) two examples of how to use the script are provided (few-shot principle)
(6) explinations of how to run the script are provided (chain-of-thought principle)

This documentation is used by BRAD to run the following script:

>>> import argparse
... import scanpy as sc
... import seaborn as sns
... import matplotlib.pyplot as plt
... import os
... 
... def main(output_directory, output_file, input_file, scanpy_commands):
... 
...     state = {
...         'output-directory': output_directory
...     }
...     sc.settings.figdir = output_directory
...     sc.set_figure_params(dpi=300)
...     
...     # Load the h5ad file using scanpy
...     adata = sc.read_h5ad(input_file)
...     print(f'adata loaded from {input_file}')
...     print(f'adata.shape={adata.shape}')
...     print(f'adata.obs.columns={adata.obs.columns}')
...     print(f'adata.obs.head()={adata.obs.head()}')
...     print(f'adata.var.head()={adata.var.head()}')
... 
...     # Deserialize the scanpy commands
...     commands = scanpy_commands.split(';')
... 
...     # Execute the list of scanpy commands in memory
...     print("****************************")
...     print("      EXECUTE COMMANDS      ")
...     print("****************************")
...     for command in commands:
...         command = command.strip()
...         print(command)
...         exec(command)
... 
...     if not os.path.exists(output_directory):
...         os.makedirs(output_directory)
...     output_path = os.path.join(output_directory, output_file)
...     adata.write(output_path)
... 
... if __name__ == "__main__":
...     parser = argparse.ArgumentParser(description='Execute scanpy commands on an AnnData object.')
...     parser.add_argument('output_directory', type=str, help='The output directory.')
...     parser.add_argument('output_file', type=str, help='The output file name.')
...     parser.add_argument('input_file', type=str, help='The input file name.')
...     parser.add_argument('scanpy_commands', type=str, help='The scanpy commands to be executed, separated by semicolons.')
...     args = parser.parse_args()
>>>     main(args.output_directory, args.output_file, args.input_file, args.scanpy_commands)

Available Methods
~~~~~~~~~~~~~~~~~

This module has the following methods:

"""

from langchain import PromptTemplate, LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
import difflib
# import matlab.engine
import os
import glob
import re
import sys
import subprocess

from BRAD.promptTemplates import pythonPromptTemplate, getPythonEditingTemplate
from BRAD import log

def callPython(state):
    """
    Executes a Python script based on the user's prompt and chat status configuration.

    This function performs the following steps:
    1. Identifies available python scripts in the specified directory.
    2. Uses an llm to select the python function that best matches the user's prompt and format code to execute the script based upon the users input.
    3. Executes the selected python function and updates the chat status.

    :param state: A dictionary containing the chat status, including user prompt, language model (LLM),
                       memory, and configuration settings.
    :type state: dict
    
    :returns: Updated chat status after executing the selected Python script.
    :rtype: dict
    
    :notes: 
        - This function is typically called from `brad.chat()` when the `Python` module is selected by the router.
        - The Python code execution can also be initiated from `coder.code_caller()`, which handles both Python and MATLAB scripts.
    """
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: June 22, 2024
    
    # Dev. Comments:
    # -------------------
    # This function is responsible for selecting a matlab function to call and
    # furmulating the function call.
    #
    # History:
    #
    # Issues:
    # - This function doesn't use the llm to select the appropriate python script.
    #   Currently, a word similarity score between the prompt and matlab codes
    #   is performed and used to select the prompts, but we could follow an approach
    #   similar to the coder.code_caller() method that uses an llm to read the docstrings
    #   and identify the best file. Note - the same approach is used by matlabCaller
    
    log.debugLog("Python Caller Start", state=state) 
    prompt = state['prompt']                                        # Get the user prompt
    llm = state['llm']                                              # Get the llm
    memory = state['memory']
    state['process'] = {}                                           # Begin saving plotting arguments
    state['process']['name'] = 'Python'

    # Add matlab files to the path
    if state['config']['py-path'] == 'py-tutorial/': # Admittedly, I'm not sure what's going on here - JP
        pyPath = state['config']['py-path']
    else:
        base_dir = os.path.expanduser('~')
        pyPath = os.path.join(base_dir, state['config']['py-path'])
    log.debugLog('Python scripts added to PATH', state=state) 

    # Identify python scripts files we are adding to the path of BRAD
    pyScripts = find_py_files(pyPath)
    log.debugLog(pyScripts, state=state) 
    # Identify which file we should use
    pyScript = find_closest_function(prompt, pyScripts)
    log.debugLog(pyScript, state=state) 
    # Identify pyton script arguments
    pyScriptPath = os.path.join(pyPath, pyScript + '.py')
    log.debugLog(pyScriptPath, state=state) 
    # Get matlab docstrings
    pyScriptDocStrings = read_python_docstrings(pyScriptPath)
    log.debugLog(pyScriptDocStrings, state=state) 
    template = pythonPromptTemplate()
    log.debugLog(template, state=state) 
    # matlabDocStrings = callMatlab(state)
    filled_template = template.format(scriptName=pyScriptPath, scriptDocumentation=pyScriptDocStrings)
    state = log.userOutput(filled_template, state=state)
    PROMPT = PromptTemplate(input_variables=["history", "input"], template=filled_template)
    log.debugLog(PROMPT, state=state) 
    conversation = ConversationChain(prompt  = PROMPT,
                                     llm     = llm,
                                     verbose = state['config']['debug'],
                                     memory  = memory,
                                    )
    response = conversation.predict(input=state['prompt'] )
    log.debugLog('START RESPONSE', state=state) 
    log.debugLog(response, state=state) 
    log.debugLog("END RESPONSE", state=state) 
    pythonCode = extract_python_code(response, state['config']['py-path'], state)
    log.debugLog(matlabCode, state=state)
    state['process']['steps'].append(log.llmCallLog(llm             = llm,
                                                         prompt          = PROMPT,
                                                         input           = state['prompt'],
                                                         output          = response,
                                                         parsedOutput    = {
                                                             'pythonCode': pythonCode,
                                                         },
                                                         purpose         = 'formulate python function call'
                                                        )
                                         )
    execute_python_code(pythonCode, state)
    return state

def execute_python_code(python_code, state):
    """
    Executes the provided Python code and handles debug printing based on the chat status configuration.

    This function performs the following steps:
    
    1. Checks if `state['output-directory']` is referenced in the provided Python code; if not, replaces a specific
       part of the code with `state['output-directory']`.
    2. Evaluates Python code using `eval()`.
    
    :param python_code: The Python code to be executed.
    :type python_code: str
    :param state: A dictionary containing the chat status, including configuration settings and possibly `state['output-directory']`.
    :type state: dict

    :notes:
        - This function is typically called after extracting Python code from a response in a conversational context but it can be called from the `coder.code_caller()` as well.
        - It assumes the presence of `eval()`-compatible Python code and handles basic error handling.
    """
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: June 22, 2024
    log.debugLog('EVAL', state=state) 
    log.debugLog(python_code, state=state) 
    log.debugLog('END EVAL CHECK', state=state) 

    # Ensure that state['output-directory'] is passed as an argument to the code:
    if "state['output-directory']" not in python_code:
        log.debugLog('PYTHON CODE OUTPUT DIRECTORY CHANGED', state=state) 
        python_code = python_code.replace(get_arguments_from_code(python_code)[2].strip(), "state['output-directory']")
        log.debugLog(python_code, state=state) 

    # Ensure that we get the output from the script
    response = None
    python_code = "response = " + python_code
    
    try:
        # Attempt to evaluate the MATLAB code
        log.debugLog("Finalized PYTHON Call:", state=state)
        log.debugLog("   ", state=state)
        log.debugLog(python_code, state=state)
        log.debugLog("   ", state=state)

        # execute the code and get the response variable
        local_scope = {'state': state, 'sys': sys, 'subprocess': subprocess}
        exec(python_code, globals(), local_scope)
        log.debugLog("Debug: PYTHON code executed successfully.", state=state)
        response = local_scope.get('response', None)
        log.debugLog("Code Execution Output:", state=state)
        log.debugLog(response, state=state)
        state['process']['steps'].append(
            {
                'func'   : 'pythonCaller.execute_python_code',
                'code'   : python_code,
                'purpose': 'execute python code',
            }
        )
        state['output'] = response.stdout.strip()
        log.debugLog("Debug: PYTHON code output saved to output.", state=state)

        # TODO: If the response contains a stderr then we should allow it the system to
        # debug and reexecute the code
    
    except SyntaxError as se:
        log.debugLog(f"Debug: Syntax error in the PYTHON code: {se}", state=state)
    except NameError as ne:
        log.debugLog(f"Debug: Name error, possibly undefined function or variable: {ne}", state=state)
    except Exception as e:
        log.debugLog(f"Debug: An error occurred during PYTHON code execution: {e}", state=state)


# Extract the arguments from the string
def get_arguments_from_code(code):
    """
    Extracts and returns arguments from a given Python code string separated by commas.

    This function splits the input `code` string by commas to extract individual arguments
    that are typically passed to a Python script. It assumes the arguments are directly
    embedded in the provided string and separated by commas.

    :param code: The Python code or function call string containing comma-separated arguments.
    :type code: str
    
    :returns: A list containing individual arguments extracted from the `code` string.
    :rtype: list of str
    
    :note:
        - The function does not perform validation or parsing beyond simple comma splitting.
        - It assumes the input `code` string represents valid Python syntax.

    Example
    -------
    >>> code_string = "function_name(arg1, arg2, arg3)"
    >>> arguments = get_arguments_from_code(code_string)
    >>> print(arguments)
    ['function_name(arg1', ' arg2', ' arg3)']
    """
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: June 22, 2024
    args_string = code.strip()
    args = args_string.split(',')
    return args

def find_py_files(path):
    """
    Recursively finds all Python (.py) files in the specified directory path.

    This function searches for all Python files (.py) in the given directory and its subdirectories,
    returning a list of the file names without their extensions.

    :param path: The directory path where the search for Python files should begin.
    :type path: str
    
    :returns: A list of Python file names (without the .py extension) found in the specified directory.
    :rtype: list of str
    
    :note:
        - The function constructs a search pattern for .py files using `os.path.join` and `glob.glob`.
        - It searches recursively (`recursive=True`) to find all matching .py files in subdirectories.
        - The file names are extracted from the full paths, and the .py extension is removed.

    Example
    -------
    >>> path = '/path/to/python/scripts'
    >>> python_files = find_py_files(path)
    >>> print(python_files)
    ['script1', 'script2', 'subdir/script3']
    """
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: June 22, 2024

    # Dev. Comments:
    # -------------------
    # This function executes a single user prompt with BRAD
    #
    # History:
    # - 2024-06-22: initial draft
    # - 2024-07-22: this function is modified to not use recursive search so as
    #               to allow a path list, as opposed to a single matlab path, to
    #               be used to point to where BRAD should find code

    # Construct the search pattern for .m files
    search_pattern = os.path.join(path, '*.py')
    
    # Find all .m files recursively
    py_files = glob.glob(search_pattern, recursive=False)

    # Extract only the file names from the full paths
    file_names = [os.path.basename(file)[:-3] for file in py_files]
    
    return file_names

def find_closest_function(query, functions):
    """
    Finds the closest matching function name to the query from a list of functions.

    This function uses the `difflib.get_close_matches` method to find the closest match
    to the provided query within the list of function names. It returns the best match
    if any matches are found, otherwise it returns None.

    :param query: The query string to match against the list of functions.
    :type query: str
    
    :param functions: The list of function names to search within.
    :type functions: list of str
    
    :returns: The closest matching function name if a match is found; otherwise, None.
    :rtype: str or None
    
    :note:
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

def read_python_docstrings(file_path):
    """
    Reads the docstrings from a Python file located at the given file path.

    This function extracts lines that are inside triple-quoted strings at the beginning of the file. It stops reading further lines once it encounters a line that does not belong to the docstring anymore.

    :param file_path: The path to the Python file from which to read the docstrings.
    :type file_path: str
    
    :returns: A string containing the extracted docstrings, with each line separated by a newline character.
    :rtype: str
    
    :note:
        - The function assumes that the docstrings are located at the beginning of the Python file.
        - It stops reading further lines once a line that does not start with triple or double quotes
          is encountered, assuming that the docstrings are defined as per Python conventions.
        - The function preserves leading and trailing spaces in the docstring lines.
    
    Example
    -------
    >>> Given a Python file '/path/to/module.py' with the following content:
    >>> 
    >>> ```
    >>> '''
    >>> This is a sample module.
    >>> 
    >>> It demonstrates how to read docstrings from a Python file.
    >>> '''
    >>> def my_function():
    >>>     pass
    >>> ```
    >>> 
    >>> Calling `read_python_docstrings('/path/to/module.py')` would return:
    >>> ```
    >>> '''
    >>> This is a sample module.
    >>> 
    >>> It demonstrates how to read docstrings from a Python file.
    >>> '''
    >>> ```
    >>> 
    >>> If the Python file does not start with a docstring, an empty string ('') is returned.

    """
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: June 22, 2024

    docstring_lines = []
    inside_docstring = False
    
    with open(file_path, 'r') as file:
        for line in file:
            stripped_line = line.strip()
            if stripped_line.startswith('"""') or stripped_line.startswith("'''"):
                if inside_docstring:
                    # This ends the docstring
                    docstring_lines.append(stripped_line)
                    break
                else:
                    # This starts the docstring
                    docstring_lines.append(stripped_line)
                    inside_docstring = True
            elif inside_docstring:
                # Collect lines inside the docstring
                docstring_lines.append(line.rstrip())
            elif not stripped_line:
                # Skip any leading empty lines
                continue
            else:
                # Stop if we encounter a non-docstring line and are not inside a docstring
                break
    
    return "\n".join(docstring_lines)

def get_py_description(file_path):
    """
    Extracts a brief description from the docstrings of a Python (.py) file.

    This function reads the docstrings from the specified Python file and returns
    a one-liner description extracted typically from the second line of the docstrings.

    Parameters
    ----------
    file_path : str
        The path to the Python file from which to extract the description.

    Returns
    -------
    str
        A one-liner description extracted from the second line of the docstrings.

    Notes
    -----
    - The function assumes that the second line of the docstrings contains a brief description 
      of the Python script.
    - It uses the `read_python_docstrings` function to read the docstrings from the file.

    Example
    -------
    >>> description = get_py_description('/path/to/python/script.py')
    >>> print(description)
    ' This script demonstrates how to read docstrings from a Python file.'
    """
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: June 22, 2024
    docstrings = read_python_docstrings(file_path)
    oneliner = docstrings.split('\n')[1]
    return oneliner

def extract_python_code(llm_output, scriptPath, state, memory=None):
    """
    Parses LLM output and extracts the Python code to execute.

    This function processes the output generated by an LLM (Language Model) or similar tool,
    typically extracting the final line of generated code intended for execution. It handles
    special cases where the generated code might need modification or formatting.

    :param llm_output: The complete output string from the LLM tool, which includes generated Python code.
    :type llm_output: str
    
    :returns: The Python code to be executed.
    :rtype: str
    
    :note:
        - The function assumes that the LLM output contains the generated Python code as the final line.
        - It may modify the format of the code for compatibility or execution purposes.
        - If the LLM output does not conform to expectations, this function may need adjustment.
    
    Example
    -------
    >>> If `llm_output` is:
    ... ```
    ... Some generated code...
    ... Execute: subprocess.call([sys.executable, 'script.py'])
    ... ```
    ... Calling `extract_python_code(llm_output, state)` would return:
    ... ```
    ... subprocess.run([sys.executable, 'script.py'])
    ... ```
    ...
    
    If the LLM output does not include a valid Python execution command, the function might modify
    or construct one based on expected patterns.
    """
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: June 22, 2024

    # Dev. Comments:
    # -------------------
    # This function should take llm output and extract python code
    #
    # History:
    # - 2024-06-22: initiall attempt to extract python from llms
    # - 2024-07-08: (1) strip('\'"`') is added to remove trailing quotations
    #               (2) <path/to/script> is set as a keyword to be replaced by
    #                   python paths
    #               (3) added a subsequent python call to edit the command when
    #                   compile() fails
    #
    # Issues:
    # - This is very sensitive to the particular model, prompt, and patterns in
    #   the llm response.
    
    log.debugLog("LLM OUTPUT PARSER", state=state) 
    log.debugLog(llm_output, state=state) 

    funcCall = llm_output.split('Execute:')
    log.debugLog(funcCall, state=state)
    funcCall = funcCall[len(funcCall)-1].strip()
    if funcCall.startswith('```python'):
        funcCall = funcCall.split('\n')[1]
    funcCall = funcCall.strip('\'"`') # remove specific characters (single quotes', double quotes ", and backticks `` ``) 
    
    # Ensure placeholder path is replaced
    funcCall = funcCall.replace('<path/to/script>/', scriptPath)
    funcCall = funcCall.replace('<path/to/script>',  scriptPath)
    
    log.debugLog('Stripped funciton call', state=state)
    log.debugLog(funcCall, state=state)
    
    if not funcCall.startswith('subprocess.run([sys.executable,'):
        funcCall = 'subprocess.run([sys.executable' + funcCall[funcCall.find(','):]

    log.debugLog('Cleaned up function call:\n', state=state)
    log.debugLog(funcCall, state=state)
    log.debugLog('\n', state=state)
    try:
        compile(funcCall, '<string>', 'exec')
    except Exception as e:
        return editPythonCode(funcCall, str(e), memory, state)
    
    return funcCall

def editPythonCode(funcCall, errorMessage, memory, state):
    """
    Edit the Python code based on the error message and chat status, with recursion depth limit.

    :param funcCall: The Python code to be edited.
    :type funcCall: str
    
    :param errorMessage: The error message related to the code.
    :type errorMessage: str
    
    :param state: Dictionary containing chat status and configuration.
    :type state: dict
    
    :returns: The edited Python code ready for execution, or an error message if recursion limit is reached.
    :rtype: str
    """

    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: July 8, 2024

    # Bound the recursion depth of this function
    if state['recursion_depth'] > 5:
        log.errorLog('Recursion depth limit reached. Please check the code and error message', state=state)
        return "Recursion depth limit reached. Please check the code and error message."
    state['recursion_depth'] += 1

    # Extract the llm
    llm = state['llm']

    # Fill in the python editting template
    template = getPythonEditingTemplate()
    filled_template = template.format(
        code1 = funcCall,
        code2 = funcCall,
        error = errorMessage
    )
    log.debugLog("Memory", state=state)
    log.debugLog(memory,   state=state)
    # Build the langchain
    PROMPT          = PromptTemplate(input_variables=["history", "input"], template=filled_template)
    log.debugLog(PROMPT, state=state)
    conversation    = ConversationChain(prompt  = PROMPT,
                                        llm     =    llm,
                                        verbose = state['config']['debug'],
                                        memory  = memory,
                                       )

    # Call the llm/langchain
    response = conversation.predict(input=state['prompt'])

    # Parse the code (recursion begins here)
    code2execute = extract_python_code(response, state['config']['CODE']['path'][0], state, memory=memory)

    # log the llm output
    state['process']['steps'].append(log.llmCallLog(llm             = llm,
                                                         prompt          = PROMPT,
                                                         input           = state['prompt'],
                                                         output          = response,
                                                         parsedOutput    = {
                                                             'code': code2execute
                                                         },
                                                         purpose         = 'Edit function call'
                                                        )
                                         )
    # Decrement recursion depth after use
    state['recursion_depth'] -= 1
    return code2execute

def has_unclosed_symbols(s):
    """
    Check for unclosed symbols in a string.

    This function checks if a string has unclosed symbols such as quotes,
    parentheses, brackets, or braces. It returns True if there are unclosed
    symbols, and False otherwise.

    :param s: The string to check for unclosed symbols.
    :type s: str

    :returns: True if there are unclosed symbols, False otherwise.
    :rtype: bool

    """
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: July 8, 2024
    
    stack = []
    pairs = {'(': ')', '[': ']', '{': '}'}
    opening = pairs.keys()
    closing = pairs.values()
    
    i = 0
    while i < len(s):
        char = s[i]
        if char in pairs:
            stack.append(pairs[char])
        elif char in closing:
            if not stack or char != stack.pop():
                return True
        elif char in ['"', "'"]:
            # Check if this is an escaped quote
            if i > 0 and s[i - 1] == '\\':
                i += 1
                continue
            # Toggle the quote in the stack
            if stack and stack[-1] == char:
                stack.pop()
            else:
                stack.append(char)
        i += 1
    
    return bool(stack)

