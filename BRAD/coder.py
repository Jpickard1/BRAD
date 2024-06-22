"""
Module responsible for executing scripts within the BRAD framework.

This module facilitates the discovery, selection, and execution of Python and MATLAB scripts based on user prompts
and configuration settings.

Requirements:

1. Python and MATLAB scripts are located at specified paths configured in `config/config.json`.
2. Scripts are executed with the first argument denoting the output directory for saving any generated files.
3. Script files within the Python and MATLAB paths contain sufficient documentation, including:

   - A concise one-line summary at the beginning of the docstring (used by the llm for script selection).
   - Comprehensive descriptions detailing arguments, inputs, purposes, and usage examples (used by the llm for execution).

Functions:
- codeCaller: Executes a script based on the user's prompt and chat status configuration.
- executeCode: Executes provided code based on the specified script type.
"""

import os

from langchain import PromptTemplate, LLMChain
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

from BRAD.matlabCaller import find_matlab_files, get_matlab_description, read_matlab_docstrings, matlabPromptTemplate, activateMatlabEngine, extract_matlab_code, execute_matlab_code
from BRAD.pythonCaller import find_py_files, get_py_description, read_python_docstrings, pythonPromptTemplate, extract_python_code, execute_python_code
from BRAD.promptTemplates import scriptSelectorTemplate

def codeCaller(chatstatus):
    """
    Executes a script based on the user's prompt and chat status configuration.

    This function performs the following steps:

        1. Finds available Python and MATLAB scripts in the specified directories.
        2. Extracts docstrings from the scripts to understand their purpose.
        3. Uses llm to select appropriate codes to execute and format command to run the code with the correct inputs
        4. Executes the selected script and updates the chat status.

    Parameters
    ----------
    chatstatus : dict
        A dictionary containing the chat status, including user prompt, language model (llm),
        configuration settings, and output directory.

    Returns
    -------
    dict
        Updated chat status after executing the selected script.

    Notes
    -----
    The function makes two calls to a language model (llm) to determine which script to execute
    and to format the execution command. It supports both Python and MATLAB scripts.

    The function assumes that the configuration dictionary (`chatstatus['config']`) contains the keys:
    - 'debug': A boolean indicating whether to print debug information.
    - 'py-path': A string specifying the path to Python scripts.
    - 'matlab-path': A string specifying the path to MATLAB scripts.
    - 'output-directory': A string specifying the directory to store output files.

    Example
    -------
    >>> chatstatus = {
    ...     'config': {
    ...         'debug': True,
    ...         'py-path': 'py-tutorial/',
    ...         'matlab-path': 'matlab-tutorial/',
    ...         'output-directory': '/path/to/output'
    ...     },
    ...     'prompt': 'Run analysis',
    ...     'llm': language_model_instance
    ... }
    >>> updated_status = codeCaller(chatstatus)
    """
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: June 22, 2024
    print('CODER') if chatstatus['config']['debug'] else None
    prompt = chatstatus['prompt']                                        # Get the user prompt
    llm = chatstatus['llm']                                              # Get the llm
    memory = ConversationBufferMemory(ai_prefix="BRAD")                  # chatstatus['memory']

    # Get available matlab and python scripts
    if chatstatus['config']['py-path'] == 'py-tutorial/': # Admittedly, I'm not sure what's going on here - JP
        pyPath = chatstatus['config']['py-path']
    else:
        base_dir = os.path.expanduser('~')
        pyPath = os.path.join(base_dir, chatstatus['config']['py-path'])

    if chatstatus['config']['matlab-path'] == 'matlab-tutorial/':
        matlabPath = chatstatus['config']['matlab-path']
    else:
        base_dir = os.path.expanduser('~')
        matlabPath = os.path.join(base_dir, chatstatus['config']['matlab-path'])

    pythonScripts = find_py_files(pyPath)
    matlabScripts = find_matlab_files(matlabPath)

    # Get matlab and python docstrings
    scriptPurpose = {}
    for script in pythonScripts:
        scriptPurpose[script] = {'description': get_py_description(os.path.join(pyPath, script + '.py')), 'type': 'python'}
    for script in matlabScripts:
        scriptPurpose[script] = {'description': get_matlab_description(os.path.join(matlabPath, script + '.m')), 'type': 'MATLAB'}
    script_list = ""
    for script in scriptPurpose.keys():
        script_list += "Script Name: " + script + ", \t Description: " + scriptPurpose[script]['description'] + '\n'
    
    # Determine which code needs to be executed (first llm call)
    template = scriptSelectorTemplate()
    template = template.format(script_list=script_list)
    PROMPT = PromptTemplate(input_variables=["user_query"], template=template)
    print(PROMPT) if chatstatus['config']['debug'] else None
    chain = PROMPT | llm # LCEL chain creation
    print('FIRST LLM CALL') if chatstatus['config']['debug'] else None
    res = chain.invoke(prompt)
    print(res.content) if chatstatus['config']['debug'] else None
    scriptName   = res.content.strip().split('\n')[0].split(':')[1].strip()
    scriptType   = scriptPurpose[scriptName]['type']
    scriptPath   = {'python': pyPath, 'MATLAB': matlabPath}.get(scriptType, print('Warning! the type doesnt exist'))
    if scriptType == 'MATLAB':
        chatstatus, _ = activateMatlabEngine(chatstatus) # turn on and add matlab files to path
    else:
        print('scriptPath=' + str(scriptPath))
        print('pyPath=' + str(pyPath))
        scriptName = os.path.join(scriptPath, scriptName)
    scriptSuffix = {'python': '.py', 'MATLAB': '.m'}.get(scriptType)
    scriptName  += scriptSuffix

    # Format code to execute: read the doc strings, format function call (second llm call), parse the llm output
    print('ALL SCRIPTS FOUND. BUILDING TEMPLATE') if chatstatus['config']['debug'] else None
    
    docstringReader = {'python': read_python_docstrings, 'MATLAB': read_matlab_docstrings}.get(scriptType)
    docstrings      = docstringReader(os.path.join(scriptPath, scriptName))
    scriptCallingTemplate = {'python': pythonPromptTemplate, 'MATLAB': matlabPromptTemplate}.get(scriptType)
    template        = scriptCallingTemplate()
    filled_template = template.format(scriptName=scriptName, scriptDocumentation=docstrings, output_path=chatstatus['output-directory'])
    PROMPT          = PromptTemplate(input_variables=["history", "input"], template=filled_template)
    print(PROMPT) if chatstatus['config']['debug'] else None
    conversation    = ConversationChain(prompt  = PROMPT,
                                        llm     =    llm,
                                        verbose =   chatstatus['config']['debug'],
                                        memory  = memory,
                                       )
    response = conversation.predict(input=chatstatus['prompt'])
    responseParser = {'python': extract_python_code, 'MATLAB': extract_matlab_code}.get(scriptType)
    code2execute = responseParser(response, chatstatus)

    # Execute code
    executeCode(chatstatus, code2execute, scriptType)

    return chatstatus

def executeCode(chatstatus, code2execute, scriptType):
    """
    Executes the given code based on the specified script type.

    This function determines the appropriate executor for the provided code
    (either Python or MATLAB) and executes it.

    Parameters
    ----------
    chatstatus : dict
        A dictionary containing the chat status, including configuration settings and other relevant data.
    code2execute : str
        The code to be executed.
    scriptType : str
        The type of the script to be executed. It should be either 'python' or 'MATLAB'.

    Notes
    -----
    The actual execution is delegated to the appropriate function based on the script type. This function uses a dictionary to map the script type to the corresponding executor function:
    - 'python': `execute_python_code`
    - 'MATLAB': `execute_matlab_code`
    """
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: June 22, 2024
    executor = {'python': execute_python_code, 'MATLAB': execute_matlab_code}.get(scriptType)
    print('Executing Code!!')
    executor(code2execute, chatstatus)

