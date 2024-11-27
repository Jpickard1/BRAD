"""
Code Caller
-----------

This module facilitates the discovery, selection, and execution of Python and MATLAB scripts 
based on user prompts and predefined configuration settings.

Key Features
~~~~~~~~~~~~

1. Python scripts must reside in the directories specified within the configuration settings.

2. Script execution requires the first argument to specify the output directory, where any 
   resulting files will be saved.

3. Each script must include clear and structured documentation, consisting of:

   - A concise one-line summary at the beginning of the docstring (used by the LLM for script selection).
   
   - Comprehensive descriptions detailing the script’s arguments, inputs, purpose, and usage 
     examples (utilized by the LLM for accurate execution).

Available Methods
~~~~~~~~~~~~~~~~~

This module has the following methods:

"""


import os
import time
from langchain import PromptTemplate, LLMChain
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_community.callbacks import get_openai_callback

from BRAD.pythonCaller import find_py_files, get_py_description, read_python_docstrings, pythonPromptTemplate, extract_python_code, execute_python_code
from BRAD.promptTemplates import scriptSelectorTemplate, pythonPromptTemplateWithFiles
from BRAD import log
from BRAD import utils

# History:
#  2024-10-01: This file was modified to remove support for running MATLAB codes
#  2024-11-27: Final stage of the pipeline added to have the LLM generate a response
#              at the end of the pipeline

def code_caller(state):
    """
    Executes a Python script based on the user's prompt and chat status settings.
    
    This function performs the following steps:
    
        1. Searches the specified directories for available Python scripts.
        
        2. Extracts and analyzes docstrings from each script to identify their purpose.
        
        3. Uses a large language model (LLM) to select the most appropriate script and format the command with the correct inputs.
        
        4. Executes the selected script and updates the chat status accordingly.
    """
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: June 22, 2024

    log.debugLog("CODER", state=state)
    prompt = state['prompt']                                        # Get the user prompt
    llm = state['llm']                                              # Get the llm
    memory = ConversationBufferMemory(ai_prefix="BRAD")                  # state['memory']

    # Get available matlab and python scripts
    path = state['config']['CODE']['path']
    
    scripts = {}
    for fdir in path:
        scripts[fdir] = {}
        scripts[fdir]['python'] = find_py_files(fdir) # pythonScripts
    #    scripts[fdir]['matlab'] = find_matlab_files(fdir) # matlabScripts
    
    # Get matlab and python docstrings
    scriptPurpose = {}
    for fdir in path:
        # print(fdir)
        for script in scripts[fdir]['python']:
            scriptPurpose[script] = {'description': get_py_description(os.path.join(fdir, script + '.py')), 'type': 'python'}
    script_list = ""
    for script in scriptPurpose.keys():
        script_list += "Script Name: " + script + ", \t Description: " + scriptPurpose[script]['description'] + '\n'

    # Determine which code needs to be executed (first llm call)
    template = scriptSelectorTemplate()
    template = template.format(script_list=script_list)
    PROMPT = PromptTemplate(input_variables=["user_query"], template=template)
    log.debugLog(PROMPT, state=state)
    chain = PROMPT | llm # LCEL chain creation
    log.debugLog("FIRST LLM CALL", state=state)

    # Call LLM
    start_time = time.time()
    with get_openai_callback() as cb:
        res = chain.invoke(prompt)
    responseOriginal = {'original': res.copy()}
    responseOriginal['time'] = time.time() - start_time
    responseOriginal['call back'] = {
        "Total Tokens": cb.total_tokens,
        "Prompt Tokens": cb.prompt_tokens,
        "Completion Tokens": cb.completion_tokens,
        "Total Cost (USD)": cb.total_cost
    }

    log.debugLog(res.content, state=state)
    scriptName   = res.content.strip().split('\n')[0].split(':')[1].strip()
    scriptType   = scriptPurpose[scriptName]['type']
    scriptPath = None
    for fdir in path:
        if scriptName in scripts[fdir][scriptType]:
            scriptPath = fdir
            break
    if scriptPath is None:
        log.debugLog('the scriptPath was not found', state=state)
        log.debugLog(f'scripts={scripts}', state=state)
        log.debugLog(f'scriptName={scriptName}', state=state)
        log.debugLog(f'scriptType={scriptType}', state=state)

    # NOTE: MATLAB is in an experimental stage and not fully integrated yet
    if scriptType == 'MATLAB':
        state, _ = activateMatlabEngine(state) # turn on and add matlab files to path
        scriptName = os.path.join(scriptPath, scriptName)
    else:
        log.debugLog('scriptPath=' + str(scriptPath), state=state)
        scriptName = os.path.join(scriptPath, scriptName)

    scriptSuffix = {'python': '.py', 'MATLAB': '.m'}.get(scriptType)
    scriptName  += scriptSuffix

    state['process']['steps'].append(
        log.llmCallLog(
            llm             = llm,
            prompt          = PROMPT,
            input           = prompt,
            output          = responseOriginal,
            parsedOutput    = {
                'scriptName': scriptName,
                'scriptType': scriptType,
                'scriptPath': scriptPath
            },
            purpose         = 'Select which code to run'
        )
    )

    # Format code to execute: read the doc strings, format function call (second llm call), parse the llm output
    log.debugLog("ALL SCRIPTS FOUND. BUILDING TEMPLATE", state=state)
    
    docstringReader = {'python': read_python_docstrings}.get(scriptType)
    docstrings      = docstringReader(os.path.join(scriptPath, scriptName))
    scriptCallingTemplate = {'python': pythonPromptTemplateWithFiles}.get(scriptType)
    template        = scriptCallingTemplate()
    if scriptType == 'python':
        createdFiles = "\n".join(utils.outputFiles(state)) # A string of previously created files
        filled_template = template.format(
            scriptName=scriptName,
            scriptDocumentation=docstrings,
            output_path=state['output-directory'],
            files=createdFiles
        )
    else:
        filled_template = template.format(
            scriptName=scriptName,
            scriptDocumentation=docstrings,
            output_path=state['output-directory']
        )

    # Create the prompt template
    PROMPT = PromptTemplate(input_variables=["history", "input"], template=filled_template)
    log.debugLog(PROMPT, state=state)
    # this will allow better logging of the response from the query API
    try:
        # LCEL chain creation: prompt | llm
        chain = PROMPT | llm
        
        # Execute the chain with input prompt
        start_time = time.time()
        with get_openai_callback() as cb:
            response = chain.invoke({"history": memory.abuffer(), "input": state['prompt']})

        responseOriginal = response
        responseOriginal['time'] = time.time() - start_time

        try:
            responseOriginal['call back'] = {
                "Total Tokens": cb.total_tokens,
                "Prompt Tokens": cb.prompt_tokens,
                "Completion Tokens": cb.completion_tokens,
                "Total Cost (USD)": cb.total_cost
            }
        except:
           responseOriginal['call back'] = {
                "Total Tokens": None,
                "Prompt Tokens": None,
                "Completion Tokens": None,
                "Total Cost (USD)": None
            }

        response = response.content

    # this catches the initial implementation
    except:
        conversation = ConversationChain(
            prompt  = PROMPT,
            llm     =    llm,
            verbose = state['config']['debug'],
            memory  = memory,
        )
        start_time = time.time()
        with get_openai_callback() as cb:
            response = conversation.predict(input=state['prompt'])

        responseOriginal = response
        try:
            responseOriginal = {
                'content' : response,
                'time' : time.time() - start_time,
                'call back': {
                    "Total Tokens": cb.total_tokens,
                    "Prompt Tokens": cb.prompt_tokens,
                    "Completion Tokens": cb.completion_tokens,
                    "Total Cost (USD)": cb.total_cost
                }
            }
        except:
            responseOriginal = {
                'content' : response,
                'time' : time.time() - start_time,
                'call back': {
                    "Total Tokens": None,
                    "Prompt Tokens": None,
                    "Completion Tokens": None,
                    "Total Cost (USD)": None
                }
            }
    
    responseParser = {'python': extract_python_code}.get(scriptType)
    code2execute = responseParser(response, scriptPath, state, memory=memory)

    state['process']['steps'].append(
        log.llmCallLog(
            llm             = llm,
            prompt          = PROMPT,
            input           = state['prompt'],
            output          = responseOriginal,
            parsedOutput    = {
                'code': code2execute
            },
            purpose         = 'Format function call'
        )
    )

    # Check if it requires previous inputs
    code2execute = utils.add_output_file_path_to_string(code2execute, state)
    
    # Execute code
    executeCode(state, code2execute, scriptType)

    return state

def executeCode(state, code2execute, scriptType):
    """
    Executes the provided code based on the specified script type.
    
    This function determines the appropriate execution environment (Python or MATLAB) based on the script type and runs the corresponding code.
    
    :param state: A dictionary containing the chat status, including configuration settings and other relevant data.
    :type state: dict
    :param code2execute: The code to be executed.
    :type code2execute: str
    :param scriptType: The type of the script to be executed. Must be either 'python' or 'MATLAB'.
    :type scriptType: str
    """
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: June 22, 2024
    executor = {'python': execute_python_code}.get(scriptType)
    executor(code2execute, state)

