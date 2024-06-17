"""This file is responsible for running scripts from BRAD."""
import os

from langchain import PromptTemplate, LLMChain
from langchain.chains import ConversationChain

from BRAD.matlabCaller import find_matlab_files, get_matlab_description, read_matlab_docstrings, matlabPromptTemplate, activateMatlabEngine, extract_matlab_code, execute_matlab_code
from BRAD.pythonCaller import find_py_files, get_py_description, read_python_docstrings, pythonPromptTemplate, extract_python_code, execute_python_code
from BRAD.promptTemplates import scriptSelectorTemplate

def codeCaller(chatstatus):
    print('CODER') if chatstatus['config']['debug'] else None
    prompt = chatstatus['prompt']                                        # Get the user prompt
    llm = chatstatus['llm']                                              # Get the llm
    memory = chatstatus['memory']
    chatstatus['process'] = {}                                           # Begin saving plotting arguments
    chatstatus['process']['name'] = 'CODER'

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
    filled_template = template.format(scriptName=scriptName, scriptDocumentation=docstrings)
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
    executor = {'python': execute_python_code, 'MATLAB': execute_matlab_code}.get(scriptType)
    print('Executing Code!!')
    executor(code2execute, chatstatus)

