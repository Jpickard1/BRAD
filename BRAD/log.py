"""
This module provides logging utilities for tracking and managing the state of chat processes 
and interactions within the BRAD framework. It includes functions for logging chat details, 
language model (LLM) calls, file loads, debug information, and user-facing outputs. 
The goal of this module is to standardize how logging and debugging information is handled 
and ensure that the chat processes are consistently tracked.

Functions:
----------
1. logger(chatlog, chatstatus, chatname, elapsed_time=None):
    Logs the current status of the chat session, including input prompts, outputs, and the process details.
    Serializes non-JSON serializable data and writes the log to a specified file.

2. llmCallLog(llm=None, memory=None, prompt=None, input=None, output=None, parsedOutput=None, purpose=None):
    Creates a log entry for a language model (LLM) call, capturing details such as the prompt, output, and purpose.

3. loadFileLog(file=None, delimiter=None):
    Creates a log entry for loading a file, logging details like the file name and delimiter used.

4. debugLog(output, chatstatus=None, display=None):
    Standardizes the logging of debugging information. Logs details based on the chat status or displays them directly.

5. errorLog(errorMessage, info=None, chatstatus=None):
    Logs error messages along with optional additional information and updates the chat process steps.

6. userOutput(output, chatstatus=None):
    Standardizes the output printed to the user and logs it into the chat status.

7. is_json_serializable(value):
    Checks if a given value can be serialized to JSON. Useful for ensuring that chat logs are compatible with JSON format.

Usage:
------
The module is intended for logging various aspects of the BRAD chatbot's operations. It tracks interactions,
such as user inputs, LLM outputs, errors, and file loading events. The functions support serialization of 
non-JSON data and are designed to facilitate debugging and logging in a clear, consistent manner.

Example:
--------
To log a chat process:
>>> chatlog, chatstatus = logger(chatlog, chatstatus, 'chat_log.json')

To log an LLM call:
>>> llm_log = llmCallLog(llm=llm, prompt='Generate text', output='Text generated')

To handle debug logs:
>>> debugLog("This is a debug message", chatstatus)

Log Format:
-----------
```
[
    0: {
        'TIME'   : <time stamp>,
        'PROMPT' : <input from the user or preplanned prompt>,
        'OUTPUT' : <output displayed to the user>,
        'STATUS' : {
                'LLM' : <primary large language model being used>,
                'Databases' : {
                        'RAG' : <primary data base>,
                        <other databases>: <more databases can be added>,
                        ...
                    },
                'config' : {
                        'debug' : True,
                        'output-directory' : <path to output directory>,
                        ...
                    }
            },
        'PROCESS' : {
                'MODULE' : <name of module i.e. RAG, DATABASE, CODER, etc.>
                'STEPS' [
                        # information for a particular step involved in executing the module. Some examples
                        {
                           'LLM' : <language model>,
                           'prompt template' : <prompt template>,
                           'model input' : <input to model>,
                           'model output' : <output of model>
                        },
                        {
                           'func' : 'rag.retreival',
                           'num articles retrieved' : 10,
                           'multiquery' : True,
                           'compression' : True,
                        }
                    ]
            },
        'PLANNED' : [
                <Next prompt in queue>,
                ...
            ]
    },
1 : {
        'TIME'   : <time stamp>,
        'PROMPT' : <second prompt>,
        ...
    },
...
]
```

"""

import json
import logging
import time

def logger(chatlog, chatstatus, chatname, elapsed_time=None):
    """
    Logs the chat status and process details into a specified chat log file.

    :param chatlog: The dictionary containing the chat log entries.
    :type chatlog: dict
    :param chatstatus: The current status of the chat, including prompt, output, and process details.
    :type chatstatus: dict
    :param chatname: The name of the file where the chat log will be saved.
    :type chatname: str

    :raises FileNotFoundError: If the specified chat log file cannot be created or accessed.
    :raises TypeError: If the chat log or status contains non-serializable data that cannot be converted to a string.

    :return: A tuple containing the updated chat log and chat status.
    :rtype: tuple
    """

    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: June 4, 2024
    
    
    debugLog('\n\nPLANNER:\n\n' + str(chatstatus['process']), chatstatus)
    
    process_serializable = {
            key: value if is_json_serializable(value) else str(value)
            for key, value in chatstatus['process'].items()
        }

    chatlog[len(chatlog)] = {
        'time'   : time.ctime(time.time()),
        'elapsed time' : elapsed_time,
        'prompt' : chatstatus['prompt'],  # the input to the user
        'output' : chatstatus['output'],  # the output to the user
        'process': process_serializable,  # chatstatus['process'], # information about the process that ran
        'status' : {                      # information about the chat at that time
            'databases'         : str(chatstatus['databases']),
            'current table'     : {
                    'key'       : chatstatus['current table']['key'],
                    'tab'       : chatstatus['current table']['tab'].to_json() if chatstatus['current table']['tab'] is not None else None,
                },
            'current documents' : chatstatus['current documents'],
            'queue pointer'     : chatstatus['queue pointer'],
            'queue'             : chatstatus['queue'].copy()
        },
    }
    with open(chatname, 'w') as fp:
        json.dump(chatlog, fp, indent=4)
    # chatstatus['process'] = None
    return chatlog, chatstatus

def llmCallLog(llm=None, memory=None, prompt=None, input=None, output=None, parsedOutput=None, purpose=None):
    """
    This function logs the information for each LLM call.
    
    `Note`: Working to make this work better with the OpenAI callback manager.

    """
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: June 19, 2024
    llmLog = {
        'llm'          : str(llm),
        'memory'       : str(memory),
        'prompt'       : str(prompt),
        'input'        : str(input),
        'output'       : str(output),
        'parsedOutput' : parsedOutput,
        'purpose'      : str(purpose)
    }
    return llmLog

def loadFileLog(file=None, delimiter=None):
    """
    This function logs the information for each file loaded.
    """
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: June 19, 2024
    loadLog = {
        'file'      : str(file),
        'delimiter' : str(delimiter)
    }
    return loadLog

def debugLog(output, chatstatus=None, display=None):
    """This function standardizes how debugging information is provided to the user"""
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: June 19, 2024
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    if display:
        logging.info(output)
    elif chatstatus['config']['debug']: # or str(chatstatus['config']['debug']).lower() == 'true':
        logging.info(output)
        # print('DEBUG')
        # print(output)

def errorLog(errorMessage, info=None, chatstatus=None):
    """This function displays and logs error messages"""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info(errorMessage)
    chatstatus['process']['steps'].append(
        {
            'func'    : 'log.errorLog',
            'message' : errorMessage,
            'info'    : info
        }
    )
    

def userOutput(output, chatstatus=None):
    """This function standardizes how information is printed to the user and allows for logging"""
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: June 20, 2024
    
    print(output)
    if 'output' not in chatstatus.keys() or chatstatus['output'] is None:
        chatstatus['output'] = output
    else:
        chatstatus['output'] += output
    return chatstatus

def is_json_serializable(value):
    """
    Checks if a given value is JSON serializable.

    :param value: The value to be checked for JSON serializability.
    :type value: Any

    :return: True if the value is JSON serializable, False otherwise.
    :rtype: bool

    """
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: June 4, 2024

    if isinstance(value, list):
        return True
    
    try:
        json.dumps(value)
        return True
    except (TypeError, OverflowError):
        return False