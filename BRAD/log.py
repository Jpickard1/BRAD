"""
This module provides utilities for logging and tracking chat processes and interactions 
within the BRAD framework. It includes functions to record the agent state, LLM calls, file loading, 
debug information, and user-facing outputs, and other pieces of information to ensure consistent
and transparent monitoring of the `Agent` and user interactions. These logging methods should be inserted
throughout the code for development, debugging, and  user-facing output. Log files are
automatically saved in the output directory.

Log Format
----------
Logs capture user interactions and actions taken by an `Agent`. These logs are saved in 
the output directory and can be reviewed for debugging or analysis purposes. Within the log
record, entries are numbered according to their order in the chat. For a single entry, the following
items are recorded:

1. time: the time when the `Agent` responds to this query
2. prompt: the input prompt from the user
3. output: the message displayed to the user
4. status: the `Agent` state after responding to the user (See :ref:`state-schema-section`)
5. process: this records the tool module and the set of particular steps the `Agent` takes using the tool
6. planned: a list of any steps that `Agent` plans to take

These items are saved in the following schema:

>>> [
...     0: {
...         'TIME'   : <time stamp>,
...         'PROMPT' : <input from the user or preplanned prompt>,
...         'OUTPUT' : <output displayed to the user>,
...         'STATUS' : {
...                 'LLM' : <primary large language model being used>,
...                 'Databases' : {
...                         'RAG' : <primary data base>,
...                         <other databases>: <more databases can be added>,
...                         ...
...                     },
...                 'config' : {
...                         'debug' : True,
...                         'output-directory' : <path to output directory>,
...                         ...
...                     }
...             },
...         'PROCESS' : {
...                 'MODULE' : <name of module i.e. RAG, DATABASE, CODER, etc.>
...                 'STEPS' [
...                         # information for a particular step involved in executing the module. Some examples
...                         {
...                            'func' : 'rag.retreival',
...                            'num articles retrieved' : 10,
...                            'multiquery' : True,
...                            'compression' : True,
...                         },
...                         {
...                            'LLM' : <language model>,
...                            'prompt template' : <prompt template>,
...                            'model input' : <input to model>,
...                            'model output' : <output of model>
...                         }
...                     ]
...             },
...         'PLANNED' : [
...                 <Next prompt in queue>,
...                 ...
...             ]
...     },
... 1 : {
...         'TIME'   : <time stamp>,
...         'PROMPT' : <second prompt>,
...         ...
...     },
... ...
>>> ]

Log Methods
-----------

The methods in the log module serve three key functions:

1. Saving information to the log file.
2. Ensuring consistency in how data is recorded within the `steps` section of the log (e.g., all LLM calls follow a uniform format).
3. Displaying output, warnings, and error messages to the user.

The following methods are available:
"""

import json
import logging
import time

def logger(chatlog, state, chatname, elapsed_time=None):
    """
    This methods writes the lof of the current chat status, user inputs, outputs, and process details to
    a specified file.

    The function serializes the chat data, including user inputs, outputs, process information, and the current
    `Agent` state. The resulting log is written to a file specified by `chatname`.

    :param chatlog: A dictionary containing the chat log entries. Each entry is a record of a specific chat interaction.
    :type chatlog: dict
    :param state: The current status of the chat, including details like the user's prompt, the system's output, 
                  process information, and other contextual information (e.g., databases, queue).
    :type state: dict
    :param chatname: The filename (including path) where the chat log will be saved. The file will be created or 
                     overwritten if it already exists.
    :type chatname: str
    :param elapsed_time: Optional. The time elapsed since the start of the chat or a specific reference point. 
                         If not provided, it defaults to None.
    :type elapsed_time: float or None, optional

    :raises FileNotFoundError: If the specified chat log file cannot be created or accessed.
    :raises TypeError: If the chat log or status contains non-serializable data that cannot be converted to a string.

    :return: A tuple containing:
             - `chatlog`: The updated chat log with the new entry.
             - `state`: The current state of the system after logging.
    :rtype: tuple

    :note: The log is saved in JSON format with proper indentation to improve readability. Non-serializable data 
           within the chat state (e.g., dataframes) is converted to string format to ensure proper logging.
    """
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: June 4, 2024
    
    
    debugLog('\n\nPLANNER:\n\n' + str(state['process']), state)
    
    process_serializable = {
            key: value if is_json_serializable(value) else str(value)
            for key, value in state['process'].items()
        }

    chatlog[len(chatlog)] = {
        'time'   : time.ctime(time.time()),
        'elapsed time' : elapsed_time,
        'prompt' : state['prompt'],  # the input to the user
        'output' : state['output'],  # the output to the user
        'continue-module': state['continue-module'],
        'process': process_serializable,  # state['process'], # information about the process that ran
        'status' : {                      # information about the chat at that time
            'databases'         : str(state['databases']),
            # 'current table'     : {
            #         'key'       : state['current table']['key'],
            #         'tab'       : state['current table']['tab'].to_json() if state['current table']['tab'] is not None else None,
            #     },
            'current documents' : state['current documents'],
            'queue pointer'     : state['queue pointer'],
            'queue'             : state['queue'].copy()
        },
    }
    with open(chatname, 'w') as fp:
        json.dump(chatlog, fp, indent=4)
    return chatlog, state

def llmCallLog(llm=None, memory=None, prompt=None, input=None, output=None, parsedOutput=None, apiInfo=None, purpose=None):
    """
    Logs the information for each LLM call, capturing relevant details for tracking and debugging. This can
    include both raw and processed outputs, as well as the context of the call.
    
    :param llm: The identifier or name of the LLM used in the call (e.g., GPT-3, GPT-4, etc.). If not provided, 
                it defaults to None.
    :type llm: str, optional
    
    :param memory: A memory object that tracks the conversational history or state associated with the LLM call.
                   This could be a representation of past interactions or context shared between the user and LLM.
    :type memory: object, optional
    
    :param prompt: The template or content used to prompt the LLM. This can be a system prompt or a structured input
                   that guides the LLM in generating a response.
    :type prompt: str, optional
    
    :param input: The full input provided to the LLM, which includes both the prompt and any additional context
                  or parameters. This represents the complete information sent to the model for processing.
    :type input: str, optional
    
    :param output: The complete output returned by the LLM. This may contain large amounts of text or data generated 
                   by the model in response to the input.
    :type output: str, optional
    
    :param parsedOutput: The processed or parsed output that is actually used or displayed. This can be a subset 
                         or reformatted version of the full LLM output, focusing on the most relevant information.
    :type parsedOutput: any, optional

    :param apiInfo: The callback information regarding LLM api utilization and fees

    :type apiInfo: dict, optional
    
    :param purpose: A description of the purpose of the LLM call. This field explains why the LLM was called and
                    what task or goal the call is intended to achieve (e.g., generate response, extract information).
    :type purpose: str, optional
    
    :return: A dictionary containing the logged information for the LLM call, including the LLM name, memory,
             prompt, input, output, parsedOutput, and purpose.
    :rtype: dict
    
    :note: Callback information from OpenAI or NVIDIA APIs can be saved in the output field, which contains the full output
           produced by the LLM. The `parsedOutput` should contain only the information used by BRAD or displayed to the user.
    """
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: June 19, 2024
    llmLog = {
        'llm'          : str(llm),     # LLM
        'memory'       : str(memory),  # memory object
        'prompt'       : str(prompt),  # prompt template for LLM
        'input'        : str(input),   # full input to LLM
        'output'       : str(output),  # full output from LLM
        'parsedOutput' : parsedOutput, # used output information
        'api-info'     : apiInfo,      # information about LLM api utilization
        'purpose'      : str(purpose)  # why is this llm call needed?
    }
    return llmLog

def loadFileLog(file=None, delimiter=None):
    """
    Logs and returns information about a file that has been loaded into the system. This function captures
    details such as the file name and the delimiter used to parse the file.

    The logged information is returned as a dictionary, making it easy to track and review the file loading 
    process during later stages, such as debugging or auditing the steps involved in data processing.

    :param file: The name or path of the file being loaded. If not provided, it defaults to `None`.
    :type file: str, optional
    :param delimiter: The delimiter used to parse the file (e.g., commas for CSV files). If not specified, 
                      it defaults to `None`.
    :type delimiter: str, optional

    :return: A dictionary containing:
             - `file`: The name or path of the file (as a string).
             - `delimiter`: The delimiter used in the file (as a string).
    :rtype: dict

    :note: This function can be useful for logging metadata about files loaded into the system, especially 
           when working with various file formats that may require different parsing strategies.
    """
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: June 19, 2024
    loadLog = {
        'file'      : str(file),
        'delimiter' : str(delimiter)
    }
    return loadLog

def debugLog(output, state=None, display=None):
    """
    Records debug information to a log file and optionally displays messages to the user based on the system's 
    debug settings.

    This function is used to capture and report errors or other important debug information. It checks the 
    current debug configuration and either writes the log to a file or displays it to the user, ensuring 
    flexibility in managing debug output.

    :param output: The message or error to be logged. This can be any information that needs to be saved for 
                   debugging or tracking purposes.
    :type output: str
    :param state: The current state of the system, containing configuration details, including whether debug 
                  logging is enabled. This is used to determine whether to log the output.
    :type state: dict, optional
    :param display: If set to `True`, the output will be displayed to the user regardless of the debug configuration. 
                    If `False` or not provided, the function will refer to the debug setting in `state['config']['debug']`.
    :type display: bool, optional

    :raises KeyError: If `state['config']['debug']` is not found when `display` is `None`.
    
    :note: Debug logs are saved using Python's logging library, and the output format includes timestamps and 
           logging levels. If `display` is enabled or debug mode is active, the output will be shown to the user.
           Otherwise, it will be silently logged to the configured log file.
    """
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: June 19, 2024
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    if display:
        logging.info(output)
    elif state is not None and state['config']['debug']:
        logging.info(output)


def errorLog(errorMessage, info=None, state=None):
    """
    Logs an error message and related information to the log file and updates the process steps in the system's state.

    This function is designed to log errors that occur during the operation of the BRAD framework. It records the 
    error message in the log file with a timestamp, and if provided, additional information (`info`) can be logged 
    as well. The error is also appended to the `state['process']['steps']` list for tracking in the system's state.

    :param errorMessage: The error message to be logged. This is a string that describes the nature of the error.
    :type errorMessage: str
    :param info: Optional additional information related to the error. This can provide context or further details 
                 about the error, such as relevant variables or states at the time of the error.
    :type info: any, optional
    :param state: The current system state, which contains information about the chat process and tracks the 
                  progression of steps. The error and related information will be appended to `state['process']['steps']`.
    :type state: dict, optional

    :raises KeyError: If `state['process']['steps']` is not found, meaning the state has not been properly initialized 
                      for logging steps.

    :note: The error is logged using Python's logging library, which outputs the error message with a timestamp 
           and logging level. Additionally, the error is tracked within the `state` object for review and debugging purposes.
           Make sure that `state['process']['steps']` is correctly initialized before calling this function.
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info(errorMessage)
    state['process']['steps'].append(
        {
            'func'    : 'log.errorLog',
            'message' : errorMessage,
            'info'    : info
        }
    )
    

def userOutput(output, state=None):
    """
    Standardizes and manages the display of information to the user, while also logging the output.

    This method is responsible for both printing messages directly to the user and saving the output to 
    the `Agent.state` for logging purposes. It ensures that the displayed output is consistently tracked and appended 
    to the system's state for future reference, debugging, or record-keeping. 

    :param output: The message or information to be displayed to the user. This can be any printable string or data 
                   that the user should see.
    :type output: str
    :param state: The current system state, which tracks the chat process and logs the output. If `state['output']` 
                  is not initialized, this method will set it to the given output. If it is already initialized, 
                  the method appends the new output to the existing log.
    :type state: dict, optional

    :return: The updated state object with the latest output appended.
    :rtype: dict

    :raises KeyError: If the state is not initialized properly or lacks an 'output' key, this method attempts to create 
                      or update the 'output' field within the state object.

    :note: This is the only method in the BRAD framework that directly displays output to the user. The logs are saved 
           in the state object to track all output communications, ensuring a record is available for debugging or review. 
           Make sure that the state is correctly initialized before calling this method.
    """
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: June 20, 2024
          
    print(output)
    if 'output' not in state.keys() or state['output'] is None:
        state['output'] = output
    else:
        state['output'] += output
    
    return state

def is_json_serializable(value):
    """
    Determines whether a given value can be serialized into a valid JSON format.

    This function checks if the provided value can be safely converted to JSON. It is useful for ensuring that data, 
    especially when dealing with logs or state information, can be stored in JSON format without errors. The function 
    handles common data types and catches serialization errors for non-compatible types.

    :param value: The data to be evaluated for JSON serializability. This can be any Python object.
    :type value: Any

    :return: Returns True if the value can be serialized to JSON. Returns False if the value raises a TypeError or 
             OverflowError when attempting serialization.
    :rtype: bool

    :raises TypeError: If the value is of a type that cannot be serialized (e.g., custom objects, functions).
    :raises OverflowError: If the value exceeds the limits of what JSON can represent (e.g., extremely large numbers).

    :note: Lists are treated as JSON serializable by default. For other complex types like dictionaries, sets, or custom 
           objects, serialization will depend on the specific structure and contents.

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
