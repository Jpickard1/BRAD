"""
This module manages routing decisions for both individual user inputs and agentic workflows using the semantic_router 
library.

Scope
-----

It serves two key purposes within the BRAD framework: determining which tool or module to use for a specific 
input, and orchestrating multi-step workflows by selecting the next stage in the process.

1. **Tool Selection for Single Inputs**:  
   When receiving input from a user, the router evaluates the context and selects the appropriate tool or module to handle 
   the input. This ensures that the most suitable functionality is used to process each user request, improving the efficiency 
   and accuracy of interactions. Semantic routing is used for this level or organization, and the routing database may be dynamically
   updated as more user queries are recieved.

2. **Agentic Workflow Management**:  
   For agent-driven processes, the router manages the progression of tasks in a predefined workflow designed by the `planner`. It selects the next 
   step based on the current state and the broader workflow plan, ensuring smooth transitions between stages. This feature is organized by the LLM and
   is essential for workflows that involve multiple steps, tools, or decision points.

Available Methods
-----------------

This module contains the following methods:

"""

import os
import json
from semantic_router import Route
from semantic_router.layer import RouteLayer
from semantic_router.encoders import HuggingFaceEncoder

from langchain.prompts import PromptTemplate
from langchain_core.prompts.prompt import PromptTemplate

from BRAD.promptTemplates import rerouteTemplate
from BRAD import log
from BRAD import utils

def reroute(state):
    """
    Reroutes the conversation flow for agentic workflows based on the current queue pointer and the user prompt.
    
    This method uses the agent history and ongoing conversation, along with an LLM, 
    to determine the subsequent step in the agentic workflow designed by the planner.
    
    :param state: A dictionary that holds the current state of the conversation, including the language model, 
                       the user prompt, the process queue, and any other relevant information necessary for 
                       effectively rerouting the conversation.
    
    :returns: An updated state dictionary reflecting changes made during the rerouting process.
    :rtype: dict
    """
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: June 24, 2024
    log.debugLog('Call to REROUTER', state=state)
    llm = state['llm']
    prompt = state['prompt']
    queue  = state['queue']

    # Build chat history
    chatlog = json.load(open(os.path.join(state['output-directory'], 'log.json')))
    history = ""
    for i in chatlog.keys():
        history += "========================================"
        history += '\n'
        history += "Input: "  + chatlog[i]['prompt']  + r'\n\n'
        history += "Output: " + chatlog[i]['output'] + r'\n\n'
    log.debugLog(history, state=state)
    
    # Put history into the conversation
    template = rerouteTemplate()
    template = template.format(chathistory=history,
                               step_number=state['queue pointer'])
    PROMPT = PromptTemplate(input_variables=["user_query"], template=template)
    chain = PROMPT | state['llm']
    res = chain.invoke(prompt)
    
    # Extract output
    log.debugLog(res, state=state)
    log.debugLog(res.content, state=state)
    # nextStep = int(res.content.split('=')[1].split('\n')[0].strip())
    nextStep = utils.find_integer_in_string(res.content.split('\n')[0])
    log.debugLog('EXTRACTED NEXT STEP', state=state)
    log.debugLog('Next Step=' + str(nextStep), state=state)
    log.debugLog(f'(nextStep is None)={(nextStep is None)}', state=state)
    if nextStep is None:
        nextStep = state['queue pointer'] + 1
    if str(nextStep) not in prompt:
        log.debugLog(f'the next step identified was not valid according to the rerouting instructions. As a result, state["queue pointer"]={state["queue pointer"]+1}', state=state)
        nextStep = state['queue pointer'] + 1
    state['process']['steps'].append(log.llmCallLog(
        llm     = llm,
        prompt  = template,
        input   = prompt,
        output  = res.content,
        parsedOutput = {'next step': nextStep},
        purpose = "Route to next step in pipeline"
    ))
    
    # Modify the queued prompts
    state['queue pointer'] = nextStep
    return state

def read_prompts(file_path):
    """
    Reads a text file where each line contains a sentence and returns a list of non-empty sentences. The
    files contain previously used user inputs associated with tool modules.

    This function opens the specified text file, reads each line, and returns a list containing the 
    sentences. Leading and trailing whitespace is removed from each line, and empty lines are ignored.

    :param file_path: The path to the text file to be read.
    :type file_path: str

    :raises FileNotFoundError: If the specified file cannot be found.

    :return: A list of non-empty sentences extracted from the text file.
    :rtype: list[str]
    """
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: May 19, 2024
    sentences = []
    with open(file_path, 'r') as file:
        for line in file:
            # Strip any leading/trailing whitespace characters (including newline)
            sentence = line.strip()
            if sentence:  # Avoid adding empty lines
                sentences.append(sentence)
    return sentences

def add_sentence(file_path, sentence):
    """
    Adds a new sentence to the specified text file. These files contain user inputs associated with tool modules.

    :param file_path: The path to the text file where the sentence is to be added.
    :type file_path: str
    :param sentence: The sentence to be added to the text file.
    :type sentence: str

    :raises FileNotFoundError: If the specified file does not exist or cannot be created.

    """
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: May 19, 2024
    with open(file_path, 'a') as file:
        file.write(sentence.strip() + '\n')

def getRouterPath(file):
    """
    Constructs and returns the absolute path to a file located in the 'routers' directory.

    This function determines the current script's directory and constructs the absolute path 
    to a specified file within the 'routers' subdirectory.

    :param file: The name of the file whose path is to be constructed.
    :type file: str

    :return: The absolute path to the specified file in the 'routers' directory.
    :rtype: str
    """
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: June 7, 2024
    current_script_path = os.path.abspath(__file__)
    current_script_dir = os.path.dirname(current_script_path)
    file_path = os.path.join(current_script_dir, 'routers', file) #'enrichr.txt')
    return file_path
    
def getRouter():
    """
    Constructs a semantic router layer configured to determine the correct tool module for user queries.
    This routing layer uses the routing files and previously used user inputs as data to predict which
    tool module to use. These routing files are updated over time.

    :return: A router layer configured with predefined routes for tasks such as querying Enrichr, web scraping, and generating tables.
    :rtype: RouteLayer

    """
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: May 16, 2024
    routeGget = Route(
        name = 'GGET',
        utterances = read_prompts(getRouterPath('enrichr.txt'))
    )
    routeScrape = Route(
        name = 'SCRAPE',
        utterances = read_prompts(getRouterPath('scrape.txt'))
    )
    routeRAG = Route(
        name = 'RAG',
        utterances = read_prompts(getRouterPath('rag.txt'))
    )
    routeTable = Route(
        name = 'TABLE',
        utterances = read_prompts(getRouterPath('table.txt'))
    )
    routeData = Route(
        name = 'DATA',
        utterances = read_prompts(getRouterPath('data.txt'))
    )
    routeMATLAB = Route(
        name = 'MATLAB',
        utterances = read_prompts(getRouterPath('matlab.txt'))
    )
    routePython = Route(
        name = 'PYTHON',
        utterances = read_prompts(getRouterPath('python.txt'))
    )
    routePlanner = Route(
        name = 'PLANNER',
        utterances = read_prompts(getRouterPath('planner.txt'))
    )
    routeCode = Route(
        name = 'CODE',
        utterances = read_prompts(getRouterPath('code.txt'))
    )
    routeWrite = Route(
        name = 'WRITE',
        utterances = read_prompts(getRouterPath('write.txt'))
    )
    routeRoute = Route(
        name = 'ROUTER',
        utterances = read_prompts(getRouterPath('router.txt'))
    )
    encoder = HuggingFaceEncoder(device='cpu')
    routes = [routeGget,
              routeScrape,
              routeTable,
              routeRAG,
              routeData,
              routeMATLAB,
              routePython,
              routePlanner,
              routeCode,
              routeWrite,
              routeRoute
             ]
    router = RouteLayer(encoder=encoder, routes=routes)    
    return router

def buildRoutes(prompt):
    """
    Constructs routes based on the provided prompt and updates the corresponding text files with the new prompts.

    This function processes the input prompt to identify specific commands (e.g., `/force`) 
    and builds a new prompt that is then appended to the designated router text file based on 
    the identified route. Each command maps to a specific file, and if the command is not 
    recognized, a KeyError will be raised.

    :param prompt: The prompt containing the information to be added to the router.
    :type prompt: str
    :raises KeyError: If the specified route is not found in the paths dictionary.

    """
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: May 19, 2024
    words = prompt.split(' ')
    rebuiltPrompt = ''
    i = 0
    while i < (len(words)):
        if words[i] == '/force':
            route = words[i + 1].upper()
            i += 1
        else:
            rebuiltPrompt += (' ' + words[i])
        i += 1
    paths = {
        'DATABASE': getRouterPath('enrichr.txt'),
        'SCRAPE'  : getRouterPath('scrape.txt'),
        'RAG'     : getRouterPath('rag.txt'),
        'TABLE'   : getRouterPath('table.txt'),
        'DATA'    : getRouterPath('data.txt'),
        'SNS'     : getRouterPath('sns.txt'),
        'MATLAB'  : getRouterPath('matlab.txt'),
        'PYTHON'  : getRouterPath('python.txt'),
        'PLANNER' : getRouterPath('planner.txt'),
        'CODE'    : getRouterPath('code.txt'),
        'WRITE'   : getRouterPath('write.txt'),
        'ROUTER'  : getRouterPath('router.txt')
    }
    filepath = paths[route]
    add_sentence(filepath, rebuiltPrompt)

