"""
This module provides functions for generating and managing agentic workflows or pipelines with multiple steps
each to be executed by individual tool modules. The main method, `planner`, builds pipelines based on user input
and the available modules. It also supports selecting pre-existing pipelines or designing new ones 
when necessary.
"""


import os
import re
import json
import difflib

from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain_core.prompts.prompt import PromptTemplate

from BRAD.promptTemplates import plannerTemplate, plannerEditingTemplate, plannerTemplateForLibrarySelection
from BRAD import log

def planner(state):
    """
    Generates a plan based on the user prompt using a language model, allows the user 
    to review and edit the plan, and then updates the state with the finalized plan.

    Args:
        state (dict): A dictionary containing the LLM, user prompt, vector database, 
                           memory, and configuration settings for the planning process.

    Returns:
        dict: The updated state containing the finalized plan and any modifications 
              made during the process.

    """
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: June 16, 2024

    # Dev. Comments:
    # -------------------
    # This function initializes a chat session and the state variable
    #
    # History:
    # - 2024-06-16: 1st draft of this method
    # - 2024-07-25: this function is refactors to allow the planner to save
    #               new pipelines and rerun old pipelines.
    #
    # Issues:
    # - The parsing of new/custom pipelines into prompts and a queue doesn't work
    #   all that well
    # - The queue should be implemented with a class rather than with only a list,
    #   IP, and set of prompts
    #
    # TODOs:
    # - add code to let BRAD fill in the template of a prebuilt pipeline
    # - add code to save new pipelines

    llm      = state['llm']              # get the llm
    prompt   = state['prompt']           # get the user prompt
    memory   = state['memory']           # get the memory of the model

    # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Select to use a prebuilt tempalte or design out own
    # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    template = plannerTemplateForLibrarySelection()
    pipelines, pipelineSummary = getKnownPipelines(state)
    template = template.format(pipeline_list=pipelineSummary)
    PROMPT = PromptTemplate(input_variables=["input"], template=template)
    conversation = LLMChain(prompt  = PROMPT,
                            llm     = llm,
                            verbose = state['config']['debug'],
                            )
    response = conversation.predict(input=prompt)
    log.debugLog(response, state=state)
    pipelineSelection = response.split('\n')[0].split(':')[1]
    pipeline_names = [name.upper() for name in pipelines.keys()]
    pipeline_names.append('CUSTOM')
    selected_pipeline = difflib.get_close_matches(pipelineSelection.upper(), pipeline_names, n=1, cutoff=0.0)
    if len(selected_pipeline) == 0:
        selected_pipeline = "CUSTOM"
    else:
        selected_pipeline = selected_pipeline[0]
    log.debugLog(f'selected_pipeline={selected_pipeline}', state=state)

    state['process']['steps'].append(
        log.llmCallLog(
            llm          = llm,
            prompt       = PROMPT,
            input        = prompt,
            output       = response,
            parsedOutput = {
                'selected pipeline': selected_pipeline
            },
            purpose      = 'determine if a known pipeline can be used or a new one si required'
        )
    )

    # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Building a custom pipeline
    # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    if selected_pipeline == 'CUSTOM':
        template = plannerTemplate()
        PROMPT = PromptTemplate(input_variables=["history", "input"], template=template)
        conversation = ConversationChain(prompt  = PROMPT,
                                         llm     = llm,
                                         verbose = state['config']['debug'],
                                         memory  = memory,
                                        )
        response = conversation.predict(input=prompt)
        response += '\n\n'
        state = log.userOutput(response, state=state)
        state['process']['steps'].append(log.llmCallLog(llm          = llm,
                                                             prompt       = PROMPT,
                                                             memory       = memory,
                                                             input        = prompt,
                                                             output       = response,
                                                             parsedOutput = {
                                                                 'output' : response,
                                                             },
                                                             purpose      = 'generate a proposed set of prompts'
                                                            )
                                             )
        while True:
            state = log.userOutput('Do you want to proceed with this plan? [Y/N/edit]', state=state)
            prompt2 = input('Input >> ')
            state['process']['steps'].append(
                {
                    'func'           : 'planner.planner',
                    'prompt to user' : 'Do you want to proceed with this plan? [Y/N/edit]',
                    'input'          : prompt2,
                    'purpose'        : 'get new user input'
                }
            )
            if prompt2 == 'Y':
                break
            elif prompt2 == 'N':
                return state
            else:
                template = plannerEditingTemplate()
                template = template.format(plan=response)
                log.debugLog(template, state=state)
                PROMPT   = PromptTemplate(input_variables=["user_query"], template=template)
                chain    = PROMPT | llm
                
                # Call chain
                response = chain.invoke(prompt2).content.strip() + '\n\n'
                state['process']['steps'].append(log.llmCallLog(llm          = llm,
                                                                     prompt       = PROMPT,
                                                                     # memory should be included in this chain
                                                                     # memory       = memory,
                                                                     input        = prompt2,
                                                                     output       = response,
                                                                     parsedOutput = {
                                                                         'output' : response,
                                                                     },
                                                                     purpose      = 'update the proposed set of prompts'
                                                                    )
                                                     )
                state = log.userOutput(response, state=state)

        processes = response2processes(response)
        log.debugLog(processes, state=state)
        state['process']['steps'].append(
            {
                'func' : 'planner.planner',
                'what' : 'designed a new pipeline'
            }
        )

        # Check if the chat is interactive
        if state['interactive']:
            # Prompt the user to decide if they want to save the pipeline to a file
            state = log.userOutput('Would you like to save this pipeline to a file? [Y/N]', state=state)
            saveNewPipeline = input(">>>")
        
            # If the user chooses to save the pipeline
            if saveNewPipeline.upper() == "Y":
                # Prompt the user to enter a name for the pipeline
                state = log.userOutput('Enter a name for your pipeline', state=state)
                fname = input(">>>")
        
                # Prompt the user to enter a description for the pipeline
                state = log.userOutput('Enter a description of your pipeline', state=state)
                description = input(">>>")
        
                # Create a dictionary to store the pipeline information
                pipelineJSONdict = {
                    'name': fname,  # Use 'fname' instead of 'name' to match the input
                    'description': description,
                    'queue': processes  # Fixed typo: 'proecesses' to 'processes'
                }
                for k in range(len(pipelineJSONdict['queue'])):
                    pipelineJSONdict['queue'][k]['output'] = []
        
                # Get the directory path to save the pipeline file
                pipelines_dir = state['config']['PLANNER']['path']
        
                # Construct the full file path
                filepath = os.path.join(pipelines_dir, f"{fname}.json")
        
                # Save the pipelineJSONdict to a file
                try:
                    with open(filepath, 'w') as f:
                        json.dump(pipelineJSONdict, f, indent=4)
                    state = log.userOutput(f"Pipeline saved successfully to {filepath}", state=state)
                except Exception as e:
                    state = log.userOutput(f"Error saving pipeline: {e}", state=state)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Parameterize a predesigned pipeline
    # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    else:
        pipeline = pipelines[selected_pipeline]
        # Initialize processes dictionary from the loaded pipeline queue
        loadedProcesses, processes = pipeline['queue'], {}
        
        # If loadedProcesses is a dictionary
        if isinstance(loadedProcesses, dict):
            # Iterate over the items in the dictionary
            for key, value in loadedProcesses.items():
                # Convert the key to an integer if it is a string representation of a digit
                new_key = int(key) if isinstance(key, str) and key.isdigit() else key
                processes[new_key] = value
        
        # If loadedProcesses is a list
        elif isinstance(loadedProcesses, list):
            # Iterate over the list with indices
            for key, value in enumerate(loadedProcesses):
                processes[key] = value
                
        # If loadedProcesses is neither a dictionary nor a list, issue a warning
        else:
            log.debugLog("Warning: loadedProcesses is neither a dictionary nor a list", state=state)
        print('Display pipeline')
        state = displayPipeline2User(processes, state=state)
        state['process']['steps'].append(
            {
                'func' : 'planner.planner',
                'what' : 'loaded an older pipeline'
            }
        )

    state['interactive'] = False
    state['queue'] = processes
    state['queue pointer'] = 1 # the 0 object is a place holder
    state['process']['steps'].append(
        {
            'func' : 'planner.planner',
            'what' : 'set the queue and set the queue pointer to 1'
        }
    )
    return state

def displayPipeline2User(process, state=None):
    """
    Displays the steps of the process pipeline to the user, logging each step.
    
    This function iterates through the steps of a process pipeline, outputs each step to the user in a
    standardized format, and updates the chat status with the logged outputs. Each step is labeled as
    **Step X**, where X is the key, followed by the corresponding value of the process step.
    
    :param process: A dictionary representing the process pipeline. Each key-value pair corresponds 
                    to a step in the process, where the key is the step number or name, and the value 
                    is the step's description or details.
    :type process: dict
    :param state: The current chat status dictionary to which the output will be appended. If not 
                       provided, a default value of `None` is used.
    :type state: dict, optional
    
    :return: The updated chat status after logging all process steps.
    :rtype: dict
    
    """
    for key, value in process.items():
        state=log.userOutput("** Step " + str(key) + "**", state=state)
        state=log.userOutput(str(value), state=state)
        state=log.userOutput("\n", state=state)
    return state

def response2processes(response):
    """
    Converts a response string into a list of processes, each with an order, module, 
    prompt, and description. It identifies modules from a predefined list and parses 
    the response into steps based on these modules.
    
    Args:
        response (str): The response string containing the steps and corresponding details.
    
    Returns:
        list: A list of dictionaries, each representing a process with the following keys:
              - 'order': The order of the process.
              - 'module': The module associated with the process.
              - 'prompt': The prompt to invoke the process.
              - 'description': A description of the process.
    
    Example:
        >>> response = '''
        ... **Step 1: RAG**
        ... Prompt: Retrieve documents related to AI research
        ... **Step 2: SCRAPE**
        ... Prompt: Scrape data from the specified website
        ... '''
        >>> processes = response2processes(response)
        >>> print(processes)
        [
            {
                'order': 0,
                'module': 'PLANNER',
                'prompt': None,
                'description': 'This step designed the plan. It is placed in the queue because we needed a placeholder for 0-indexed lists.',
            },
            {
                'order': 1,
                'module': 'RAG',
                'prompt': '/force RAG Retrieve documents related to AI research',
                'description': '**Step 1: RAG\\nPrompt: Retrieve documents related to AI research\\n',
            },
            {
                'order': 2,
                'module': 'SCRAPE',
                'prompt': '/force SCRAPE Scrape data from the specified website',
                'description': '**Step 2: SCRAPE\\nPrompt: Scrape data from the specified website\\n',
            }
        ]
    """
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: June 16, 2024

    # History:
    # - 2024-06-16: 1st draft of this method
    # - 2024-08-02: this was changed to split the prompts at the word "Step" as
    #               opposed to "**Step ". It remains a problem that this will be
    #               very brittle to the use by different LLMs.
    #
    # Issues:
    # - This is VERY brittle to use by different LLMs
    modules = ['RAG', 'SCRAPE', 'DATABASE', 'CODE', 'WRITE', 'ROUTER']
    stageStrings = response.split('Step')
    processes = [
        {
            'order'  : 0,
            'module' : 'PLANNER',
            'prompt' : None,
            'description' : 'This step designed the plan. It is placed in the queue because we needed a place holder for 0 indexed lists.',
        }
    ]
    print(stageStrings)
    for i, stage in enumerate(stageStrings):
        stageNum = i
        found_modules = [module for module in modules if module in stage]
        if len(found_modules) == 0:
            continue
        prompt = re.findall(r'Prompt: (.*?)\n', stage)
        for module in found_modules:
            
            processes.append({
                'order':stageNum,
                'module':module,
                'prompt':'/force ' + module + ' ' + stage, # + prompt[0],
                'description':stage,
            })
    return processes

def getKnownPipelines(state):
    """
    This function reads all available pipeline JSON files in the 'pipelines' directory
    and extracts their 'name' and 'description' fields. It formulates a summary string
    that can be used as input to an LLM prompt for selecting the appropriate pipeline.

    Returns:
        tuple: A tuple containing two elements:
            - pipelines (list): A list of dictionaries where each dictionary represents
              a pipeline read from a JSON file.
            - summary (str): A formatted string summarizing the 'name' and 'description'
              of each pipeline.
    
    """
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: July 25, 2024
    
    # Get the path to the 'pipelines' directory
    # current_script_path = os.path.abspath(__file__)
    # current_script_dir = os.path.dirname(current_script_path)
    # pipelines_dir = os.path.join(current_script_dir, 'pipelines')
    pipelines_dir = state['config']['PLANNER']['path']

    # Initialize an empty list to store pipeline dictionaries
    pipelines = {}
    summary = ""
    # Read all JSON files in the 'pipelines' directory
    for file_name in os.listdir(pipelines_dir):
        if file_name.endswith('.json'):
            file_path = os.path.join(pipelines_dir, file_name)
            with open(file_path, 'r') as file:
                pipeline = json.load(file)
                pipelines[pipeline['name'].upper()] = {
                    'path' : file_path,
                    'description': pipeline['description'],
                    'queue': pipeline['queue']
                }
                # Extract 'name' and 'description' to build the summary string
                summary += f"Name: {pipeline['name']}\tDescription: {pipeline['description']}\n"
    return pipelines, summary
