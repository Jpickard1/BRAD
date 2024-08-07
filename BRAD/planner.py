from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
import re
from BRAD.promptTemplates import plannerTemplate, plannerEditingTemplate
from BRAD import log

"""This module is responsible for creating sequences of steps to be run by other modules of BRAD"""

def planner(chatstatus):
    """
    Generates a plan based on the user prompt using a language model, allows the user 
    to review and edit the plan, and then updates the chatstatus with the finalized plan.

    Args:
        chatstatus (dict): A dictionary containing the LLM, user prompt, vector database, 
                           memory, and configuration settings for the planning process.

    Returns:
        dict: The updated chatstatus containing the finalized plan and any modifications 
              made during the process.

    Example
    -------
    >>> chatstatus = {
    ...     'llm': llm_instance,
    ...     'prompt': "Plan my week",
    ...     'databases': {'RAG': vectordb_instance},
    ...     'memory': memory_instance,
    ...     'config': {
    ...         'debug': True
    ...     },
    ...     'process': {'steps': []}
    ... }
    >>> updated_chatstatus = planner(chatstatus)
    """
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: June 16, 2024
    llm      = chatstatus['llm']              # get the llm
    prompt   = chatstatus['prompt']           # get the user prompt
    vectordb = chatstatus['databases']['RAG'] # get the vector database
    memory   = chatstatus['memory']           # get the memory of the model
    template = plannerTemplate()
    PROMPT = PromptTemplate(input_variables=["history", "input"], template=template)
    conversation = ConversationChain(prompt  = PROMPT,
                                     llm     = llm,
                                     verbose = chatstatus['config']['debug'],
                                     memory  = memory,
                                    )
    response = conversation.predict(input=prompt)
    response += '\n\n'
    chatstatus = log.userOutput(response, chatstatus=chatstatus)
    chatstatus['process']['steps'].append(log.llmCallLog(llm          = llm,
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
        chatstatus = log.userOutput('Do you want to proceed with this plan? [Y/N/edit]', chatstatus=chatstatus)
        prompt2 = input('Input >> ')
        chatstatus['process']['steps'].append(
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
            return chatstatus
        else:
            template = plannerEditingTemplate()
            template = template.format(plan=response)
            log.debugLog(template, chatstatus=chatstatus)
            PROMPT   = PromptTemplate(input_variables=["user_query"], template=template)
            chain    = PROMPT | llm
            
            # Call chain
            response = chain.invoke(prompt2).content.strip() + '\n\n'
            chatstatus['process']['steps'].append(log.llmCallLog(llm          = llm,
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
            chatstatus = log.userOutput(response, chatstatus=chatstatus)
            
    processes = response2processes(response)
    log.debugLog(processes, chatstatus=chatstatus)
    chatstatus['queue'] = processes
    chatstatus['queue pointer'] = 1 # the 0 object is a place holder
    chatstatus['process']['steps'].append(
        {
            'func' : 'planner.planner',
            'what' : 'set the queue and set the queue pointer to 1'
        }
    )
    return chatstatus

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

    Example
    -------
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
            'description': 'This step designed the plan. It is placed in the queue because we needed a place holder for 0 indexed lists.',
        },
        {
            'order': 1,
            'module': 'RAG',
            'prompt': '/force RAG Retrieve documents related to AI research',
            'description': '**Step 1: RAG\nPrompt: Retrieve documents related to AI research\n',
        },
        {
            'order': 2,
            'module': 'SCRAPE',
            'prompt': '/force SCRAPE Scrape data from the specified website',
            'description': '**Step 2: SCRAPE\nPrompt: Scrape data from the specified website\n',
        }
    ]
    """
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: June 16, 2024
    modules = ['RAG', 'SCRAPE', 'DATABASE', 'CODE', 'WRITE', 'ROUTER']
    stageStrings = response.split('**Step ')
    processes = [
        {
            'order'  : 0,
            'module' : 'PLANNER',
            'prompt' : None,
            'description' : 'This step designed the plan. It is placed in the queue because we needed a place holder for 0 indexed lists.',
        }
    ]
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
                'prompt':'/force ' + module + ' ' + prompt[0],
                'description':stage,
            })
    return processes


