from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
import re
from BRAD.promptTemplates import plannerTemplate, plannerEditingTemplate
from BRAD import log

"""This module is responsible for creating sequences of steps to be run by other modules of BRAD"""

def planner(chatstatus):
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
    while True:
        chatstatus = log.userOutput('Do you want to proceed with this plan? [Y/N/edit]', chatstatus=chatstatus)
        prompt2 = input('Input >> ')
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
            chatstatus = log.userOutput(response, chatstatus=chatstatus)
            
    processes = response2processes(response)
    log.debugLog(processes, chatstatus=chatstatus)
    chatstatus['queue'] = processes
    chatstatus['queue pointer'] = 1 # the 0 object is a place holder
    log.debugLog('exit planner', chatstatus=chatstatus)
    return chatstatus

def response2processes(response):
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
            
            #log.debugLog(stageNum, chatstatus=chatstatus)
            #log.debugLog(module, chatstatus=chatstatus)
            #log.debugLog(prompt, chatstatus=chatstatus)
            #log.debugLog(stage, chatstatus=chatstatus)
            
            processes.append({
                'order':stageNum,
                'module':module,
                'prompt':'/force ' + module + ' ' + prompt[0],
                'description':stage,
            })
    return processes


