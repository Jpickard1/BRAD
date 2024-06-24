from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
import re
from BRAD.promptTemplates import plannerTemplate, plannerEditingTemplate

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
            chatstatus = log.chatstatus(template, chatstatus=chatstatus)
            PROMPT   = PromptTemplate(input_variables=["user_query"], template=template)
            chain    = PROMPT | llm
            
            # Call chain
            response = chain.invoke(prompt2).content.strip() + '\n\n'
            chatstatus = log.userOutput(response, chatstatus=chatstatus)
    processes = response2processes(response)
    chatstatus = log.userOutput(processes, chatstatus=chatstatus)
    chatstatus['planned'] = processes
    chatstatus['process']['stages'] = processes
    return chatstatus

def response2processes(response):
    modules = ['RAG', 'SCRAPE', 'DATABASE', 'CODE', 'WRITE']
    stageStrings = response.split('**Step ')
    processes = []
    for i, stage in enumerate(stageStrings):
        stageNum = i
        found_modules = [module for module in modules if module in stage]
        if len(found_modules) == 0:
            continue
        prompt = re.findall(r'Prompt: (.*?)\n', stage)
        for module in found_modules:
            # we have both
            
            #log.debugLog(stageNum, chatstatus=chatstatus)
            #log.debugLog(module, chatstatus=chatstatus)
            #log.debugLog(prompt, chatstatus=chatstatus)
            #log.debugLog(stage, chatstatus=chatstatus)
            
            # and (I'm not really sure if I should use print statements or our function here)
            
            # print(stageNum)
            # print(module)
            # print(prompt)
            # print(stage)
            processes.append({
                'order':stageNum,
                'module':module,
                'prompt':'/force ' + module + ' ' + prompt[0],
                'description':stage,
            })
    return processes


