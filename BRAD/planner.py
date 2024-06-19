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
    chatstatus['process'] = {'name':'PLANNER'}
    template = plannerTemplate()
    PROMPT = PromptTemplate(input_variables=["history", "input"], template=template)
    conversation = ConversationChain(prompt  = PROMPT,
                                     llm     = llm,
                                     verbose = chatstatus['config']['debug'],
                                     memory  = memory,
                                    )
    response = conversation.predict(input=prompt)
    response += '\n\n'
    print(response) if chatstatus['config']['debug'] else None
    while True:
        print('Do you want to proceed with this plan? [Y/N/edit]')
        prompt2 = input('Input >> ')
        if prompt2 == 'Y':
            break
        elif prompt2 == 'N':
            return chatstatus
        else:
            template = plannerEditingTemplate()
            template = template.format(plan=response)
            print(template)
            PROMPT   = PromptTemplate(input_variables=["user_query"], template=template)
            chain    = PROMPT | llm
            
            # Call chain
            response = chain.invoke(prompt2).content.strip() + '\n\n'
            print(response) if chatstatus['config']['debug'] else None
            
    processes = response2processes(response)
    print(processes) if chatstatus['config']['debug'] else None
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


