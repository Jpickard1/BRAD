from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
import re

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
    processes = response2processes(response)
    chatstatus['planned'] = processes
    chatstatus['process'] = {'type': 'PLANNER', 'stages' : processes}
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

def plannerTemplate():
    template = """
You are planning a bioinformatics analysis pipeline to address a user's query. Your task is to outline a multi-step workflow using the available methods listed below. Each method should be used appropriately to ensure a thorough analysis.
For each step in the pipeline, explain the goal of the process as well as provide a prompt to the chatbot that will execute that step. You can use one method per step and the pipeline can be between 3-12 steps.

Available Methods:
1. **Retrieval Augmented Generation**(RAG): Look up literature and documents from a text database.
2. **Web Search for New Literature**(SCRAPE): Search platforms like arXiv, bioRxiv, and PubMed for the latest research.
3. **Bioinformatics Databases**(DATABASE): Utilize databases such as Gene Ontology and Enrichr to perform gene set enrichment analyses.
4. **Run Codes**(CODE): Execute bioinformatics pipelines using Python and MATLAB. Utilize prebuilt pipelines or develop new ones as needed.
5. **Write a Report**(WRITE): Synthesize and summarize information. This can include summarizing database searches, code pipeline results, or creating a final report to encapsulate the entire analysis.

Based on the user's query, create a detailed plan outlining the steps of the analysis. Ensure that each step is clearly defined and makes use of the appropriate method(s) listed above.

Current conversation: {history}\n

User Query: {input}

Plan:
**Step 1 (method, eg. RAG)**:
Prompt: [Description of action for chatbot to do Step 1, i.e. do X, Y, Z]

**Step 2 (method, eg. CODE)**:
Prompt: [Description of action for chatbot to do Step 2, i.e. do X, Y, Z]

**Step 3 (method, eg. DATABASE)**:
Prompt: [Description of action for chatbot to do Step 3, i.e. do X, Y, Z]
...
"""
    return template

