"""
Bioinformatics Database
-----------------------

This module provides functionality to retrieve structured data from bioinformatics databases 
such as Enrichr and Gene Ontology. User queries are processed by an LLM to select and query an
appropriate database.

Main Methods
~~~~~~~~~~~~

1. geneDBRetriever:
    This method selects which database and search terms or files to use. After formualting the query terms
    or loading data from a file, the method corresponding to each database is used for the corresponding query.

Available Methods
~~~~~~~~~~~~~~~~~

This module has the following methods:

"""


import time

from langchain import PromptTemplate, LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain_community.callbacks import get_openai_callback

from BRAD.gene_ontology import geneOntology
from BRAD.enrichr import queryEnrichr
from BRAD import utils
from BRAD.promptTemplates import geneDatabaseCallerTemplate, geneDatabaseLLMreply
from BRAD import log

def geneDBRetriever(state):
    """
    Retrieves gene information from a specified database based on the user query. 
    It uses a language model to determine the appropriate database and performs 
    the search, handling various configurations and logging the process.
    
    :param state: A dictionary containing the user query, language model, 
                       configurations, and other necessary data for the retrieval process.
    :type state: dict
    
    :returns: The updated state containing the results of the database search 
              and any modifications made during the process.
    :rtype: dict
    """
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: June 6, 2024
    
    query    = state['prompt']
    llm      = state['llm']              # get the llm
    memory   = state['memory']           # get the memory of the model
    # memory = ConversationBufferMemory(ai_prefix="BRAD")
    
    # Define the mapping of keywords to functions
    database_functions = {
        'ENRICHR'      : queryEnrichr,
        'GENEONTOLOGY' : geneOntology,
    }

    # Identify the database and the search terms
    template = geneDatabaseCallerTemplate()

    tablesInfo = getTablesFormatting(state['tables'])
    filled_template = template.format(tables=tablesInfo)
    PROMPT = PromptTemplate(input_variables=["history", "input"], template=filled_template)
    
    conversation = ConversationChain(prompt  = PROMPT,
                                     llm     = llm,
                                     verbose = state['config']['debug'],
                                     memory  = memory,
                                    )
    # Invoke LLM tracking its usage
    start_time = time.time()
    with get_openai_callback() as cb:
        chainResponse = conversation.predict(input=query)        
    responseDetails = {
        'content' : chainResponse,
        'time' : time.time() - start_time,
        'call back': {
            "Total Tokens": cb.total_tokens,
            "Prompt Tokens": cb.prompt_tokens,
            "Completion Tokens": cb.completion_tokens,
            "Total Cost (USD)": cb.total_cost
        }
    }
    
    log.debugLog(chainResponse, state=state)    # Print gene list if debugging
    response = parse_llm_response(chainResponse, state)

    state['process']['steps'].append(
        log.llmCallLog(
            llm          = llm,
            prompt       = PROMPT,
            input        = query,
            output       = responseDetails,
            parsedOutput = response,
            purpose      = 'Select database'
        )
    )

    log.debugLog(response, state=state)    # Print gene list if debugging

    similarity_to_enrichr      = utils.word_similarity(response['database'], "ENRICHR")
    similarity_to_geneontology = utils.word_similarity(response['database'], "GENEONTOLOGY")
    if similarity_to_enrichr > similarity_to_geneontology:
        database = "ENRICHR"
    else:
        database = "GENEONTOLOGY"

    dbCaller = database_functions[database]
    state['process']['database-function'] =  dbCaller
    
    geneList = []
    if response['load'] == 'True':
        state, geneList = utils.loadFromFile(state)
    else:
        geneList = response['genes']

    if len(geneList) > state['config']['DATABASE']['max_search_terms']:
        geneList = geneList[:state['config']['DATABASE']['max_search_terms']]

    # Print gene list if debugging
    log.debugLog(geneList, state=state)

    try:
        state = dbCaller(state, geneList)
    except Exception as e:
        output = f'Error occurred while searching database: {e}'
        log.errorLog(output, info='geneDatabaseCaller.geneDBRetriever', state=state)

    # Generate LLM response based on the search results
    template = geneDatabaseLLMreply()
    filled_template = template.format(
        database_selection=database,
        search_results=state['output']
    )
    PROMPT = PromptTemplate(input_variables=["history", "input"], template=filled_template)
    conversation = ConversationChain(prompt  = PROMPT,
                                     llm     = llm,
                                     verbose = state['config']['debug'],
                                     memory  = memory,
                                    )
    # Invoke LLM tracking its usage
    start_time = time.time()
    with get_openai_callback() as cb:
        chainResponse = conversation.predict(input=query)        
    responseDetails = {
        'content' : chainResponse,
        'time' : time.time() - start_time,
        'call back': {
            "Total Tokens": cb.total_tokens,
            "Prompt Tokens": cb.prompt_tokens,
            "Completion Tokens": cb.completion_tokens,
            "Total Cost (USD)": cb.total_cost
        }
    }    
    log.debugLog(chainResponse, state=state)    # Print gene list if debugging
    response = chainResponse # parse_llm_response(chainResponse, state)
    state['process']['steps'].append(
        log.llmCallLog(
            llm          = llm,
            prompt       = PROMPT,
            input        = query,
            output       = responseDetails,
            parsedOutput = response,
            purpose      = 'LLM response after search'
        )
    )
    print(f"{response=}")
    state['output'] += response
    return state

def parse_llm_response(response, state):
    """
    Parses the LLM response to extract the database name and search terms.
    
    :param response: The response from the LLM.
    :type response: str
    
    :returns: A dictionary with the database name and a list of search terms.
    :rtype: dict
    """
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: June 26, 2024
    
    # Initialize an empty dictionary to hold the parsed data
    parsed_data = {}

    # Split the response into lines
    response = response.replace("'", "")
    response = response.replace('"', "")
    log.debugLog(response, state=state)
    lines = response.strip().split('\n')

    # Extract the database name
    database_line = lines[0].replace("database:", "").strip()
    parsed_data["database"] = database_line

    genes_line = lines[1].replace("genes:", "").strip()
    parsed_data["genes"] = genes_line.split(',')

    code_line = lines[2].replace("load:", "").strip()
    parsed_data["load"] = code_line.split(',')[0]

    return parsed_data
    
def getTablesFormatting(tables):
    """
    Formats the columns of each table in the given dictionary into a readable string. 
    For each table, it lists the first 10 column names, appending '...' if there are more 
    than 10 columns.
    
    :param tables: A dictionary where keys are table names and values are pandas DataFrame objects.
    :type tables: dict
    
    :returns: A formatted string listing the first 10 column names of each table.
    :rtype: str
    """
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: June 6, 2024
    tablesString = ""
    for tab in tables:
        columns_list = list(tables[tab].columns)
        truncated_columns = columns_list[:10]  # Get the first 10 entries
        if len(columns_list) > 10:
            truncated_columns.append("...")  # Add '...' if the list is longer than 10
        tablesString += tab + '.columns = ' + str(truncated_columns) + '\n'
    return tablesString


