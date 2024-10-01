import time

from langchain import PromptTemplate, LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain_community.callbacks import get_openai_callback

from BRAD.gene_ontology import geneOntology
from BRAD.enrichr import queryEnrichr
from BRAD import utils
from BRAD.promptTemplates import geneDatabaseCallerTemplate
from BRAD import log

def geneDBRetriever(chatstatus):
    """
    Retrieves gene information from a specified database based on the user query. 
    It uses a language model to determine the appropriate database and performs 
    the search, handling various configurations and logging the process.

    Args:
        chatstatus (dict): A dictionary containing the user query, language model, 
                           configurations, and other necessary data for the retrieval process.

    Returns:
        dict: The updated chatstatus containing the results of the database search 
              and any modifications made during the process.

    Example
    -------
    >>> chatstatus = {
    ...     'llm': llm_instance,
    ...     'prompt': "Retrieve gene information",
    ...     'config': {
    ...         'debug': True,
    ...         'DATABASE': {'max_search_terms': 50}
    ...     },
    ...     'tables': [],
    ...     'process': {'steps': []}
    ... }
    >>> chatstatus = geneDBRetriever(chatstatus)
    """
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: June 6, 2024
    
    query    = chatstatus['prompt']
    llm      = chatstatus['llm']              # get the llm
    # memory   = chatstatus['memory']           # get the memory of the model
    memory = ConversationBufferMemory(ai_prefix="BRAD")
    
    # Define the mapping of keywords to functions
    database_functions = {
        'ENRICHR'      : queryEnrichr,
        'GENEONTOLOGY' : geneOntology,
    }

    # Identify the database and the search terms
    template = geneDatabaseCallerTemplate()

    tablesInfo = getTablesFormatting(chatstatus['tables'])
    filled_template = template.format(tables=tablesInfo)
    PROMPT = PromptTemplate(input_variables=["history", "input"], template=filled_template)
    
    conversation = ConversationChain(prompt  = PROMPT,
                                     llm     = llm,
                                     verbose = chatstatus['config']['debug'],
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
    
    log.debugLog(chainResponse, chatstatus=chatstatus)    # Print gene list if debugging
    response = parse_llm_response(chainResponse, chatstatus)

    chatstatus['process']['steps'].append(
        log.llmCallLog(
            llm          = llm,
            prompt       = PROMPT,
            input        = query,
            output       = responseDetails,
            parsedOutput = response,
            purpose      = 'Select database'
        )
    )

    log.debugLog(response, chatstatus=chatstatus)    # Print gene list if debugging

    similarity_to_enrichr      = utils.word_similarity(response['database'], "ENRICHR")
    similarity_to_geneontology = utils.word_similarity(response['database'], "GENEONTOLOGY")
    if similarity_to_enrichr > similarity_to_geneontology:
        database = "ENRICHR"
    else:
        database = "GENEONTOLOGY"

    dbCaller = database_functions[database]
    chatstatus['process']['database-function'] =  dbCaller
    
    geneList = []
    if response['load'] == 'True':
        chatstatus, geneList = utils.loadFromFile(chatstatus)
    else:
        geneList = response['genes']

    if len(geneList) > chatstatus['config']['DATABASE']['max_search_terms']:
        geneList = geneList[:chatstatus['config']['DATABASE']['max_search_terms']]

    # Print gene list if debugging
    log.debugLog(geneList, chatstatus=chatstatus)

    try:
        chatstatus = dbCaller(chatstatus, geneList)
    except Exception as e:
        output = f'Error occurred while searching database: {e}'
        log.errorLog(output, info='geneDatabaseCaller.geneDBRetriever', chatstatus=chatstatus)
    return chatstatus

def parse_llm_response(response, chatstatus):
    """
    Parses the LLM response to extract the database name and search terms.
    
    Parameters:
    response (str): The response from the LLM.
    
    Returns:
    dict: A dictionary with the database name and a list of search terms.
    """
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: June 26, 2024
    
    # Initialize an empty dictionary to hold the parsed data
    parsed_data = {}

    # Split the response into lines
    response = response.replace("'", "")
    response = response.replace('"', "")
    log.debugLog(response, chatstatus=chatstatus)
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

    Args:
        tables (dict): A dictionary where keys are table names and values are pandas DataFrame objects.

    Returns:
        str: A formatted string listing the first 10 column names of each table.

    Example
    -------
    >>> tables = {
    ...     'table1': pd.DataFrame(columns=['col1', 'col2', 'col3', 'col4', 'col5', 'col6', 'col7', 'col8', 'col9', 'col10', 'col11']),
    ...     'table2': pd.DataFrame(columns=['a', 'b', 'c'])
    ... }
    >>> result = getTablesFormatting(tables)
    >>> print(result)
    table1.columns = ['col1', 'col2', 'col3', 'col4', 'col5', 'col6', 'col7', 'col8', 'col9', 'col10', '...']
    table2.columns = ['a', 'b', 'c']
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


