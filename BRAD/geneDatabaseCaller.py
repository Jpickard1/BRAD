

from langchain import PromptTemplate, LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

from BRAD.gene_ontology import geneOntology
from BRAD.enrichr import queryEnrichr

def geneDBRetriever(chatstatus):
    query    = chatstatus['prompt']
    llm      = chatstatus['llm']              # get the llm
    memory   = chatstatus['memory']           # get the memory of the model
    
    # Define the mapping of keywords to functions
    database_functions = {
        'ENRICHR'   : queryEnrichr,
        'GENEONTOLOGY' : geneOntology,
    }

    # Identify the database and the search terms
    template = """Current conversation:\n{{history}}
    
    GENEONTOLOGY: The Gene Ontology (GO) is an initiative to unify the representation of gene and gene product attributes across all species via the aims: 1) maintain and develop its controlled vocabulary of gene and gene product attributes; 2) annotate genes and gene products, and assimilate and disseminate annotation data; and 3) provide tools for easy access to all aspects of the data provided by the project, and to enable functional interpretation of experimental data using the GO.
    
    ENRICHR: is a tool used to lookup sets of genes and their functional association. ENRICHR has access to many gene-set libraries including Allen_Brain_Atlas_up, ENCODE_Histone_Modifications_2015, Enrichr_Libraries_Most_Popular_Genes, FANTOM6_lncRNA_KD_DEGs, GO_Biological_Process_2023, GTEx, Human_Gene_Atlas, KEGG, REACTOME, Transcription_Factor_PPIs, WikiPathways and many others databases.
    
    Available Tables:
    {tables}
    
    Query:{{input}}
    
    **INSTRUCTIONS**
    1. From the query, decide if GENEONTOLOGY or ENRICHR should be used
    2. Identify genes in the users input that should be searched. propose genes with similar names, correct the users spelling, or make small modifications to the list, but do not propose genes that are not found in the humans input.
    3. If the user wants to extract genes from a table (python dataframe), provide the necessary code to get the genes into a python list, otherwise, say None.
    4. If there is code required, identify the name of the table, otherwise say None
    5. If there is code required, identify the row or column name of the table, otherwise say None
    6. If there is code required, specify if it is a row or column
    Format your output as follows with no additional information:
    
    database: <ENRICHR or GENEONTOLOGY>
    genes: <List of genes separated by commas in query or None if code is required>
    code: <True or None>
    table: <Table Name>
    mode name: <Row or Column Name>
    mode: <Row or Column>
    """
    tablesInfo = getTablesFormatting(chatstatus['tables'])
    filled_template = template.format(tables=tablesInfo)
    PROMPT = PromptTemplate(input_variables=["history", "input"], template=filled_template)
    
    conversation = ConversationChain(prompt  = PROMPT,
                                     llm     = llm,
                                     verbose = True,
                                     memory  = memory,
                                    )
    response = conversation.predict(input=query)

    # Print gene list if debugging
    print(response) if chatstatus['config']['debug'] else None
    response = parse_llm_response(response)

    # Print gene list if debugging
    print(response) if chatstatus['config']['debug'] else None

    dbCaller = database_functions[response['database']]

    geneList = []
    if response['code'] == 'True':
        if response['mode'] == 'Column':
            geneList = list(chatstatus['tables'][response['mode name']].values)
        else:
            geneLIst = list(chatstatus['tables'].loc[(response['mode name'])].values)
    else:
        geneList = response['genes']

    # Print gene list if debugging
    print(geneList) if chatstatus['config']['debug'] else None
    
    try:
        dbCaller(chatstatus, geneList)
    except Exception as e:
        output = f'Error occurred while searching database: {e}'
        print(output)
    chatstatus['process'] = {'name':'geneDatabaseCaller'}
    return chatstatus

def parse_llm_response(response):
    """
    Parses the LLM response to extract the database name and search terms.
    
    Parameters:
    response (str): The response from the LLM.
    
    Returns:
    dict: A dictionary with the database name and a list of search terms.
    """
    # Initialize an empty dictionary to hold the parsed data
    parsed_data = {}

    # Split the response into lines
    response = response.replace("'", "")
    response = response.replace('"', "")
    print(response)
    lines = response.strip().split('\n')

    # Extract the database name
    database_line = lines[0].replace("database:", "").strip()
    parsed_data["database"] = database_line

    genes_line = lines[1].replace("genes:", "").strip()
    parsed_data["genes"] = genes_line.split(',')

    code_line = lines[2].replace("code:", "").strip()
    parsed_data["code"] = code_line.split(',')[0]

    print(parsed_data["code"])
    if parsed_data["code"].lower() != 'none':
        table_line = lines[3].replace("table:", "").strip()
        parsed_data["table"] = table_line.split(',')[0]
    
        mode_name_line = lines[4].replace("mode name:", "").strip()
        parsed_data["mode_name"] = mode_name_line.split(',')[0]
    
        mode_line = lines[5].replace("mode:", "").strip()
        parsed_data["mode"] = mode_name_line.split(',')[0]

    return parsed_data
    
def getTablesFormatting(tables):
    tablesString = ""
    for tab in tables:
        columns_list = list(tables[tab].columns)
        truncated_columns = columns_list[:10]  # Get the first 10 entries
        if len(columns_list) > 10:
            truncated_columns.append("...")  # Add '...' if the list is longer than 10
        tablesString += tab + '.columns = ' + str(truncated_columns) + '\n'
    return tablesString


def geneDatabaseRetriever(chatstatus):
    query    = chatstatus['prompt']
    llm      = chatstatus['llm']              # get the llm
    memory   = chatstatus['memory']           # get the memory of the model
    
    # Define the mapping of keywords to functions
    database_functions = {
        'ENRICHR'   : queryEnrichr,
        'GENEONTOLOGY' : geneOntology,
    }
    
    # Identify the database and the search terms
    template = """Current conversation:\n{history}

    GENEONTOLOGY: The Gene Ontology (GO) is an initiative to unify the representation of gene and gene product attributes across all species via the aims: 1) maintain and develop its controlled vocabulary of gene and gene product attributes; 2) annotate genes and gene products, and assimilate and disseminate annotation data; and 3) provide tools for easy access to all aspects of the data provided by the project, and to enable functional interpretation of experimental data using the GO.

    ENRICHR: is a tool used to lookup sets of genes and their functional association. ENRICHR has access to many gene-set libraries including Allen_Brain_Atlas_up, ENCODE_Histone_Modifications_2015, Enrichr_Libraries_Most_Popular_Genes, FANTOM6_lncRNA_KD_DEGs, GO_Biological_Process_2023, GTEx, Human_Gene_Atlas, KEGG, REACTOME, Transcription_Factor_PPIs, WikiPathways and many others databases.
    
    Query:{input}
    
    From the query, decide if GENEONTOLOGY or ENRICHR should be searched, and propose no more than 10 search terms for this query and database. Separate each term with a comma, and provide no extra information/explination for either the database or search terms. Format your output as follows with no additions:
    
    Database: <ENRICHR or GENEONTOLOGY>
    Search Terms: <improved search terms>
    Code to get genes: <provide code to extract genes from dataframes>
    """
    PROMPT = PromptTemplate(input_variables=["history", "input"], template=template)
    conversation = ConversationChain(prompt  = PROMPT,
                                     llm     = llm,
                                     verbose = chatstatus['config']['debug'],
                                     memory  = memory,
                                    )
    response = conversation.predict(input=query)
    llmResponse = parse_llm_response(response)
    llmKey, searchTerms = llmResponse['database'], llmResponse['search_terms']

    # Determine the target source
    source = next((key for key in scraping_functions if key == llmKey), 'PUBMED')
    process = {'searched': source}
    scrape_function = scraping_functions[source]
    
    # Execute the scraping function and handle errors
    try:
        output = f'searching on {source}...'
        print(output)
        print('Search Terms: ' + str(searchTerms)) if chatstatus['config']['debug'] else None
        for st in searchTerms:
            scrape_function(st)
        searchTerms = ' '.join(searchTerms)
        scrape_function(searchTerms)
    except Exception as e:
        output = f'Error occurred while searching on {source}: {e}'
        print(output)
        process = {'searched': 'ERROR'}

    chatstatus['process'] = process
    chatstatus['output']  = output
    return chatstatus