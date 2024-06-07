import os
from semantic_router import Route
from semantic_router.layer import RouteLayer
from semantic_router.encoders import HuggingFaceEncoder

def read_prompts(file_path):
    """
    Reads a text file where each line represents a sentence and returns a list of sentences.

    :param file_path: The path to the text file to be read.
    :type file_path: str

    :raises FileNotFoundError: If the specified file does not exist.

    :return: A list of sentences read from the text file.
    :rtype: list

    """
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
    Adds a new sentence to the specified text file.

    :param file_path: The path to the text file where the sentence is to be added.
    :type file_path: str
    :param sentence: The sentence to be added to the text file.
    :type sentence: str

    :raises FileNotFoundError: If the specified file does not exist or cannot be created.

    :return: None
    :rtype: None

    """
    with open(file_path, 'a') as file:
        file.write(sentence.strip() + '\n')

def getRouterPath(file):
    current_script_path = os.path.abspath(__file__)
    current_script_dir = os.path.dirname(current_script_path)
    file_path = os.path.join(current_script_dir, 'routers', file) #'enrichr.txt')
    return file_path
    
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

def getRouter():
    """
    Returns a router layer configured with predefined routes for various tasks.

    :param None: This function does not take any parameters.

    :raises None: This function does not raise any specific errors.

    :return: A router layer configured with predefined routes for tasks such as querying Enrichr, web scraping, and generating tables.
    :rtype: RouteLayer

    """
    encoder = HuggingFaceEncoder()
    routes = [routeGget, routeScrape, routeTable, routeRAG]
    router = RouteLayer(encoder=encoder, routes=routes)    
    return router

def buildRoutes(prompt):
    """
    Builds routes based on the provided prompt and updates the corresponding text files with the new prompts.

    :param prompt: The prompt containing the information to be added to the router.
    :type prompt: str

    :raises KeyError: If the specified route is not found in the paths dictionary.

    :return: None
    :rtype: None

    """
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
        'GGET'   : getRouterPath('enrichr.txt'),
        'SCRAPE' : getRouterPath('scrape.txt'),
        'RAG'    : getRouterPath('rag.txt'),
        'TABLE'  : getRouterPath('table.txt'),
        'DATA'   : getRouterPath('data.txt'),
        'SNS'    : getRouterPath('sns.txt')
    }
    filepath = paths[route]
    add_sentence(filepath, rebuiltPrompt)
    

def getTableRouter():
    """
    Returns a router layer configured specifically for handling table-related tasks.

    :param None: This function does not take any parameters.

    :raises None: This function does not raise any specific errors.

    :return: A router layer configured with a route for handling data-related tasks.
    :rtype: RouteLayer
    """
    encoder = HuggingFaceEncoder()
    routes = [routeData]
    router = RouteLayer(encoder=encoder, routes=routes)    
    return router