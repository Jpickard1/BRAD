import numpy as np
import chromadb
import subprocess
import os
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.output_parsers import CommaSeparatedListOutputParser
from semantic_router.layer import RouteLayer
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from langchain_core.prompts.prompt import PromptTemplate

#Extraction
import re
from nltk.corpus import words
from unidecode import unidecode
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

import BRAD.gene_ontology as gonto
from BRAD.gene_ontology import geneOntology

def queryDocs(chatstatus):
    """
    Query the RAG database and interact with the llama model.

    This function queries the RAG (Related Articles Generator) database using 
    the user's prompt, obtains relevant documents and their relevance scores, 
    passes them to the llama-2 language model (LLM) for question answering, 
    processes the LLm output, updates the chat status, and returns the updated 
    chat status.

    Args:
        chatstatus (dict): The current status of the chat, including the LLM 
                           instance, user prompt, RAG database instance, and 
                           other metadata.

    Returns:
        dict: The updated chat status, including the LLm output text, LLm process 
              information, and any additional metadata.

    Notes:
        - The function uses the LLM to answer questions based on the retrieved 
          documents.
        - It updates the chat status with the LLm's output text, process information, 
          and any relevant metadata.
        - The function interacts with the RAG database and may call additional 
          functions, such as `getDocumentSimilarity` and `geneOntology`, to process 
          the retrieved documents.
    """
    process = {}
    llm      = chatstatus['llm']              # get the llm
    prompt   = chatstatus['prompt']           # get the user prompt
    vectordb = chatstatus['databases']['RAG'] # get the vector database
    memory   = chatstatus['memory']           # get the memory of the model
    
    # query to database
    if vectordb is not None:
        documentSearch = vectordb.similarity_search_with_relevance_scores(prompt)
        docs, scores = getDocumentSimilarity(documentSearch)

        # pass the database output to the llm
        chain = load_qa_chain(llm,
                              chain_type="stuff",
                              verbose   = chatstatus['config']['debug'],
                             )
        res = chain({"input_documents": docs, "question": prompt})
        print(res['output_text'])
    
        # change inputs to be json readable
        res['input_documents'] = getInputDocumentJSONs(res['input_documents'])
        chatstatus['output'], chatstatus['process'] = res['output_text'], res
    else:
        template = """Current conversation: {history}\n\n\nNew Input: \n{input}"""
        PROMPT = PromptTemplate(input_variables=["history", "input"], template=template)
        conversation = ConversationChain(prompt  = PROMPT,
                                         llm     = llm,
                                         verbose = chatstatus['config']['debug'],
                                         memory  = memory,
                                        )
        prompt = getDefaultContext() + prompt
        response = conversation.predict(input=prompt)
        print(response)
        chatstatus['output'] = response
        chatstatus['process'] = {'type': 'LLM Conversation'}
    # update and return the chatstatus
    # chatstatus['output'], chatstatus['process'] = res['output_text'], res
    # chatstatus = geneOntology(chatstatus['output'], chatstatus)
    return chatstatus

def getPreviousInput(log, key):
    """
    Retrieve previous input or output text from the chat log.

    This function retrieves and returns either the previous user input or the 
    output text from the chat log, based on the provided key.

    Warnings:
        This function is in the process of being depricated.

    Args:
        log (dict): The chat log dictionary containing previous chat entries.
        key (str): The key indicating which previous input or output to retrieve. 
                   It should be in the format 'nI' or 'nO', where 'n' is an integer 
                   representing the index in the log, and 'I' or 'O' specifies 
                   whether to retrieve the input or output text.

    Returns:
        str: The previous input text if 'key' ends with 'I', or the previous output 
             text if 'key' ends with 'O'.

    Notes:
        - The 'log' parameter should be a dictionary where keys are integers 
          representing chat session indices, and values are dictionaries containing 
          'prompt' (input) and 'output' (output) keys.
    """
    num = key[:-1]
    text = key[-1]
    if text == 'I':
        return log[int(num)][text]
    else:
        return log[int(num)][text]['output_text']
    
def getInputDocumentJSONs(input_documents):
    """
    Convert a list of input documents into a JSON-compatible format.

    This function iterates through a list of input documents, extracts 
    relevant information (page content and source metadata), and returns 
    a dictionary where each document is represented as a JSON object.

    Args:
        input_documents (list): A list of input documents, each containing 
                                page content and metadata.

    Returns:
        dict: A dictionary where each key is an index and each value is a 
              JSON object representing a document, containing 'page_content' 
              and 'metadata'.

    Notes:
        - Each input document should be an object with attributes 'page_content' 
          and 'metadata', where 'metadata' is a dictionary containing at least 
          a 'source' key.
    """
    inputDocsJSON = {}
    for i, doc in enumerate(input_documents):
        inputDocsJSON[i] = {
            'page_content' : doc.page_content,
            'metadata'     : {
                'source'   : doc.metadata['source']
            }
        }
    return inputDocsJSON

def getDocumentSimilarity(documents):
    """
    Extract documents and their similarity scores from a list of tuples.

    This function extracts the documents and their similarity scores from 
    a list of tuples and returns them separately.

    Args:
        documents (list): A list of tuples, where each tuple contains a 
                          document and its similarity score.

    Returns:
        tuple: A tuple containing:
            - list: The list of documents.
            - numpy.ndarray: An array of similarity scores.

    Notes:
        - Each tuple in the 'documents' list should be in the format (document, score).
        - The function separates the documents and scores into two separate lists.
    """
    scores = []
    docs   = []
    for doc in documents:
        docs.append(doc[0])
        scores.append(doc[1])
    return docs, np.array(scores)

# Define a function to get the wordnet POS tag
def get_wordnet_pos(word):
    """
    Map POS tag to first character lemmatize() accepts.

    This function maps a Part-Of-Speech (POS) tag to the first character 
    that the WordNetLemmatizer in NLTK accepts for lemmatization.

    Args:
        word (str): A word for which the POS tag needs to be mapped.

    Returns:
        str: The corresponding WordNet POS tag.

    Notes:
        - This function uses NLTK's `pos_tag` function to get the POS tag 
          of the input word.
        - It maps POS tags to WordNet's POS tag format for lemmatization.

    Example:
        # Get WordNet POS tag for a word
        word = "running"
        pos_tag = get_wordnet_pos(word)  # returns 'v' for verb
    """
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

def extract_non_english_words(text):
    """
    Extract non-English words from a given text.

    This function extracts words from the given text, lemmatizes them, filters 
    out English words using a set of English words and a custom word list, and 
    returns non-English words.

    Args:
        text (str): The input text from which non-English words need to be extracted.

    Returns:
        list: A list of non-English words extracted from the input text.

    Notes:
        - English words are filtered out using a set of words from NLTK's words 
          corpus and a custom word list.
        - The function uses NLTK's WordNetLemmatizer and part-of-speech tagging.
        - The input text is normalized to ASCII using the unidecode library.

    Example:
        # Extract non-English words from a text
        text = "The pluripotency of stem cells in biology is fascinating."
        non_english_words = extract_non_english_words(text)
        # non_english_words would be: ['pluripotency', 'biology', 'genomics', 'reprogramming']
    """
    # Set of English words
    custom_word_list = ["pluripotency", "differentiation", "stem", "cell", "biology", "genomics", "reprogramming"]
    english_words = set(words.words()+custom_word_list)
    # Normalize text to ASCII
    normalized_text = unidecode(text)
    
    # Extract words from the text using regex
    word_list = re.findall(r'\b\w+\b', normalized_text.lower())
    
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in word_list]
    filtered_words = [word for word in lemmatized_words if not word.isnumeric()]
    
    # Filter out English words
    non_english_words = [word for word in filtered_words if word not in english_words]
    
    return non_english_words

def getDefaultContext():
    llmContext = """Context: You are BRAD (Bioinformatic Retrieval Augmented Data), a chatbot specializing in biology,
bioinformatics, genetics, and data science. You can be connected to a text database to augment your answers
based on the literature with Retrieval Augmented Generation, or you can use several additional modules including
searching the web for new articles, searching Gene Ontology or Enrichr bioinformatics databases, running snakemake
and matlab pipelines, or analyzing your own codes. Please answer the following questions to the best of your
ability.

Prompt: """
    return llmContext

def create_database(docsPath='papers/', dbName='database', dbPath='databases/', HuggingFaceEmbeddingsModel = 'BAAI/bge-base-en-v1.5', chunk_size=[700], chunck_overlap=[200], v=False):
    """
    Create a Chroma database from PDF documents.

    Args:
        docsPath (str, optional): Path where the document files are located. Default is '/nfs/turbo/umms-indikar/shared/projects/RAG/papers/'.
        dbName (str, optional): Name of the database to create. Default is None.
        dbPath (str, optional): Path where the database will be saved. Default is '/nfs/turbo/umms-indikar/shared/projects/RAG/databases/'.
        HuggingFaceEmbeddingsModel (str, optional): Model name for HuggingFace embeddings. Default is 'BAAI/bge-base-en-v1.5'.
        chunk_size (list, optional): List of chunk sizes for splitting documents. Default is [700].
        chunk_overlap (list, optional): List of chunk overlaps for splitting documents. Default is [200].
        v (bool, optional): Verbose mode. If True, print progress messages. Default is False.
    """
    # Handle arguments
    dbPath   += dbName
    
    local = os.getcwd()  ## Get local dir
    os.chdir(local)      ## shift the work dir to local dir
    
    print('\nWork Directory: {}'.format(local)) if v else None

    #%% Phase 1 - Load DB
    embeddings_model = HuggingFaceEmbeddings(model_name=HuggingFaceEmbeddingsModel)
    
    print('\nDocuments loading from:', docsPath) if v else None

    text_loader_kwargs={'autodetect_encoding': True}
    loader = DirectoryLoader(docsPath,
                             glob="**/*.pdf",
                             loader_cls=UnstructuredPDFLoader, 
                             loader_kwargs=text_loader_kwargs,
                             show_progress=True,
                             use_multithreading=True)
    docs_data = loader.load()

    print('\nDocuments loaded...') if v else None
    
    chunk_size = [700] #Chunk size 
    chunk_overlap = [200] #Chunk overlap

    for i in range(len(chunk_size)):
        for j in range(len(chunk_overlap)):
            text_splitter = RecursiveCharacterTextSplitter(chunk_size = chunk_size[i],
                                                            chunk_overlap = chunk_overlap[j],
                                                            separators=[" ", ",", "\n", ". "])
            data_splits = text_splitter.split_documents(docs_data)
            
            print('Documents split into chunks...') if v else None
            print('Initializing Chroma Database...') if v else None

            dbName = "DB_cosine_cSize_%d_cOver_%d" %(chunk_size[i], chunk_overlap[j])

            p2_2 = subprocess.run('mkdir  %s/*'%(dbPath+dbName), shell=True)
            _client_settings = chromadb.PersistentClient(path=(dbPath+dbName))

            vectordb = Chroma.from_documents(documents           = data_splits,
                                             embedding           = embeddings_model,
                                             client              = _client_settings,
                                             collection_name     = dbName,
                                             collection_metadata = {"hnsw:space": "cosine"})

            print('Completed Chroma Database: ', dbName) if v else None
            del vectordb, text_splitter, data_splits
