"""
This module implements functions to facilitate Retrieval Augmented Generation (RAG), which combines the strengths of document retrieval and language model generation to enhance the user's experience with rich, contextually relevant responses.

Key Functions
=============

1. queryDocs:
    Queries documents based on a user prompt and updates the chat status with the results. This function handles the retrieval of relevant documents from a vector database, applies contextual compression, reranks the documents if required, and invokes the language model to generate a response based on the retrieved documents. It also logs the interaction and displays the sources of the information.

2. create_database:
    Constructs a database of papers that can be used by the RAG pipeline in BRAD. This method requires a single directory of papers, books, or other pdf documents. This method should be used directly, outside of and prior to constructing an instance of the `Agent` class. Once a database is constructed, documents can be added or removed, and the database will persist on the local disk so that it only needs to be constructed once.

There are several supporting methods as well.

Available Methods
=================

This module contains the following methods:

"""


import pandas as pd
import numpy as np
import chromadb
import time
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
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.output_parsers import CommaSeparatedListOutputParser
from semantic_router.layer import RouteLayer
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
from langchain.docstore.document import Document
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from sklearn.metrics.pairwise import cosine_similarity
from langchain_text_splitters import CharacterTextSplitter
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.callbacks import get_openai_callback
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_text_splitters import RecursiveCharacterTextSplitter

from sentence_transformers import SentenceTransformer, util

#BERTscore
import logging
# import transformers
# transformers.tokenization_utils.logger.setLevel(logging.ERROR)
# transformers.configuration_utils.logger.setLevel(logging.ERROR)
# transformers.modeling_utils.logger.setLevel(logging.ERROR)

from BRAD.promptTemplates import historyChatTemplate, summarizeDocumentTemplate, getDefaultContext


#Extraction
import re



import BRAD.gene_ontology as gonto
from BRAD.gene_ontology import geneOntology
from BRAD import utils
from BRAD import log

# History:
#  2024-09-22: Changing the chains to return information regarding API usage
#              such as tokens, time, etc.

def queryDocs(state):
    """
    Queries documents based on the user prompt and updates the chat status with the results.

    :param state: A dictionary containing the current chat status, including the prompt, LLM, vector database, and memory.
    :type state: dict

    :raises KeyError: If required keys are not found in the state dictionary.
    :raises AttributeError: If methods on the vector database or LLM objects are called incorrectly.

    :return: The updated chat status dictionary with the query results.
    :rtype: dict
    """
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: May 20, 2024

    # Developer Comments:
    # -------------------
    # This function performs Retrieval Augmented Generation. A separate method
    # does the retireival, and this is primarily focused on the generation or
    # llm calling
    #
    # History:
    #  2024-07-21: Added a new feature to change the doc.page_content to include
    #              the source
    #  2024-10-16: JP changes made to make the logs of this suitable to the GUI

    # Issues:

    llm      = state['llm']              # get the llm
    prompt   = state['prompt']           # get the user prompt
    vectordb = state['databases']['RAG'] # get the vector database
    memory   = state['memory']           # get the memory of the model

    # query to database
    if vectordb is not None:
        # solo, mutliquery, similarity, and mmr retrieval
        state, docs = retrieval(state)

        # rerank the documents according to pagerank algorithm
        if state['config']['RAG']['rerank']:
            docs = pagerank_rerank(docs, state)

        # document enrichment
        if state['config']['RAG']['documentEnrichment']:
            docs = documentEnrichment(docs, state)
        
        # contextual compression of the documents
        if state['config']['RAG']['contextual_compression']:
            docs = contextualCompression(docs, state)

        for i, doc in enumerate(docs):
            source = doc.metadata.get('source')
            short_source = os.path.basename(str(source))
            pageContent = doc.page_content
            addingInRefs = "Source: " + short_source + "\nContent: " + pageContent
            doc.page_content = addingInRefs
            docs[i] = doc

        # build chain
        chain = load_qa_chain(llm, chain_type="stuff", verbose = state['config']['debug'])

        # invoke the chain
        start_time = time.time()

        with get_openai_callback() as cb:
            response = chain({"input_documents": docs, "question": prompt})
        response['metadata'] = {
            'content' : response,
            'time' : time.time() - start_time,
            'call back': {
                "Total Tokens": cb.total_tokens,
                "Prompt Tokens": cb.prompt_tokens,
                "Completion Tokens": cb.completion_tokens,
                "Total Cost (USD)": cb.total_cost
            }
        }

        # display output
        state = log.userOutput(response['output_text'], state=state)

        # display sources
        sources = []
        for doc in docs:
            source = doc.metadata.get('source')
            short_source = os.path.basename(str(source))
            sources.append(short_source)
        sources = list(set(sources))
        state = log.userOutput("Sources:", state=state) 
        state = log.userOutput('\n'.join(sources), state=state) 
        state['process']['sources'] = sources
        
        # format outputs for logging
        response['input_documents'] = getInputDocumentJSONs(response['input_documents'])
        state['output'], ragResponse = response['output_text'], response
        state['process']['steps'].append(
            log.llmCallLog(
                llm=llm,
                prompt=str(chain),
                input=prompt,
                output=response,
                parsedOutput=state['output'],
                apiInfo=response['metadata']['call back'],
                purpose='RAG'
            )
        )
    else:
        template = historyChatTemplate()
        PROMPT = PromptTemplate(input_variables=["history", "input"], template=template)
        conversation = ConversationChain(prompt  = PROMPT,
                                         llm     = llm,
                                         verbose = state['config']['debug'],
                                         memory  = memory,
                                        )
        prompt = getDefaultContext() + prompt

        # Invoke LLM tracking its usage
        start_time = time.time()
        with get_openai_callback() as cb:
            response = conversation.predict(input=prompt)
        responseDetails = {
            'content' : response,
            'time' : time.time() - start_time,
            'call back': {
                "Total Tokens": cb.total_tokens,
                "Prompt Tokens": cb.prompt_tokens,
                "Completion Tokens": cb.completion_tokens,
                "Total Cost (USD)": cb.total_cost
            }
        }
        # Log the LLM response
        state['process']['steps'].append(
            log.llmCallLog(
                llm=llm,
                prompt=PROMPT,
                input=prompt,
                output=responseDetails,
                parsedOutput=response,
                apiInfo=responseDetails['call back'],
                purpose='chat without RAG'
            )
        )

        # Display output to the user
        state = log.userOutput(response, state=state)
        # state['output'] = response
    return state


def retrieval(state):
    """
    Performs retrieval from a vectorized database as the initial stage of the RAG pipeline. 
    This function handles different types of retrieval including multiquery, similarity search, 
    and max marginal relevance search.

    Args:
        state (dict): A dictionary containing the LLM, user prompt, vector database, 
                           and configuration settings for the RAG pipeline.

    Returns:
        tuple: A tuple containing the updated state and a list of retrieved documents.

    """
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: June 16, 2024
    
    # Developer Comments:
    # -------------------
    # This function performs retrieval from a vectorized database as the initial
    # stage of the RAG pipeline. It performs several different types of retrieval
    # including multiquery, similarity search, and max marginal relevance search.
    #
    # History:
    # - 2024-06-16: JP initialized the function with similarity search and multiquery
    # - 2024-06-26: MC added cut() to remove poorly chunked pieces of text
    # - 2024-06-29: MC added max_marginal_relevance_search for retrieval
    # - 2024-10-16: JP saved the doc sources and text as strings that can be sent to
    #               the GUI for display
    #
    # Issues:
    # - The MultiQueryRetriever.from_llm doesn't give control over the number of
    #   prompts or retrieved documents. Also, I don't think it generates great
    #   prompts. We could reimplement this ourselves.
    
    llm      = state['llm']              # get the llm
    prompt   = state['prompt']           # get the user prompt
    vectordb = state['databases']['RAG'] # get the vector database
    memory   = state['memory']           # get the memory of the model

    start_time = time.time()
    
    if state['config']['RAG']['cut']:    # Can we remove this code?
        vectordb = cut(state, vectordb)

    if not state['config']['RAG']['multiquery']:
        # initialize empty lists
        docsSimilaritySearch, docsMMR = [], []
        if state['config']['RAG']['similarity']:
            documentSearch = vectordb.similarity_search_with_relevance_scores(prompt, k=state['config']['RAG']['num_articles_retrieved'])
            docsSimilaritySearch, scores = getDocumentSimilarity(documentSearch)

        if state['config']['RAG']['mmr']:
            docsMMR = vectordb.max_marginal_relevance_search(prompt, k=state['config']['RAG']['num_articles_retrieved'])
        
        docs = docsSimilaritySearch + docsMMR
    else:
        logging.basicConfig()
        logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)
        retriever = MultiQueryRetriever.from_llm(retriever=vectordb.as_retriever(),
                                                 llm=llm
                                                )
        docs = retriever.get_relevant_documents(query=prompt)

    docsText = []
    for doc in docs:
        docsText.append({
            'source': doc.metadata.get('source'),
            'text' : doc.page_content
        })

    state['process']['steps'].append({
        'func' : 'rag.retrieval',
        'multiquery' : state['config']['RAG']['multiquery'],
        'similarity' : state['config']['RAG']['similarity'],
        'mmr' : state['config']['RAG']['mmr'],
        'num-docs' : len(docs),
        'docs' : str(docs),
        'docs-to-gui': docsText,
        'time' : time.time() - start_time
    })
    return state, docs

def contextualCompression(docs, state):
    """
    Summarizes the content of documents based on a user query, updating the 
    document search results with these summaries.

    Args:
        docs (list): A list of documents where each document has an attribute 
                     `page_content` containing the text content of the document.
        state (dict): BRAD state used to track debuging

    Returns:
        list: The modified `documentSearch` list with updated `page_content` for each 
              document, replaced by their summaries.

    """
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: July 5, 2024
    template = summarizeDocumentTemplate()
    PROMPT = PromptTemplate(input_variables=["user_query"], template=template)
    reducedDocs = []
    for i, doc in enumerate(docs):
        pageContent = doc.page_content
        prompt = PROMPT.format(text=pageContent, user_query=state['prompt'])

        # Use LLM
        start_time = time.time()
        with get_openai_callback() as cb:
            res = state['llm'].invoke(input=prompt)
        res.response_metadata['time'] = time.time() - start_time
        res.response_metadata['call back'] = {
            "Total Tokens": cb.total_tokens,
            "Prompt Tokens": cb.prompt_tokens,
            "Completion Tokens": cb.completion_tokens,
            "Total Cost (USD)": cb.total_cost
        }

        summary = res.content.strip()

        # Log LLM
        state['process']['steps'].append(
            log.llmCallLog(
                llm             = state['llm'],       # what LLM?
                prompt          = PROMPT,                  # what prompt template?
                input           = prompt,                  # what specific input to the llm or template?
                output          = res,                     # what is the full llm output?
                parsedOutput    = summary,                 # what is the useful output?
                purpose         = 'contextual compression' # why did you use an llm
            )
        )
        
        # Display debug information
        if state['config']['debug']:
            log.debugLog('============', state=state) 
            log.debugLog(pageContent, state=state) 
            log.debugLog('Summary: ' + summary, state=state) 
        doc.page_content = summary
        docs[i] = doc
    return docs

def documentEnrichment(docs, state):
    """
    Enhances the input list of documents by retrieving additional text chunks 
    from the same source and page, ensuring no duplicate entries are added.

    This function searches through a vector database (vectordb) to find more text 
    that is related to the documents in the input `docs` list. It retrieves all 
    document chunks from the same file and page that were found during the first retrieval.
    Duplicate text chunks are avoided by maintaining a set of already seen texts.

    Args:
        docs (list): A list of `langchain_core.documents.base.Document` objects. 
                     Each `Document` contains metadata, including 'source' and 'page',
                     and `page_content`, which holds the text content.
        state (dict): The `Agent.state` containing the RAG database

    Returns:
        list: A list of enriched `Document` objects. These are constructed from 
              the additional text chunks found on the same page and source 
              as the originally retrieved documents. The list will only contain
              unique documents to avoid duplication.

    Notes:
        - This function assumes that the `vectordb` object has a method `get()` 
          that returns a dictionary with keys: 'metadatas' and 'documents'. 
          The 'metadatas' key contains metadata for each document chunk, 
          including its source and page. The 'documents' key contains the text content.
        - The function prevents duplication of text chunks by using a `set` to track 
          previously added texts.
    """
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: October 9, 2024
    
    # Extract the vectordb from the state
    vectordb = state['databases']['RAG']

    # A list to hold all documents or text chunks found in the same file and page
    page_enriched_documents = []
    
    # A set of all text chunks that have been added to page_enriched_documents
    # This prevents adding duplicate entries
    unique_texts = set()
    
    # Retrieve index dictionary from vectordb containing document metadata and text
    # Example structure of indexDictionary:
    # dict_keys(['ids', 'embeddings', 'metadatas', 'documents', 'uris', 'data'])
    # The keys: 'metadatas' and 'documents' must be present
    index_dictionary = vectordb.get()
    
    # Iterate over retrieved documents
    for retrieved_doc in docs:
        # Extract metadata for the current document
        source = retrieved_doc.metadata.get('source')
        page = retrieved_doc.metadata.get('page')
        
        # Initialize a list to store indices of related documents from the same source and page
        related_document_idxs = [
            idx for idx, metadata in enumerate(index_dictionary['metadatas'])
            if metadata.get('source') == source and metadata.get('page') == page
        ]
    
        # Extract text chunks from documents found on the same page
        doc_text_chunks = [index_dictionary['documents'][i] for i in related_document_idxs]
    
        # Save each text chunk as a new document if it's not already added
        for text_chunk in doc_text_chunks:
            if text_chunk in unique_texts:
                continue
            
            # Add the text chunk to the set of unique texts
            unique_texts.add(text_chunk)
    
            # Create a new document with the text chunk and its metadata
            new_doc = Document(
                page_content=text_chunk,
                metadata={
                    'source': source,
                    'page': page
                }
            )
    
            # Append the newly created document to the list
            page_enriched_documents.append(new_doc)
    
    return page_enriched_documents


def getPreviousInput(log, key):
    """
    .. warning:: This method will be removed soon.
    
    Retrieves previous input or output from the log based on the specified key.

    :param log: A list containing the chat log or history of interactions.
    :type log: list
    :param key: A string key indicating which previous input or output to retrieve.
    :type key: str

    :raises IndexError: If the specified index is out of bounds in the log.
    :raises KeyError: If the specified key is not found in the log entry.

    :return: The previous input or output corresponding to the specified key.
    :rtype: str

    """
    num = key[:-1]
    text = key[-1]
    if text == 'I':
        return log[int(num)][text]
    else:
        return log[int(num)][text]['output_text']
    
def getInputDocumentJSONs(input_documents):
    """
    Converts a list of input documents into a JSON serializable dictionary format.

    :param input_documents: A list of document objects containing page content and metadata.
    :type input_documents: list

    :return: A dictionary where keys are indices and values are JSON serializable representations of the input documents.
    :rtype: dict

    """
    inputDocsJSON = {}
    for i, doc in enumerate(input_documents):
        inputDocsJSON[i] = {
            'page_content' : doc.page_content,
            'metadata'     : {
                'source'   : str(doc.metadata)
            }
        }
    return inputDocsJSON

def getDocumentSimilarity(documents):
    """
    Extracts documents and their similarity scores from a list of document-score pairs.

    :param documents: A list of tuples where each tuple contains a document object and its similarity score.
    :type documents: list

    :return: A tuple containing two elements: 
        - A list of `langchain_core.documents.base.Document` document objects.
        - A `numpy` array of similarity scores.
    :rtype: tuple
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
    .. warning:: This function may be removed in the near future.
    
    Gets the WordNet part of speech (POS) tag for a given word.

    :param word: The word for which to retrieve the POS tag.
    :type word: str

    :return: The WordNet POS tag corresponding to the given word. Defaults to noun if no specific tag is found.
    :rtype: str

    """
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)


def create_database(docsPath='papers/', dbName='database', dbPath='databases/', HuggingFaceEmbeddingsModel = 'BAAI/bge-base-en-v1.5', chunk_size=[700], chunk_overlap=[200], v=False):
    """
    .. note: This funciton is not called by the chatbot. Instead, it is required that the user build the database prior to using the chat.
    
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
    # dbPath   += dbName
    
    local = os.getcwd()  ## Get local dir
    os.chdir(local)      ## shift the work dir to local dir
    print('\nWork Directory: {}'.format(local)) if v else None

    # Phase 1 - Load DB
    embeddings_model = HuggingFaceEmbeddings(model_name=HuggingFaceEmbeddingsModel)
    print("\nDocuments loading from: 'str(docsPath)") if v else None
    # text_loader_kwargs={'autodetect_encoding': True}
    text_loader_kwargs={}
    loader = DirectoryLoader(docsPath,
                             glob="**/*.pdf",
                             loader_cls=PyPDFLoader, 
                             # loader_kwargs=text_loader_kwargs,
                             show_progress=True,
                             use_multithreading=True)
    docs_data = loader.load()
    print('\nDocuments loaded...') if v else None

    for i in range(len(chunk_size)):
        for j in range(len(chunk_overlap)):
            text_splitter = RecursiveCharacterTextSplitter(chunk_size = chunk_size[i],
                                                            chunk_overlap = chunk_overlap[j],
                                                            separators=[" ", ",", "\n", ". "])
            data_splits = text_splitter.split_documents(docs_data)
            print("Documents split into chunks...") if v else None
            print("Initializing Chroma Database...") if v else None

            # dbName = "DB_cosine_cSize_%d_cOver_%d" %(chunk_size[i], chunk_overlap[j])

            print("dbName reset")
            # p2_2 = subprocess.run('mkdir  %s/*'%os.path.join(dbPath,dbName), shell=True)
            p2_2 = os.makedirs(os.path.join(dbPath, dbName), exist_ok=True)
            # print(os.path.join(dbPath, dbName))
            # print("subprocess run")
            # print("_client_settings set")
            # print("Starting database construction")
            _client_settings = chromadb.PersistentClient(path=os.path.join(dbPath,dbName))
            vectordb = Chroma.from_documents(documents           = data_splits,
                                             embedding           = embeddings_model,
                                             client              = _client_settings,
                                             collection_name     = dbName,
                                             collection_metadata = {"hnsw:space": "cosine"})
            # print(f"{vectordb=}")
            # log.debugLog("Completed Chroma Database: ", display=v)
            del text_splitter, data_splits
    return vectordb



def best_match(prompt, title_list):
    """
    Find the best matching title from the list based on cosine similarity with a given prompt.

    Parameters:
    - prompt (str): The prompt or query to find the best match for.
    - title_list (list): A list of titles (strings) to compare against the prompt.

    Returns:
    - best_title (str): The title from title_list that best matches the prompt.
    - best_score (float): The cosine similarity score of the best matching title with the prompt.
    """
    # Initialize a sentence transformer model
    sentence_model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
    
    # Remove duplicate titles from the list
    unique_title_list = list(set(title_list))
    
    # Encode the prompt and titles into embeddings
    query_embedding = sentence_model.encode(prompt)
    passage_embedding = sentence_model.encode(unique_title_list)

    # Initialize variables to store the best match title and score
    best_title = ""
    best_score = 0.0
    
    # Set a threshold score for saving the best match
    save_score = 0.50  # Adjust as needed
    
    # Compare cosine similarity between query embedding and each title embedding
    for score, title in zip(util.cos_sim(query_embedding, passage_embedding)[0], unique_title_list):
        if score > save_score:
            save_score = score
            save_title = title
    log.debugLog(f"The best match is {save_title} with a score of {save_score}", state=state) 
    return save_title, save_score

#Split into two methods?
def get_all_sources(vectordb, prompt, path):
    """
    Retrieve sources from the vector database based on a prompt and path, and filter the results according to the prompt.
    
    :param vectordb: The vector database object containing metadata and sources for retrieval.
    :type vectordb: object
    
    :param prompt: The prompt or query used to filter the sources retrieved from the vector database.
    :type prompt: str
    
    :param path: The path used to filter and clean source file paths, ensuring consistency and relevance in the results.
    :type path: str
    
    :return: A tuple containing:
        - real_source_list: A list of cleaned and filtered source names that match the given prompt.
        - filtered_ids: A list of IDs corresponding to the filtered sources based on the prompt.
    :rtype: tuple
    """
    prompt = prompt.lower()
    
    # Retrieve metadata from vectordb
    metadata_full = vectordb.get()['metadatas']
    
    # Extract source file paths
    source_list = [item['source'] for item in metadata_full]
    
    # Clean and filter source paths based on the provided path
    real_source_list = [((item.replace(path, '')).removesuffix('.pdf')).lower() for item in source_list]
    
    # Create a dataframe with IDs and cleaned source names
    db = pd.DataFrame({'id': vectordb.get()['ids'], 'metadatas': real_source_list})
    
    # Filter dataframe based on the prompt
    filtered_df = db[db['metadatas'].apply(lambda x: x in prompt)]
    
    # Extract IDs of filtered sources
    filtered_ids = filtered_df['id'].to_list()
    
    return real_source_list, filtered_ids



#Given the prompt, find the title and corresponding score that is the best match
def adj_matrix_builder(docs, state):
    """
    Build an adjacency matrix based on cosine similarity between a prompt and document content.
    
    :param docs: A list of documents or pages (objects) from which to build the adjacency matrix.
    :type docs: list
    
    :param state: A dictionary containing information about the chat status and configuration, 
                       including 'prompt', 'config', and 'num_articles_retrieved'.
    :type state: dict
    
    :return: A 2D numpy array representing the adjacency matrix, where each element at position (i, j) 
             indicates the similarity score between documents i and j.
    :rtype: np.ndarray
    """
    prompt_scale = 0.5  # Weighting scale for prompt similarity
    dimension = len(docs)
    adj_matrix = np.zeros([dimension, dimension])  # Initialize adjacency matrix
    
    # Initialize a sentence transformer model
    sentence_model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
    
    # Create a list to store document content (including prompt)
    doc_list = [state['prompt']] + [doc.dict()['page_content'] for doc in docs]
    
    # Encode document content into embeddings
    passage_embedding = sentence_model.encode(doc_list)
    
    # Calculate cosine similarities between embeddings
    cosine_similarities = cosine_similarity(passage_embedding[:state['config']['RAG']['num_articles_retrieved'] + 1])
    
    # Extract similarity scores between prompt and other documents
    prompt_sim = cosine_similarities[0, 1:]
    
    # Adjust cosine similarities based on prompt scale and similarity scores
    real_cosine_sim = np.zeros((len(prompt_sim), len(prompt_sim)))
    for i in range(len(prompt_sim)):
        for j in range(len(prompt_sim)):
            real_cosine_sim[i, j] = prompt_scale * cosine_similarities[i, j] + 0.5 * (1 - prompt_scale) * (prompt_sim[i] + prompt_sim[j])
    
    return real_cosine_sim


 
def normalize_adjacency_matrix(A):
    """
    Normalize an adjacency matrix by dividing each element by the sum of its column.
    
    :param A: Input adjacency matrix to be normalized, where each element (i, j) represents the weight of 
              the edge from node i to node j.
    :type A: np.ndarray
    
    :return: Normalized adjacency matrix where each element at position (i, j) is divided by the sum of 
             the j-th column of the original matrix A. This normalization ensures that the columns of the 
             resulting matrix sum to 1, facilitating interpretation as probabilities or relative weights.
    :rtype: np.ndarray
    """
    col_sums = A.sum(axis=0)  # Calculate sum of each column
    normalized_A = A / col_sums[np.newaxis, :]  # Normalize each element by its column sum
    
    return normalized_A


#weighted pagerank algorithm
def pagerank_weighted(A, alpha=0.85, tol=1e-6, max_iter=100):
    """
    Calculate the PageRank vector for a weighted adjacency matrix A using the power iteration method.
    
    :param A: Weighted adjacency matrix representing the graph structure, where each element (i, j) indicates 
              the weight of the edge from node i to node j.
    :type A: np.ndarray
    
    :param alpha: Damping factor for the PageRank calculation, which controls the probability of following 
                  an outgoing link versus randomly jumping to any node (default is 0.85).
    :type alpha: float, optional
    
    :param tol: Tolerance threshold for convergence, determining the acceptable difference between successive 
                PageRank vectors (default is 1e-6).
    :type tol: float, optional
    
    :param max_iter: Maximum number of iterations for the power method, limiting the computation to ensure 
                      it does not run indefinitely (default is 100).
    :type max_iter: int, optional
    
    :return: PageRank vector representing the importance score of each node in the graph, where higher scores 
             indicate greater importance or influence within the network.
    :rtype: np.ndarray
    """
    n = A.shape[0]  # Number of nodes in the graph
    A_normalized = normalize_adjacency_matrix(A)  # Normalize the adjacency matrix
    v = np.ones(n) / n  # Initial PageRank vector

    for _ in range(max_iter):
        v_next = alpha * A_normalized.dot(v) + (1 - alpha) / n  # Power iteration step
        if np.linalg.norm(v_next - v, 1) < tol:  # Check convergence using L1 norm
            break
        v = v_next  # Update PageRank vector
    
    return v


#reranker

def pagerank_rerank(docs, state):
    """
    Rerank a list of documents based on their PageRank scores computed from an adjacency matrix.
    
    :param docs: List of documents or pages to be reranked, where each document is represented as an object 
                 containing relevant information for scoring.
    :type docs: list
    
    :param state: A dictionary containing information about the chat status and configuration, including 
                       parameters for building the adjacency matrix, such as the 'num_articles_retrieved' 
                       and 'config' settings.
    :type state: dict
    
    :return: A reranked list of documents based on their PageRank scores, ordered from highest to lowest 
             score, reflecting the importance of each document in the context of the provided adjacency matrix.
    :rtype: list
    """
    adj_matrix = adj_matrix_builder(docs, state)  # Build adjacency matrix
    pagerank_scores = pagerank_weighted(A=adj_matrix)  # Compute PageRank scores
    top_rank_scores = sorted(range(len(pagerank_scores)), key=lambda i: pagerank_scores[i], reverse=True)
    reranked_docs = [docs[i] for i in top_rank_scores]  # Rerank documents based on PageRank scores
    
    return reranked_docs


#removes repeat chunks in vectordb
def remove_repeats(vectordb):
    """
    Removes repeated chunks in the provided vector database.

    This function identifies duplicate documents in the vector database and removes
    the repeated entries, keeping only the last occurrence of each duplicated document.

    :param vectordb: The vector database from which repeated documents should be removed.
    :type vectordb: An instance of a vector database class with 'get' and 'delete' methods.

    :raises KeyError: If the vector database does not contain 'ids' or 'documents' keys.

    :return: The updated vector database with duplicate documents removed.
    :rtype: An instance of the vector database class.
    """
    # Auth: Marc Choi
    #       machoi@umich.edu
    # Date: June 18, 2024

    # Fetch document IDs and contents from vector database
    df = pd.DataFrame({'id': vectordb.get()['ids'], 'documents': vectordb.get()['documents']})
    
    # Find IDs of documents that have duplicate content
    repeated_ids = df[df.duplicated(subset='documents', keep='last')]['id'].tolist()
    
    # Delete duplicate documents if any found
    if len(repeated_ids) > 0:
        vectordb.delete(repeated_ids)
    
    return vectordb



#experimental - see the relative frequency of periods showing up in a given doc
def relative_frequency_of_char(input_string):
    """
    Calculate the relative frequency of the dot character ('.') in a given input string.
    
    :param input_string: The input string in which to calculate the relative frequency of the dot character.
    :type input_string: str
    
    :return: The relative frequency of the dot character ('.') in the input string, expressed as the ratio 
             of dot occurrences to the total number of characters. If the input string is empty, it returns 
             0.0 to indicate no occurrences.
    :rtype: float
    """
    if not input_string:
        return 0.0  # Return 0 if the string is empty
    
    dot_count = input_string.count('\n')
    total_characters = len(input_string)
    
    relative_frequency = dot_count / total_characters
    return relative_frequency

def cut(state, vectordb):
    """
    Remove documents from a vector database based on a relative frequency threshold of a specific character.
    
    :param state: A dictionary containing chat status information, including the relative frequency
                       threshold for the character and other contextual details related to the ongoing 
                       conversation.
    :type state: dict
    
    :param vectordb: An object representing the vector database, which provides methods for fetching 
                      and deleting documents based on certain criteria, including relative frequency.
    
    :return: The updated vector database object after removing documents that exceed the specified 
             relative frequency threshold for the character.
    """    
    # Fetch document IDs and contents from vector database
    df = pd.DataFrame({'id': vectordb.get()['ids'], 'documents': vectordb.get()['documents']})
    
    # Calculate relative frequencies of the specific character ('.') for each document
    relfreq = [relative_frequency_of_char(docs) for docs in df['documents']]
    
    # Add relative frequencies as a new column in the dataframe
    df['relfreq'] = relfreq
    
    # Determine cutoff value for relative frequency (e.g., 80th percentile)
    cutoff = np.percentile(relfreq, 80)
    print(f"Cutoff relative frequency: {cutoff}")
    
    # Filter documents based on the cutoff
    filtered_df = df[df['relfreq'] > cutoff]
    
    # Get IDs of filtered documents
    filtered_ids = filtered_df['id'].tolist()
    
    # Delete filtered documents from the vector database if any are found
    if len(filtered_ids) > 0:
        vectordb.delete(filtered_ids)
    
    return vectordb

    
    
