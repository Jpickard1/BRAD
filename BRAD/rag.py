import pandas as pd
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

from sentence_transformers import SentenceTransformer, util

#BERTscore
import logging
import transformers
transformers.tokenization_utils.logger.setLevel(logging.ERROR)
transformers.configuration_utils.logger.setLevel(logging.ERROR)
transformers.modeling_utils.logger.setLevel(logging.ERROR)

from BRAD.promptTemplates import historyChatTemplate, summarizeDocumentTemplate


#Extraction
import re
#from nltk.corpus import words
#from unidecode import unidecode
#import nltk
#from nltk.stem import WordNetLemmatizer
#from nltk.corpus import wordnet

import BRAD.gene_ontology as gonto
from BRAD.gene_ontology import geneOntology
from BRAD import utils
from BRAD import log

def queryDocs(chatstatus):
    """
    Queries documents based on the user prompt and updates the chat status with the results.

    :param chatstatus: A dictionary containing the current chat status, including the prompt, LLM, vector database, and memory.
    :type chatstatus: dict

    :raises KeyError: If required keys are not found in the chatstatus dictionary.
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
    # Issues:

    llm      = chatstatus['llm']              # get the llm
    prompt   = chatstatus['prompt']           # get the user prompt
    vectordb = chatstatus['databases']['RAG'] # get the vector database
    memory   = chatstatus['memory']           # get the memory of the model

    # query to database
    if vectordb is not None:
        # solo, mutliquery, similarity, and mmr retrieval
        chatstatus, docs = retrieval(chatstatus)

        # rerank the documents according to pagerank algorithm
        if chatstatus['config']['RAG']['rerank']:
            docs = pagerank_rerank(docs, chatstatus)

        # contextual compression of the documents
        if chatstatus['config']['RAG']['contextual_compression']:
            docs = contextualCompression(docs, chatstatus)

        for i, doc in enumerate(docs):
            source = doc.metadata.get('source')
            short_source = os.path.basename(str(source))
            pageContent = doc.page_content
            addingInRefs = "Source: " + short_source + "\nContent: " + pageContent
            doc.page_content = addingInRefs
            docs[i] = doc
        
        # build chain
        chain = load_qa_chain(llm, chain_type="stuff", verbose = chatstatus['config']['debug'])

        # invoke the chain
        response = chain({"input_documents": docs, "question": prompt})

        # display output
        chatstatus = log.userOutput(response['output_text'], chatstatus=chatstatus)

        # display sources
        sources = []
        for doc in docs:
            source = doc.metadata.get('source')
            short_source = os.path.basename(str(source))
            sources.append(short_source)
        sources = list(set(sources))
        chatstatus = log.userOutput("Sources:", chatstatus=chatstatus) 
        chatstatus = log.userOutput('\n'.join(sources), chatstatus=chatstatus) 
        chatstatus['process']['sources'] = sources
        
        # format outputs for logging
        response['input_documents'] = getInputDocumentJSONs(response['input_documents'])
        chatstatus['output'], ragResponse = response['output_text'], response
        chatstatus['process']['steps'].append(log.llmCallLog(llm=llm,
                                                             prompt=str(chain),
                                                             input=prompt,
                                                             output=response,
                                                             parsedOutput=chatstatus['output'],
                                                             purpose='RAG'
                                                            )
                                             )
    else:
        template = historyChatTemplate()
        PROMPT = PromptTemplate(input_variables=["history", "input"], template=template)
        conversation = ConversationChain(prompt  = PROMPT,
                                         llm     = llm,
                                         verbose = chatstatus['config']['debug'],
                                         memory  = memory,
                                        )
        prompt = getDefaultContext() + prompt
        response = conversation.predict(input=prompt)
        chatstatus = log.userOutput(response, chatstatus=chatstatus) 
        chatstatus['output'] = response
        chatstatus['process']['steps'].append(log.llmCallLog(llm=llm,
                                                             prompt=PROMPT,
                                                             input=prompt,
                                                             output=response,
                                                             parsedOutput=chatstatus['output'],
                                                             purpose='chat without RAG'
                                                            )
                                             )
    return chatstatus


def retrieval(chatstatus):
    """
    Performs retrieval from a vectorized database as the initial stage of the RAG pipeline. 
    This function handles different types of retrieval including multiquery, similarity search, 
    and max marginal relevance search.

    Args:
        chatstatus (dict): A dictionary containing the LLM, user prompt, vector database, 
                           and configuration settings for the RAG pipeline.

    Returns:
        tuple: A tuple containing the updated chatstatus and a list of retrieved documents.

    Example
    -------
    >>> chatstatus = {
    ...     'llm': llm_instance,
    ...     'prompt': "What is the capital of France?",
    ...     'databases': {'RAG': vectordb_instance},
    ...     'memory': memory_instance,
    ...     'config': {
    ...         'RAG': {
    ...             'cut': True,
    ...             'multiquery': False,
    ...             'similarity': True,
    ...             'mmr': True,
    ...             'num_articles_retrieved': 5
    ...         }
    ...     },
    ...     'process': {'steps': []}
    ... }
    >>> chatstatus, docs = retrieval(chatstatus)
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
    #
    # Issues:
    # - The MultiQueryRetriever.from_llm doesn't give control over the number of
    #   prompts or retrieved documents. Also, I don't think it generates great
    #   prompts. We could reimplement this ourselves.
    
    llm      = chatstatus['llm']              # get the llm
    prompt   = chatstatus['prompt']           # get the user prompt
    vectordb = chatstatus['databases']['RAG'] # get the vector database
    memory   = chatstatus['memory']           # get the memory of the model

    if chatstatus['config']['RAG']['cut']:
        vectordb = cut(chatstatus, vectordb)

    if not chatstatus['config']['RAG']['multiquery']:
        # initialize empty lists
        docsSimilaritySearch, docsMMR = [], []
        if chatstatus['config']['RAG']['similarity']:
            documentSearch = vectordb.similarity_search_with_relevance_scores(prompt, k=chatstatus['config']['RAG']['num_articles_retrieved'])
            docsSimilaritySearch, scores = getDocumentSimilarity(documentSearch)

        if chatstatus['config']['RAG']['mmr']:
            docsMMR = vectordb.max_marginal_relevance_search(prompt, k=chatstatus['config']['RAG']['num_articles_retrieved'])
        
        docs = docsSimilaritySearch + docsMMR
    else:
        logging.basicConfig()
        logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)
        retriever = MultiQueryRetriever.from_llm(retriever=vectordb.as_retriever(),
                                                 llm=llm
                                                )
        docs = retriever.get_relevant_documents(query=prompt)
    chatstatus['process']['steps'].append({
        'func' : 'rag.retrieval',
        'multiquery' : chatstatus['config']['RAG']['multiquery'],
        'similarity' : chatstatus['config']['RAG']['similarity'],
        'mmr' : chatstatus['config']['RAG']['mmr'],
        'num-docs' : len(docs),
        'docs' : str(docs),
    })
    return chatstatus, docs

def contextualCompression(docs, chatstatus):
    """
    Summarizes the content of documents based on a user query, updating the 
    document search results with these summaries.

    Args:
        docs (list): A list of documents where each document has an attribute 
                     `page_content` containing the text content of the document.
        chatstatus (dict): BRAD chatstatus used to track debuging

    Returns:
        list: The modified `documentSearch` list with updated `page_content` for each 
              document, replaced by their summaries.

    Example:
        documentSearch = [Document(page_content="..."), ...]
        chatstatus = {'config': {'debug': True}}
        updatedDocs = contextualCompression(documentSearch, chatstatus)
    """
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: July 5, 2024
    template = summarizeDocumentTemplate()
    PROMPT = PromptTemplate(input_variables=["user_query"], template=template)
    reducedDocs = []
    for i, doc in enumerate(docs):
        pageContent = doc.page_content
        prompt = PROMPT.format(text=pageContent, user_query=chatstatus['prompt'])
        res = chatstatus['llm'].invoke(input=prompt)
        summary = res.content.strip()
        if chatstatus['config']['debug']:
            log.debugLog('============', chatstatus=chatstatus) 
            log.debugLog(pageContent, chatstatus=chatstatus) 
            log.debugLog('Summary: ' + summary, chatstatus=chatstatus) 
        doc.page_content = summary
        docs[i] = doc
    return docs

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
        - A list of document objects.
        - A numpy array of similarity scores.
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
    .. warning:: Marc do we need/use this function?
    
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


def getDefaultContext():
    """
    Returns the default context string for the chatbot, which provides background information and capabilities.

    :param None: This function does not take any parameters.

    :raises None: This function does not raise any specific errors.

    :return: A string containing the default context for the chatbot.
    :rtype: str
    """
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
    dbPath   += dbName
    
    local = os.getcwd()  ## Get local dir
    os.chdir(local)      ## shift the work dir to local dir
    print('\nWork Directory: {}'.format(local)) if v else None

    #%% Phase 1 - Load DB
    embeddings_model = HuggingFaceEmbeddings(model_name=HuggingFaceEmbeddingsModel)
    print("\nDocuments loading from: 'str(docsPath)") if v else None
    text_loader_kwargs={'autodetect_encoding': True}
    loader = DirectoryLoader(docsPath,
                             glob="**/*.pdf",
                             loader_cls=PyPDFLoader, 
                             loader_kwargs=text_loader_kwargs,
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

            dbName = "DB_cosine_cSize_%d_cOver_%d" %(chunk_size[i], chunk_overlap[j])

            p2_2 = subprocess.run('mkdir  %s/*'%(dbPath+dbName), shell=True)
            _client_settings = chromadb.PersistentClient(path=(dbPath+dbName))

            vectordb = Chroma.from_documents(documents           = data_splits,
                                             embedding           = embeddings_model,
                                             client              = _client_settings,
                                             collection_name     = dbName,
                                             collection_metadata = {"hnsw:space": "cosine"})
            log.debugLog("Completed Chroma Database: ", chatstatus=chatstatus) if v else None 
            del vectordb, text_splitter, data_splits

def crossValidationOfDocumentsExperiment(chain, docs, scores, prompt, chatstatus):
    scores = list(scores)
    outputs = {
        'prompt':[],
        'response':[],
        'hidden':[],
        'hiddenRef':[],
        'hiddenScore':[]
    }
    for i in range(len(docs) - 1):
        outputs['known' + str(i)] = []
        outputs['knownRef' + str(i)] = []
        outputs['knownScore' + str(i)] = []
    for i in range(len(docs)):
        # query the model
        usedDocs = docs[:i] + docs[i + 1:]
        usedScores = scores[:i] + scores[i + 1:]
        hiddenDoc = docs[i]
        hiddenScore = scores[i]
        response   = chain({"input_documents": usedDocs, "question": prompt})
        # save the info
        outputs['prompt'].append(prompt)
        outputs['response'].append(response['output_text'])
        outputs['hidden'].append(hiddenDoc.page_content)
        outputs['hiddenRef'].append(hiddenDoc.metadata)
        outputs['hiddenScore'].append(scores[i])
        for j in range(len(docs) - 1):
            outputs['known' + str(j)].append(usedDocs[j].page_content)
            outputs['knownRef' + str(j)].append(usedDocs[j].metadata)
            outputs['knownScore' + str(j)].append(scores[j])

    df = pd.DataFrame(outputs)
    # Check if the file exists
    if os.path.isfile(chatstatus['experiment-output']):
        # File exists, append to it
        df.to_csv(chatstatus['experiment-output'], mode='a', header=False, index=False)
    else:
        # File does not exist, create a new file
        df.to_csv(chatstatus['experiment-output'], mode='w', header=True, index=False)


def scoring_experiment(chain, docs, scores, prompt):
    log.debugLog(f"output of similarity search: {scores}", chatstatus=chatstatus) 
    candidates = []    # also llm respons (maybe remove this)
    reference = []     # hidden dodument
    document_list = [] # LLM RESPONSES
    # Iterate through the indices of the original list
    for i in range(len(docs)):
        # removes one of the documents
        new_list = docs[:i] + docs[i + 1:]
        reference.append(docs[i].dict()['page_content'])
        chatstatus = log.userOutput(f"Masked Document: {docs[i].dict()['page_content']}\n", chatstatus=chatstatus) 
        res = chain({"input_documents": new_list, "question": prompt})
        chatstatus = log.userOutput(f"RAG response: {res['output_text']}", chatstatus=chatstatus) 
        # Add the new list to the combinations list
        candidates.append(res['output_text'])
        document_list.append(Document(page_content = res['output_text']))
    text_splitter = CharacterTextSplitter(chunk_size=700, chunk_overlap=200)
    embedding_docs= text_splitter.split_documents(document_list)
    # create the open-source embedding function
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    # load it into Chroma
    db = Chroma.from_documents(embedding_docs, embedding_function)
    new_docs, new_scores = getDocumentSimilarity(db.similarity_search_with_relevance_scores(prompt))
    
    # print results
    chatstatus = log.userOutput(new_scores, chatstatus=chatstatus) 
    #scorer = BERTScorer(lang="en", rescale_with_baseline=True)
    #P, R, F1 = scorer.score(candidates, reference)
    #log.debugLog(F1, chatstatus=chatstatus) 
    
    


#To get a single document
def restrictedDB(chatstatus, vectordb, path):
    """
    Create a restricted database (newdb) based on a given prompt from chatstatus.

    Parameters:
    - chatstatus (dict): A dictionary containing information about the chat status,
      including 'prompt' and 'output-directory'.
    - vectordb: The vector database from which documents are retrieved.
    - path (str): The path to the directory containing source documents.

    Returns:
    - best_score (float): The cosine similarity score of the best matching document.
    - newdb (Chroma): A Chroma database object initialized with documents retrieved
      based on the best matching document's title.
    """
    prompt = chatstatus['prompt']
    embeddings_model = HuggingFaceEmbeddings(model_name='BAAI/bge-base-en-v1.5')
    db_name = "new_DB_cosine_cSize_%d_cOver_%d" % (700, 200)
    
    # Retrieve all source document titles and corresponding IDs
    title_list, real_id_list = get_all_sources(vectordb, prompt, path)
    
    # Find the best matching document based on the prompt
    best_title, best_score = best_match(prompt, title_list)
    
    # Retrieve titles and IDs based on the best matching document
    title_list, real_id_list = get_all_sources(vectordb, best_title, path)
    
    # Retrieve texts of the best matching documents from vectordb
    text = vectordb.get(ids=real_id_list)['documents']
    
    # Define the directory path for the new restricted database
    new_path = chatstatus['output-directory'] + '/restricted'
    
    # Create a new Chroma database (newdb) and add texts to it
    newdb = Chroma(persist_directory=new_path, embedding_function=embeddings_model, collection_name=db_name)
    newdb.add_texts(text)
    
    return best_score, newdb



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
    log.debugLog(f"The best match is {save_title} with a score of {save_score}", chatstatus=chatstatus) 
    return save_title, save_score

#Split into two methods?
def get_all_sources(vectordb, prompt, path):
    """
    Retrieve sources from vectordb based on a prompt and path, and filter based on the prompt.

    Parameters:
    - vectordb: The vector database object containing metadata and sources.
    - prompt (str): The prompt or query to filter sources.
    - path (str): The path to filter and clean source file paths.

    Returns:
    - real_source_list (list): A list of cleaned and filtered source names.
    - filtered_ids (list): A list of IDs corresponding to filtered sources based on the prompt.
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
def adj_matrix_builder(docs, chatstatus):
    """
    Build an adjacency matrix based on cosine similarity between a prompt and document content.

    Parameters:
    - docs (list): A list of documents or pages (objects) from which to build the adjacency matrix.
    - chatstatus (dict): A dictionary containing information about the chat status and configuration,
      including 'prompt', 'config', and 'num_articles_retrieved'.

    Returns:
    - real_cosine_sim (np.ndarray): A 2D numpy array representing the adjacency matrix,
      where each element at position (i, j) indicates the similarity score between documents i and j.
    """
    prompt_scale = 0.5  # Weighting scale for prompt similarity
    dimension = len(docs)
    adj_matrix = np.zeros([dimension, dimension])  # Initialize adjacency matrix
    
    # Initialize a sentence transformer model
    sentence_model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
    
    # Create a list to store document content (including prompt)
    doc_list = [chatstatus['prompt']] + [doc.dict()['page_content'] for doc in docs]
    
    # Encode document content into embeddings
    passage_embedding = sentence_model.encode(doc_list)
    
    # Calculate cosine similarities between embeddings
    cosine_similarities = cosine_similarity(passage_embedding[:chatstatus['config']['RAG']['num_articles_retrieved'] + 1])
    
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

    Parameters:
    - A (np.ndarray): Input adjacency matrix to be normalized.

    Returns:
    - normalized_A (np.ndarray): Normalized adjacency matrix where each element at position (i, j)
      is divided by the sum of the j-th column of the original matrix A.
    """
    col_sums = A.sum(axis=0)  # Calculate sum of each column
    normalized_A = A / col_sums[np.newaxis, :]  # Normalize each element by its column sum
    
    return normalized_A


#weighted pagerank algorithm
def pagerank_weighted(A, alpha=0.85, tol=1e-6, max_iter=100):
    """
    Calculate PageRank vector for a weighted adjacency matrix A using the power iteration method.

    Parameters:
    - A (np.ndarray): Weighted adjacency matrix representing the graph structure.
    - alpha (float, optional): Damping factor for the PageRank calculation (default is 0.85).
    - tol (float, optional): Tolerance threshold for convergence (default is 1e-6).
    - max_iter (int, optional): Maximum number of iterations for the power method (default is 100).

    Returns:
    - v (np.ndarray): PageRank vector representing the importance score of each node in the graph.
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

def pagerank_rerank(docs, chatstatus):
    """
    Rerank a list of documents based on their PageRank scores computed from an adjacency matrix.

    Parameters:
    - docs (list): List of documents or pages to be reranked.
    - chatstatus (dict): A dictionary containing information about the chat status and configuration,
      including parameters for building the adjacency matrix.

    Returns:
    - reranked_docs (list): Reranked list of documents based on their PageRank scores.
    """
    adj_matrix = adj_matrix_builder(docs, chatstatus)  # Build adjacency matrix
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

    Parameters:
    - input_string (str): The input string in which to calculate the relative frequency.

    Returns:
    - relative_frequency (float): The relative frequency of the dot character ('.') in the input string,
      expressed as the ratio of dot occurrences to the total number of characters. Returns 0.0 if the input
      string is empty.
    """
    if not input_string:
        return 0.0  # Return 0 if the string is empty
    
    dot_count = input_string.count('\n')
    total_characters = len(input_string)
    
    relative_frequency = dot_count / total_characters
    return relative_frequency

def cut(chatstatus, vectordb):
    """
    Remove documents from a vector database based on a relative frequency threshold of a specific character.

    Parameters:
    - chatstatus (dict): A dictionary containing chat status information.
    - vectordb (object): Object representing the vector database, capable of fetching and deleting documents.

    Returns:
    - vectordb (object): Updated vector database object after removing documents with a relative frequency
      of a specific character above the determined cutoff.
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

    
    