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
from langchain.prompts import PromptTemplate
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
from langchain_text_splitters import CharacterTextSplitter
from langchain.retrievers.multi_query import MultiQueryRetriever


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
    llm      = chatstatus['llm']              # get the llm
    prompt   = chatstatus['prompt']           # get the user prompt
    vectordb = chatstatus['databases']['RAG'] # get the vector database
    memory   = chatstatus['memory']           # get the memory of the model

    # query to database
    if vectordb is not None:

        #Ask Joshua about easy way to get path of database?
        #path = "/nfs/turbo/umms-indikar/shared/projects/RAG/papers/EXP2/"
        #best_score, text = restrictedDB(prompt, vectordb, path)
        #threshold to invoke new vector database
        #if best_score > 0.75:
        #    chatstatus['databases']['RAG'] = text

        # solo & mutliquery retrieval determined by config.json
        chatstatus, docs, scores = retrieval(chatstatus)

        # We could put reranking here\
        #docs = pagerank_rerank(docs, scores)

        # We could put contextual compression here
        docs = contextualCompression(docs, chatstatus)

        # Build chain
        chain = load_qa_chain(llm, chain_type="stuff", verbose = chatstatus['config']['debug'])

        # pass the database output to the llm
        res = chain({"input_documents": docs, "question": prompt})
        chatstatus = log.userOutput('output_text', chatstatus=chatstatus) # <- This is for the user
        sources = []
        for doc in docs:
            source = doc.metadata.get('source')
            short_source = os.path.basename(source)
            sources.append(short_source)
        sources = list(set(sources))
        chatstatus = log.userOutput("Sources:", chatstatus=chatstatus) # <- This is for the user
        chatstatus = log.userOutput(sources, chatstatus=chatstatus) # <- This is for the user
        chatstatus['process']['sources'] = sources
        # change inputs to be json readable
        res['input_documents'] = getInputDocumentJSONs(res['input_documents'])
        chatstatus['output'], ragResponse = res['output_text'], res
        chatstatus['process']['steps'].append(ragResponse)
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
        chatstatus = log.userOutput(response, chatstatus=chatstatus) # <- This is for the user
        
        chatstatus['output'] = response
    return chatstatus

def retrieval(chatstatus):
    llm      = chatstatus['llm']              # get the llm
    prompt   = chatstatus['prompt']           # get the user prompt
    vectordb = chatstatus['databases']['RAG'] # get the vector database
    memory   = chatstatus['memory']           # get the memory of the model

    if not chatstatus['config']['RAG']['multiquery']:
        documentSearch = vectordb.similarity_search_with_relevance_scores(prompt, k=chatstatus['config']['RAG']['num_articles_retrieved'])
        docs, scores = getDocumentSimilarity(documentSearch)
        chatstatus['process']['steps'].append({
            'func' : 'rag.retrieval',
            'multiquery' : False,
            'num-docs' : len(docs),
            'docs' : str(docs),
        })
    else:
        logging.basicConfig()
        logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)
        retriever = MultiQueryRetriever.from_llm(retriever=vectordb.as_retriever(),
                                                 llm=llm
                                                )
        docs = retriever.get_relevant_documents(query=prompt)  # Note: No scores are generated when using multiquery
        chatstatus['process']['steps'].append({
            'func' : 'rag.retrieval',
            'multiquery' : True,
            'num-docs' : len(docs),
            'docs' : str(docs),
        })
        scores = []
    return chatstatus, docs, scores

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
    template = summarizeDocumentTemplate()
    PROMPT = PromptTemplate(input_variables=["user_query"], template=template)
    reducedDocs = []
    for i, doc in enumerate(docs):
        pageContent = doc.page_content
        prompt = PROMPT.format(text=pageContent, user_query=chatstatus['prompt'])
        res = chatstatus['llm'].invoke(input=prompt)
        summary = res.content.strip()
        if chatstatus['config']['debug']:
            log.debugLog('============', chatstatus=chatstatus) # <- This is for debugging
            log.debugLog(pageContent, chatstatus=chatstatus) # <- This is for debugging
            log.debugLog('Summary: ' + summary, chatstatus=chatstatus) # <- This is for debugging
        doc.page_content = summary
        docs[i] = doc
    return docs

def getPreviousInput(log, key):
    """
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
                'source'   : doc.metadata['source']
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

def extract_non_english_words(text):
    """
    Extracts non-English words from a given text.

    :param text: The input text from which to extract non-English words.
    :type text: str

    :raises LookupError: If the necessary NLTK data (like word lists or lemmatizer models) is not found.

    :return: A list of words from the input text that are not recognized as English words.
    :rtype: list

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
    log.debugLog('\nWork Directory: {}'.format(local), chatstatus=chatstatus) if v else None # <- This is for debugging

    #%% Phase 1 - Load DB
    embeddings_model = HuggingFaceEmbeddings(model_name=HuggingFaceEmbeddingsModel)
    log.debugLog('\nDocuments loading from:', docsPath, chatstatus=chatstatus) iv v else None # <- This is for debugging

    text_loader_kwargs={'autodetect_encoding': True}
    loader = DirectoryLoader(docsPath,
                             glob="**/*.pdf",
                             loader_cls=PyPDFLoader, 
                             loader_kwargs=text_loader_kwargs,
                             show_progress=True,
                             use_multithreading=True)
    docs_data = loader.load()
    log.debugLog('\nDocuments loaded...', chatstatus=chatstatus) if v else None# <- This is for debugging

    for i in range(len(chunk_size)):
        for j in range(len(chunk_overlap)):
            text_splitter = RecursiveCharacterTextSplitter(chunk_size = chunk_size[i],
                                                            chunk_overlap = chunk_overlap[j],
                                                            separators=[" ", ",", "\n", ". "])
            data_splits = text_splitter.split_documents(docs_data)
            log.debugLog("Documents split into chunks...", chatstatus=chatstatus) if v else None # <- This is for debugging
            log.debugLog("Initializing Chroma Database...", chatstatus=chatstatus) if v else None # <- This is for debugging

            dbName = "DB_cosine_cSize_%d_cOver_%d" %(chunk_size[i], chunk_overlap[j])

            p2_2 = subprocess.run('mkdir  %s/*'%(dbPath+dbName), shell=True)
            _client_settings = chromadb.PersistentClient(path=(dbPath+dbName))

            vectordb = Chroma.from_documents(documents           = data_splits,
                                             embedding           = embeddings_model,
                                             client              = _client_settings,
                                             collection_name     = dbName,
                                             collection_metadata = {"hnsw:space": "cosine"})
            log.debugLog("Completed Chroma Database: ", chatstatus=chatstatus) if v else None # <- This is for debugging
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
    log.debugLog(f"output of similarity search: {scores}", chatstatus=chatstatus) # <- This is for debugging
    candidates = []    # also llm respons (maybe remove this)
    reference = []     # hidden dodument
    document_list = [] # LLM RESPONSES
    # Iterate through the indices of the original list
    for i in range(len(docs)):
        # removes one of the documents
        new_list = docs[:i] + docs[i + 1:]
        reference.append(docs[i].dict()['page_content'])
        chatstatus = log.userOutput(f"Masked Document: {docs[i].dict()['page_content']}\n", chatstatus=chatstatus) # <- This is for the user
        res = chain({"input_documents": new_list, "question": prompt})
        chatstatus = log.userOutput(f"RAG response: {res['output_text']}", chatstatus=chatstatus) # <- This is for the user
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
    chatstatus = log.userOutput(new_scores, chatstatus=chatstatus) # <- This is for the user
    #scorer = BERTScorer(lang="en", rescale_with_baseline=True)
    #P, R, F1 = scorer.score(candidates, reference)
    #log.debugLog(F1, chatstatus=chatstatus) # <- This is for debugging
    
    


#To get a single document

#ASK JOSHUA ABOUT PATHING
def restrictedDB(prompt, vectordb, path):
    #path = "/nfs/turbo/umms-indikar/shared/projects/RAG/papers/EXP2/"
    embeddings_model = HuggingFaceEmbeddings(model_name='BAAI/bge-base-en-v1.5')
    db_name = "new_DB_cosine_cSize_%d_cOver_%d" % (700, 200)
    title_list, real_id_list = get_all_sources(prompt, path)
    best_title, best_score = best_match(prompt, title_list)
    title_list, real_id_list = get_all_sources(best_title, path)
    text = vectordb.get(ids=real_id_list)['documents']
    #maybe make this a new path argument idk
    newdb = Chroma(persist_directory='/nfs/turbo/umms-indikar/shared/projects/RAG/databases/new_EXP3/', embedding_function=embeddings_model, collection_name=db_name)
    newdb.add_texts(text)
    return best_score, newdb


#Given the prompt, find the title and corresponding score that is the best match
def best_match(prompt, title_list):
    sentence_model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
    unique_title_list = list(set(title_list))
    query_embedding = sentence_model.encode(prompt)
    passage_embedding = sentence_model.encode(unique_title_list)

    save_title = ""
    #adjust the save_score to be the threshold cutoff - set to 0.75 but maybe thats too high
    #So far this is for single papers, to make it multiple - change save_title into a list
    save_score = 0.50

    for score, title in zip(util.cos_sim(query_embedding, passage_embedding)[0], unique_title_list):
        if score > save_score:
            save_score = score
            save_title = title
    log.debugLog(f"The best match is {save_title} with a score of {save_score}", chatstatus=chatstatus) # <- This is for debugging
    return save_title, save_score

#Split into two methods?
def get_all_sources(prompt, path):
    prompt = prompt.lower()
    metadata_full = vectordb.get()['metadatas']
    source_list = [item['source'] for item in metadata_full]   
    real_source_list = [((item.replace(path, '')).removesuffix('.pdf')).lower() for item in source_list]
    db = pd.DataFrame({'id' :vectordb.get()['ids'] , 'metadatas' : real_source_list})
    filtered_df = db[db['metadatas'].apply(lambda x: x in prompt)]
    return real_source_list, filtered_df['id'].to_list()


#Given the prompt, find the title and corresponding score that is the best match
def adj_matrix_builder(docs, scores):
    dimension = len(docs)+1
    adj_matrix = np.zeros([dimension, dimension])
    pos = 1
    for score in scores:
        adj_matrix[0][pos] = score
        adj_matrix[pos][0] = score
        pos += 1
    return adj_matrix

# Normalize columns of A
def normalize_adjacency_matrix(A):
    col_sums = A.sum(axis=0)
    return A / col_sums[np.newaxis, :]

#weighted pagerank algorithm
def pagerank_weighted(A, alpha=0.85, tol=1e-6, max_iter=100):
    n = A.shape[0]
    A_normalized = normalize_adjacency_matrix(A)
    v = np.ones(n) / n  # Initial PageRank vector

    for _ in range(max_iter):
        v_next = alpha * A_normalized.dot(v) + (1 - alpha) / n
        if np.linalg.norm(v_next - v, 1) < tol:
            break
        v = v_next

    return v

#reranker

def pagerank_rerank(docs, scores):
    adj_matrix = adj_matrix_builder(docs, scores)
    pagerank_scores = pagerank_weighted(A = adj_matrix)
    top_rank_scores = sorted(range(len(pagerank_scores)), key=lambda i: pagerank_scores[i], reverse=True)[1:11]
    reranked_docs = [docs[i] for i in top_rank_scores]
    return reranked_docs