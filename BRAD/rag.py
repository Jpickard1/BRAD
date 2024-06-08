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
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain_text_splitters import CharacterTextSplitter


#BERTscore
import bert_score
import logging
import transformers
transformers.tokenization_utils.logger.setLevel(logging.ERROR)
transformers.configuration_utils.logger.setLevel(logging.ERROR)
transformers.modeling_utils.logger.setLevel(logging.ERROR)
from bert_score import BERTScorer


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
    Queries documents based on the user prompt and updates the chat status with the results.

    :param chatstatus: A dictionary containing the current chat status, including the prompt, LLM, vector database, and memory.
    :type chatstatus: dict

    :raises KeyError: If required keys are not found in the chatstatus dictionary.
    :raises AttributeError: If methods on the vector database or LLM objects are called incorrectly.

    :return: The updated chat status dictionary with the query results.
    :rtype: dict
    """
    process = {}
    llm      = chatstatus['llm']              # get the llm
    prompt   = chatstatus['prompt']           # get the user prompt
    vectordb = chatstatus['databases']['RAG'] # get the vector database
    memory   = chatstatus['memory']           # get the memory of the model
    
    # query to database
    if vectordb is not None:
        documentSearch = vectordb.similarity_search_with_relevance_scores(prompt, k=chatstatus['config']['num_articles_retrieved'])
        docs, scores = getDocumentSimilarity(documentSearch)
        chain = load_qa_chain(llm, chain_type="stuff", verbose = chatstatus['config']['debug'])
        #bertscores
        if chatstatus['config']['experiment'] is True:
            # scoring_experiment(chain, docs, scores, prompt)
            crossValidationOfDocumentsExperiment(chain, docs, scores, prompt, chatstatus)
            chatstatus['process'] = {'type': 'docs cross validation experiment'}
        else:
        # pass the database output to the llm       
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
    return chatstatus

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
    print(f"output of similarity search: {scores}")
    candidates = []
    reference = []
    document_list = []
    # Iterate through the indices of the original list
    for i in range(len(docs)):
        # removes one of the documents
        new_list = docs[:i] + docs[i + 1:]
        reference.append(docs[i].dict()['page_content'])
        print(f"Masked Document: {docs[i].dict()['page_content']}\n")
        res = chain({"input_documents": new_list, "question": prompt})
        print(f"RAG response: {res['output_text']}")
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
    print(new_scores)
    #scorer = BERTScorer(lang="en", rescale_with_baseline=True)
    #P, R, F1 = scorer.score(candidates, reference)
    #print(F1)