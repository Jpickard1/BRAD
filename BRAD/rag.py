import numpy as np
import chromadb
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
        documentSearch = vectordb.similarity_search_with_relevance_scores(prompt)
        docs, scores = getDocumentSimilarity(documentSearch)

        # pass the database output to the llm
        chain = load_qa_chain(llm, chain_type="stuff")
        res = chain({"input_documents": docs, "question": prompt})
        print(res['output_text'])
    
        # change inputs to be json readable
        res['input_documents'] = getInputDocumentJSONs(res['input_documents'])
        chatstatus['output'], chatstatus['process'] = res['output_text'], res
    else:
        template = """Current conversation:\n{history}\n\n\nNew Input:\n{input}"""
        PROMPT = PromptTemplate(input_variables=["history", "input"], template=template)
        conversation = ConversationChain(prompt  = PROMPT,
                                         llm     = llm,
                                         verbose = True,
                                         memory  = memory,
                                        )
        prompt = getDefaultContext() + prompt
        response = conversation.predict(input=prompt)
        print(response)
        chatstatus['output'] = response
    # update and return the chatstatus
    # chatstatus['output'], chatstatus['process'] = res['output_text'], res
    chatstatus = geneOntology(chatstatus['output'], chatstatus)
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

