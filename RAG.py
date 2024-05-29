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

#Extraction
import re
from nltk.corpus import words
from unidecode import unidecode
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
nltk.download('words')
nltk.download('wordnet')
nltk.download('omw-1.4')

import gene_ontology as gonto

def queryDocs(chatstatus):
    process = {}
    llm      = chatstatus['llm']              # get the llm
    prompt   = chatstatus['prompt']           # get the user prompt
    vectordb = chatstatus['databases']['RAG'] # get the vector database
    
    # query to database
    documentSearch = vectordb.similarity_search_with_relevance_scores(prompt)
    docs, scores = getDocumentSimilarity(documentSearch)

    # pass the database output to the llm
    chain = load_qa_chain(llm, chain_type="stuff")
    res = chain({"input_documents": docs, "question": prompt})

    # change inputs to be json readable
    res['input_documents'] = getInputDocumentJSONs(res['input_documents'])

    # update and return the chatstatus
    chatstatus['output'], chatstatus['process'] = res['output_text'], res
    goQuery = list(set(extract_non_english_words(chatstatus['output'])))
    with open('gene_list.txt', 'r') as file:
        contents = file.read()
    gene_list = contents.split('\n')
    real_list = []
    for words in goQuery:
        words = words.upper()
        if words in gene_list:
            real_list.append(words)
    if len(real_list) > 0:
        print(real_list)
        chatstatus['output'] += '\n would you search Gene Ontology for these terms [Y/N]?'
        print('\n would you search Gene Ontology for these terms [Y/N]?')
        go = input().strip().upper()
        process['search'] = (go == 'Y')
        if go == 'Y':
        #id_list = df['id'].to_list()
            go_process = gonto.goSearch(real_list)
            chatstatus['process']['GO'] = go_process
    return chatstatus

def getPreviousInput(log, key):
    num = key[:-1]
    text = key[-1]
    if text == 'I':
        return log[int(num)][text]
    else:
        return log[int(num)][text]['output_text']
    
def getInputDocumentJSONs(input_documents):
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
    scores = []
    docs   = []
    for doc in documents:
        docs.append(doc[0])
        scores.append(doc[1])
    return docs, np.array(scores)


# Define a function to get the wordnet POS tag
def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

def extract_non_english_words(text):

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


