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

def queryDocs(prompt, chatstatus, llm):
    # llm = chatstatus['llm']
    vectordb = chatstatus['databases']['RAG']
    # query to database
    v = vectordb.similarity_search(prompt)
    chain = load_qa_chain(llm, chain_type="stuff")
    res = chain({"input_documents": v, "question": prompt})
    # change inputs to be json readable
    res['input_documents'] = getInputDocumentJSONs(res['input_documents'])
    return res['output_text'], res

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
        