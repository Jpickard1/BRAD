import os
import sys
import numpy as np
import pandas as pd
import subprocess
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import UnstructuredPDFLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
import chromadb
from itertools import combinations

from BRAD import llms
from BRAD import brad
from BRAD import rag

def runPrompt(prompt,
              llm,                                         # preloaded LLM
              numNoRagCalls    = 12,                       # Number of trials without the RAG
              llmOnlyOutputFile='EX-10-JUNE-2024-LLM.csv'):# Outputs of the LLM without the RAG

    # Make the RAG chain
    chain = load_qa_chain(llm, chain_type="stuff", verbose = True)

    # No RAG
    outputs = {
        'prompt':[prompt],
    }
    for i in range(numNoRagCalls):
        # Call LLM
        response   = chain({"input_documents": [], "question": prompt})
        outputs['response-'+str(i)] = response['output_text']

    # Save RAG output to file
    df = pd.DataFrame(outputs)
    if os.path.isfile(llmOnlyOutputFile):
        # File exists, append to it
        df.to_csv(llmOnlyOutputFile, mode='a', header=False, index=False)
    else:
        # File does not exist, create a new file
        df.to_csv(llmOnlyOutputFile, mode='w', header=True, index=False)

def main():
    if len(sys.argv) != 2:
        print("Usage: python print_argument.py <argument>")
        sys.exit(1)
    
    temp = (float(sys.argv[1]) / 20)
    print('temp=' + str(temp))
    
    llm = llms.load_llama('/nfs/turbo/umms-indikar/shared/projects/RAG/models/llama-2-7b-chat.Q2_K.gguf', temperature=temp, max_tokens=500)
    df = pd.read_csv('EX-10-June-2024-Qs.csv')
    prompts =  df['Question']
    for i, prompt in enumerate(prompts):
        print(prompt)
        runPrompt(prompt,
                  llm,                                        # preloaded LLM
                  numNoRagCalls    = 5,                       # Number of trials without the RAG
                  llmOnlyOutputFile='EX-11-JUNE-2024-NO-RAG-TEMP-llama2-7b-temp-' + str(temp) + '.csv') # Outputs of the LLM without the RAG

if __name__ == "__main__":
    main()
