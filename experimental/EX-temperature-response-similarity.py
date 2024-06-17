import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from langchain.embeddings import HuggingFaceEmbeddings # Embedding text to vectors

def cosineSim(doc1, doc2):
    dotprod = np.dot(doc1, doc2)
    doc1norm = np.linalg.norm(doc1)
    doc2norm = np.linalg.norm(doc2)
    return (dotprod / (doc1norm*doc2norm))

def cosine_sim(docs, embedderModel=None):
    docVecs = []
    for doc in docs:
        docVecs.append(np.array(embedderModel.embed_query(doc)))
    cosSim = []
    for i in range(len(docVecs)):
        for j in range(i+1, len(docVecs)):
            cosSim.append(cosineSim(docVecs[i], docVecs[j]))
    return cosSim

def main():
    numResponses = 5
    # assemble files
    dfExp = pd.DataFrame()
    for i in range(21):
        temp = i / 20
        NORAGFile = 'EX-11-JUNE-2024-NO-RAG-TEMP-llama2-7b-temp-' + str(temp) + '.csv'
        RAGFile   = 'EX-12-JUNE-2024-RAG-TEMP-llama2-7b-temp-' + str(temp) + '.csv'
        dfRagTemp = pd.read_csv(RAGFile)
        dfNoRagTemp = pd.read_csv(NORAGFile)
        dfRagTemp['temp'] = temp
        dfNoRagTemp['temp'] = temp
        dfRagTemp['rag'] = True
        dfNoRagTemp['rag'] = False
        dfExp = pd.concat([dfExp, dfRagTemp, dfNoRagTemp], ignore_index=True)
    embedder = HuggingFaceEmbeddings(model_name='BAAI/bge-base-en-v1.5')

    temps = [] # Measured temp
    rags   = [] # if rag is used in a response
    responseSimilarity = [] # similaries between each input output pair
    prompts = []
    for temp, dfTemp in dfExp.groupby('temp'):
        print(f"Processing group with temp = {temp}")
        for rag, dfTempRag in dfTemp.groupby('rag'):
            for prompt, dfTempRagPrompt in dfTempRag.groupby('prompt'):
                responses = []
                for rep in range(5):
                    response = dfTempRagPrompt['response-' + str(rep)].values
                    responses.append(response[0])
                responseSimilarity.append(cosine_sim(responses, embedderModel=embedder))
                rags.append(rag)
                temps.append(temp)
                prompts.append(prompt)
            dfRes = pd.DataFrame(responseSimilarity)
            dfRes.columns = ['Rep-' + str(i) for i in dfRes.columns]
            dfRes['avgSim'] = dfRes.mean(axis=1)
            dfRes['Temperature'] = temps
            dfRes['RAG'] = rags
            dfRes['prompt'] = prompts
            dfRes.to_csv('EX-temperature-response-similarity-full-prompt.csv')


if __name__ == "__main__":
    main()
