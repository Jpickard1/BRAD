"""
This module is the access point for having a conversation with the LLM and no other tools.
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

import logging

from BRAD.promptTemplates import (
    history_chat_template,
    get_default_context,
)

# Extraction
import re

from BRAD import utils
from BRAD import log


def llm_only(state):
    """
    Executes a conversational interaction using a large language model (LLM) without retrieving external knowledge from a vector database.

    This function sets up and runs a conversational chain that uses a given LLM and a memory module. It constructs the prompt, tracks the LLM's usage (e.g., token count and cost), and logs the details of the interaction. The function also appends the LLM's response to the process log and displays the output to the user.

    Parameters:
        state (dict): A dictionary containing the current execution state, including:
            - 'llm' (object): The large language model to use.
            - 'prompt' (str): The user's input prompt.
            - 'databases' (dict): A dictionary of available databases; 'RAG' represents the vector database.
            - 'memory' (object): The memory module for the conversation.
            - 'config' (dict): Configuration settings, including 'debug' for verbosity.
            - 'process' (dict): A process log tracking the steps taken.

    Returns:
        dict: Updated `state` object with the LLM's response logged and user output appended.
    """
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: Nov. 17, 2024

    # Get information from Agent.state
    llm = state["llm"]  # get the llm
    prompt = state["prompt"]  # get the user prompt
    vectordb = state["databases"]["RAG"]  # get the vector database
    memory = state["memory"]  # get the memory of the model

    # Get information to invoke the llm
    template = history_chat_template()
    PROMPT = PromptTemplate(input_variables=["history", "input"], template=template)
    conversation = ConversationChain(
        prompt=PROMPT,
        llm=llm,
        verbose=state["config"]["debug"],
        memory=memory,
    )
    prompt = get_default_context() + prompt

    # Invoke LLM tracking its usage
    start_time = time.time()
    with get_openai_callback() as cb:
        response = conversation.predict(input=prompt)

    # Track llm usage
    responseDetails = {
        "content": response,
        "time": time.time() - start_time,
        "call back": {
            "Total Tokens": cb.total_tokens,
            "Prompt Tokens": cb.prompt_tokens,
            "Completion Tokens": cb.completion_tokens,
            "Total Cost (USD)": cb.total_cost,
        },
    }

    # Log the LLM response
    state["process"]["steps"].append(
        log.llmCallLog(
            llm=llm,
            prompt=PROMPT,
            input=prompt,
            output=responseDetails,
            parsedOutput=response,
            apiInfo=responseDetails["call back"],
            purpose="justchat.llm_only",
        )
    )

    # Display output to the user
    state = log.userOutput(response, state=state)

    return state
