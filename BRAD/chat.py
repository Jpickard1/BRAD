

"""
Chat Interface
~~~~~~~~~~~~~~

The below method can be used to open a chat session with BRAD without requiring the user to explicitly creat an `Agent` object.
"""

import os
import sys

# Enusre the BRAD package is accessible, even if not installed fully
parent_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_directory)

from BRAD.agent import *

def chat(
        model_path = '/nfs/turbo/umms-indikar/shared/projects/RAG/models/llama-2-7b-chat.Q8_0.gguf',
        persist_directory = "/nfs/turbo/umms-indikar/shared/projects/RAG/databases/DigitalLibrary-10-June-2024/",
        llm=None,
        ragvectordb=None,
        embeddings_model=None,
        restart=None,
        name='BRAD',
        max_api_calls=None,
        config=None
    ):
    """
    To interact with a BRAD Agent, this method instantiates a new agent, with all of the specified functionality, and uses the `chat()` method.
    This allows a user to interface with the code without creating an agent.

    :param model_path: The path to the Llama model file, defaults to '/nfs/turbo/umms-indikar/shared/projects/RAG/models/llama-2-7b-chat.Q8_0.gguf'.
    :type model_path: str, optional
    :param persist_directory: The directory where the literature database is stored, defaults to "/nfs/turbo/umms-indikar/shared/projects/RAG/databases/Transcription-Factors-5-10-2024/".
    :type persist_directory: str, optional
    :param llm: The language model to be used. If None, it will be loaded within the function.
    :type llm: PreTrainedModel, optional
    :param ragvectordb: The RAG vector database to be used. If None, it will prompt the user to load it.
    :type ragvectordb: Chroma, optional
    :param embeddings_model: The embeddings model to be used. If None, it will be loaded within the function.
    :type embeddings_model: HuggingFaceEmbeddings, optional

    :raises FileNotFoundError: If the specified model or database directories do not exist.
    :raises json.JSONDecodeError: If the configuration file contains invalid JSON.
    :raises KeyError: If required keys are missing from the configuration or chat status.

    """
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: June 4, 2024

    # Dev. Comments:
    # -------------------
    # This function executes a chat session
    #
    # History:
    # - 2024-06-04: wrote 1st draft of this code to execute a series of prompts
    #               or a conversation between the user and BRAD
    #
    #       ...     modified this funciton adding new features, logging, routing
    #               and more
    #
    # - 2024-07-10: refactored brad.py file to a class and split this method
    #               into the chatbot.__init__(), chatbot.invoke(), and
    #               chatbot.chat() methods. This method was maintained so that
    #               a user can still fun `brad.chat()` easily and without knowledge
    #               of the class structure

    bot = Agent(
        model_path = model_path,
        persist_directory = persist_directory,
        llm=llm,
        ragvectordb=ragvectordb,
        embeddings_model=embeddings_model,
        restart=restart,
        name=name,
        max_api_calls=max_api_calls,
        config=config
    )
    bot.chat()

chat()
