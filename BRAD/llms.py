"""
This module provides utilities for loading various language models, including LLaMA, NVIDIA, 
and OpenAI models, for use within the BRAD framework. The module defines functions that facilitate
the setup and initialization of these models by specifying key parameters such as model paths, 
API keys, and optional configurations like token limits and temperature.

Available Functions:
---------------------
1. load_llama:
    Loads the LLaMA language model from a specified model path, with configurable parameters 
    such as context length, maximum tokens, and temperature. This model is primarily loaded
    using the LangChain framework.

2. load_nvidia:
    Loads an NVIDIA language model using the provided model name and API key. If no API key 
    is provided, the function prompts the user to input it interactively. This model is loaded 
    using NVIDIA's endpoints within the LangChain framework.

3. load_openai:
    Loads an OpenAI language model using the specified model name and an API key. If the key 
    is not available in the environment, the user is prompted for it. The OpenAI model is 
    initialized via LangChain's OpenAI integration.

Usage:
------
Each function in this module supports customizable arguments to load the desired model. 
These functions are designed to integrate with the BRAD framework and provide easy access 
to large language models through LangChain.

Example Usage:
--------------
To load a LLaMA model:

>>> llama_model = load_llama(model_path='/path/to/llama-model')

To load an NVIDIA model:

>>> nvidia_model = load_nvidia(model_name='meta/llama3-70b-instruct')

To load an OpenAI model:

>>> openai_model = load_openai(model_name='gpt-3.5-turbo')
"""

import os
import getpass

def load_llama(model_path = '/nfs/turbo/umms-indikar/shared/projects/RAG/models/llama-2-7b-chat.Q8_0.gguf',
               n_ctx = 4096,
               max_tokens = 1000,
               temperature = 0,
               verbose = False,
              ):
    """
    Loads the Llama language model from the specified model path with given parameters.

    :param model_path: Path to the Llama model file.
    :type model_path: str, optional
    :param n_ctx: Number of context tokens for the model.
    :type n_ctx: int, optional
    :param max_tokens: Maximum number of tokens for the model's output.
    :type max_tokens: int, optional
    :param verbose: If True, enables verbose logging.
    :type verbose: bool, optional

    :return: The loaded Llama model.
    :rtype: langchain.llms.LlamaCpp

    :example:
    >>> llama_model = load_llama()
    """
    from langchain.llms import LlamaCpp
    from langchain.callbacks.manager import CallbackManager
    from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    llm = LlamaCpp(model_path       = model_path,
                   n_ctx            = n_ctx,
                   max_tokens       = max_tokens,
                   callback_manager = callback_manager,
                   temperature      = temperature,
                   verbose          = verbose)
    return llm

def load_nvidia(model_name='meta/llama3-70b-instruct', nvidia_api_key=None, temperature=None):
    """
    Loads the NVIDIA language model with the specified model name and API key.

    :param nvidia_model: Name of the NVIDIA model to load.
    :type nvidia_model: str, optional
    :param nvidia_api_key: API key for accessing NVIDIA's services. If not provided, it will be prompted.
    :type nvidia_api_key: str, optional

    :raises AssertionError: If the provided NVIDIA API key is not valid.

    :return: The loaded NVIDIA language model.
    :rtype: langchain_nvidia_ai_endpoints.ChatNVIDIA

    :example:
    >>> nvidia_model = load_nvidia()
    """
    from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
    if not os.environ.get("NVIDIA_API_KEY", "").startswith("nvapi-"):
        nvidia_api_key = getpass.getpass("Enter your NVIDIA API key: ")
        assert nvidia_api_key.startswith("nvapi-"), f"{nvidia_api_key[:5]}... is not a valid key"
        os.environ["NVIDIA_API_KEY"] = nvidia_api_key
    else:
        nvidia_api_key = os.environ["NVIDIA_API_KEY"]
        
    llm = ChatNVIDIA(model       = model_name,
                     api_key     = nvidia_api_key,
                     temperature = temperature,
          )
    return llm

    



def load_openai(model_name='gpt-3.5-turbo-0125', api_key=None):
    """
    Loads the NVIDIA language model with the specified model name and API key.

    :param nvidia_model: Name of the NVIDIA model to load.
    :type nvidia_model: str, optional
    :param nvidia_api_key: API key for accessing NVIDIA's services. If not provided, it will be prompted.
    :type nvidia_api_key: str, optional

    :raises AssertionError: If the provided NVIDIA API key is not valid.

    :return: The loaded NVIDIA language model.
    :rtype: langchain_nvidia_ai_endpoints.ChatNVIDIA

    :example:
    >>> nvidia_model = load_nvidia()
    """
    # Auth: Marc Choi
    #       machoi@umich.edu
    # Date: July 1, 2024
    from openai import OpenAI
    from langchain_openai import ChatOpenAI
    from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
    #all the Open AI API keys do not all start with a similar character
    if not os.environ.get("OPENAI_API_KEY", "").startswith("sk-"):
        api_key = getpass.getpass("Enter your Open AI API key: ")
        assert api_key.startswith("sk-"), f"{api_key}... is not a valid key"
        os.environ["OPENAI_API_KEY"] = api_key
    else:
        api_key = os.environ["OPENAI_API_KEY"]
        
    llm = ChatOpenAI(
        model=model_name,
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )

    return llm