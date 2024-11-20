"""
This module provides utilities for loading various language models, including OpenAI models, LLaMAs that
run locally, or any model hosted on the NVIDIA NIM platform, for use within the BRAD framework. The module
defines functions that facilitate the setup and initialization of these models by specifying key parameters
such as model paths,  API keys, and optional configurations like token limits and temperature. By default, a
BRAD `Agent` will use OpenAI's `gpt-3.5-turbo-0125` model.

Available LLMs
--------------

The BRAD architecture is LLM agnostic and interoperable with different LLMs. For convenience, the following LLMs have been integrated into BRAD (1-3), but in prinicpal, 
any LLM could be used in this system

1. `OpenAI <https://openai.com/>`_
    The OpenAI API supports the use of gpt3, gpt-4o, and soon the o1 series of LLMs. To use these models, the user must (1) provide an OpenAI API key and (2) select an LLM

2. `NVIDA <https://build.nvidia.com/explore/discover>`_
    NVIDIA hosts LLMs from a variety of providers to streamline using LLMs from a variety of providers. Currently, these include `Llama <https://www.llama.com/>`_ models from META AI, `Gemma <https://ai.google.dev/gemma>`_ from Google, `Nemotron <https://docs.nvidia.com/nemo-framework/user-guide/latest/llms/nemotron/index.html>`_ from NVIDIA, and other models from Microsft, Stability AI, Mistral AI, and more. To use these models, a user must supply an API key from NVIDIA.

3. Llama with `llama.cpp <https://github.com/ggerganov/llama.cpp>`_
    A user can run LLM inference locally and integrate their LLM with BRAD. This is available directly for Llama models using the `llama.cpp` interface. Running models locally allows users to finetune their LLM of choice to be further customized for their usecase. This requires the user to have the hardward to support running LLM inference, independent of the BRAD architecture.

4. Any LLM with `LangChain <https://python.langchain.com/docs/how_to/custom_llm/>`_
    In principle, any LLM can be integrated into this system. To integrate a custom LLM or that from a different provider besides OpenAI or NVIDIA, the LLM must be made `LangChain` compatible. This can be done with minimal code, similar to the `BradLLM` class.

Loading LLMs
------------
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

    :param model_name: Name of the NVIDIA model to load.
    :type model_name: str, optional
    :param nvidia_api_key: API key for accessing NVIDIA's services. If not provided, it will be prompted.
    :type nvidia_api_key: str, optional
    :param temperature: temperature (i.e. creativity or randomness) of the llm
    :type temperature: float, optional

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



def load_openai(model_name='gpt-3.5-turbo-0125', api_key=None, temperature=0):
    """
    Loads the OPENAI language model with the specified model name and API key.

    :param model_name: Name of the OPENAI model to load.
    :type model_name: str, optional
    :param api_key: API key for accessing OPENAI's services. If not provided, it will be prompted.
    :type api_key: str, optional
    :param temperature: temperature (i.e. creativity or randomness) of the llm
    :type temperature: float, optional (default 0)

    :raises AssertionError: If the provided OPENAI API key is not valid.

    :return: The loaded OPENAI language model.
    :rtype: langchain_openai.ChatOpenAI

    :example:
        >>> openai_model = load_openai()
    
    """
    # Auth: Marc Choi
    #       machoi@umich.edu
    # Date: July 1, 2024
    from openai import OpenAI
    from langchain_openai import ChatOpenAI

    #all the Open AI API keys do not all start with a similar character
    if not os.environ.get("OPENAI_API_KEY", "").startswith("sk-"):
        api_key = getpass.getpass("Enter your Open AI API key: ")
        assert api_key.startswith("sk-"), f"{api_key}... is not a valid key"
        os.environ["OPENAI_API_KEY"] = api_key
    else:
        api_key = os.environ["OPENAI_API_KEY"]
        
    llm = ChatOpenAI(
        model=model_name,
        temperature=temperature,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )

    return llm

def llm_switcher(llm_choice, temperature=0):
    llm_host = 'OPENAI' if 'gpt' in llm_choice.lower() else 'NVIDIA'
    if llm_host == "NVIDIA":
        print(f"{llm_choice=}")
        llm = load_nvidia(
            model_name = llm_choice,
            temperature = temperature,
        )
        print(f"{llm=}")
    elif llm_host == "OPENAI":
        llm = load_openai(
            model_name = llm_choice,
            temperature = temperature,
        )
    else:
        raise Exception("not a valid llm model")
    
    return llm