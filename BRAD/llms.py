import os
import getpass

def load_llama(model_path = '/nfs/turbo/umms-indikar/shared/projects/RAG/models/llama-2-7b-chat.Q8_0.gguf',
               n_ctx = 4096,
               max_tokens = 1000,
               verbose = False):
    """
    Load and initialize the Llama model with specified parameters.

    Parameters:
    model_path (str): The file path to the Llama model. Defaults to 
                      '/nfs/turbo/umms-indikar/shared/projects/RAG/models/llama-2-7b-chat.Q8_0.gguf'.
    n_ctx (int): The context length for the model. Defaults to 4096.
    max_tokens (int): The maximum number of tokens to generate. Defaults to 1000.
    verbose (bool): If True, enable verbose logging. Defaults to False.

    Returns:
    LlamaCpp: An instance of the LlamaCpp model initialized with the specified parameters.
    """
    from langchain.llms import LlamaCpp
    from langchain.callbacks.manager import CallbackManager
    from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    llm = LlamaCpp(model_path       = model_path,
                   n_ctx            = n_ctx,
                   max_tokens       = max_tokens,
                   callback_manager = callback_manager,
                   verbose          = verbose)
    return llm

def load_nvidia(nvidia_model='mistral_7b', nvidia_api_key=None):
    """
    Load and initialize an NVIDIA model with the specified parameters.

    Parameters:
    nvidia_model (str): The name of the NVIDIA model to load. Defaults to 'mistral_7b'.
    nvidia_api_key (str, optional): The API key for accessing NVIDIA's services. If not provided,
                                    the function will prompt for it.

    Returns:
    ChatNVIDIA: An instance of the ChatNVIDIA model initialized with the specified parameters.

    Raises:
    AssertionError: If the provided or entered API key does not start with "nvapi-".
    """
    from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
    if not os.environ.get("NVIDIA_API_KEY", "").startswith("nvapi-"):
        nvidia_api_key = getpass.getpass("Enter your NVIDIA API key: ")
        assert nvidia_api_key.startswith("nvapi-"), f"{nvidia_api_key[:5]}... is not a valid key"
        os.environ["NVIDIA_API_KEY"] = nvidia_api_key
    else:
        nvidia_api_key = os.environ["NVIDIA_API_KEY"]
        
    llm = ChatNVIDIA(model   = nvidia_model,
                     api_key = nvidia_api_key,
          )
    return llm