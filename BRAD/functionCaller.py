"""    
    .. warning:: We may be removing this file soon.
"""
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import ast
import time

def getFunctionArgs(chatstatus):
    """
    Performs inference to fill in missing values in a dictionary based on a given prompt.

    :param chatstatus: A dictionary containing the prompt, process parameters, and configuration settings.
    :type chatstatus: dict

    :raises FileNotFoundError: If the model or tokenizer files are not found in the specified directory.
    :raises KeyError: If required keys are missing from the chatstatus dictionary.
    :raises RuntimeError: If the model cannot be moved to the specified device (CPU or GPU).

    :return: A dictionary with the same keys as the input process parameters but with inferred values based on the prompt.
    :rtype: dict
    """
    prompt    = chatstatus['prompt']
    plot_args = chatstatus['process']['params']

    if chatstatus['config']['debug']:
        chatstatus = log.userOutput("Possible Values:", chatstatus=chatstatus)
        display(plot_args)

    # Directory where the model and tokenizer are saved
    model_dir = '/home/jpic/JP-CS-CrazyIdea/t5-small-finetuned-data-v4' # t5-small-finetuned-data-v2'
    log.debugLog(model_dir, chatstatus=chatstatus)

    # Load the tokenizer and model from the saved directory
    tokenizer = T5Tokenizer.from_pretrained(model_dir)
    model = T5ForConditionalGeneration.from_pretrained(model_dir)

    # Move model to appropriate device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    plot_args_empty = plot_args.copy()
    for k in plot_args_empty.keys():
        plot_args_empty[k] = None

    # Example prompt for inference
    example_prompt = (
        f"Given the following dictionary with missing values:\n\n"
        f"{plot_args_empty}\n\n"
        f"Fill in the missing values based on the information provided in the following sentence. If a key is not found in the sentence, leave its value as None:\n\n"
        f"\"{prompt}\""
    )

    start_time = time.time()
    plot_args_infr = perform_inference(example_prompt, model, tokenizer, device)
    end_time = time.time()
    
    for k in plot_args.keys():
        if k not in plot_args_infr.keys():
            plot_args_infr[k] = plot_args[k]

    if chatstatus['config']['debug']:
        log.debugLog("Inference result:", chatstatus=chatstatus)
        display(plot_args_infr)
        log.debugLog(f"Time taken for inference: {end_time - start_time} seconds", chatstatus=chatstatus)
    return plot_args_infr
    
# Function to perform inference
def perform_inference(prompt, model, tokenizer, device):
    """
    Performs inference using a pre-trained model to generate a dictionary of inferred values from a given prompt.

    :param prompt: The input prompt containing the information for the model to process.
    :type prompt: str
    :param model: The pre-trained model used for generating the output.
    :type model: torch.nn.Module
    :param tokenizer: The tokenizer used for preparing the input prompt for the model.
    :type tokenizer: transformers.PreTrainedTokenizer
    :param device: The device (CPU or GPU) on which the model and inputs are placed.
    :type device: torch.device

    :raises ValueError: If the result cannot be parsed into a dictionary.

    :return: A dictionary with inferred values based on the input prompt.
    :rtype: dict
    """
    model.eval()  # Set the model to evaluation mode
    inputs = tokenizer(prompt, return_tensors='pt', max_length=512, padding='max_length', truncation=True).to(device)
    with torch.no_grad():
        outputs = model.generate(inputs.input_ids, max_length=512)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return ast.literal_eval('{'+result+'}')
