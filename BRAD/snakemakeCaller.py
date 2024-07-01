import subprocess
import shlex
from typing import Dict, Any
import json
from BRAD import log

def callSnakemake(chatstatus, chatlog):
    """
    This function calls Snakemake with parameters extracted from chatstatus.

    :param chatstatus: A dictionary containing chat status information.
    :type chatstatus: dict
    :param chatlog: A log of the chat.
    :type chatlog: str

    The function retrieves the user prompt from chatstatus, reads Snakemake
    parameters from a JSON configuration file, and then executes Snakemake
    with the retrieved parameters.

    It also updates chatstatus with information about the Snakemake process.

    :return: None
    :rtype: None

    """
    
    prompt = chatstatus['prompt']                                        # Get the user prompt
    chatstatus['process'] = {}                                           # Begin saving plotting arguments
    chatstatus['process']['name'] = 'Snakemake'    
    config_file_path = 'configSnakemake.json'
    chatstatus['process']['params'] = read_config(config_file_path)
    chatstatus['process']['params'] = getFunctionArgs(chatstatus)            # Apply t5-small fine tuned to extract plotting args
    run_snakemake(chatstatus['process']['params'])

def read_config(file_path='configSnakemake.json') -> Dict[str, Any]:
    """
    Read JSON configuration from a file.

    :param file_path: Path to the JSON file containing the configuration.
    :type file_path: str

    :return: A dictionary representing the configuration.
    :rtype: dict

    """
    with open(file_path, 'r') as file:
        config = json.load(file)
    return config

def run_snakemake(params: Dict[str, Any]):
    """
    Run a Snakemake workflow with the given parameters.

    :param params: A dictionary of parameters where the keys are the Snakemake options
                   and the values are the corresponding arguments.
    :type params: dict

    """
    command = ["snakemake"]
    
    for param, value in params.items():
        if isinstance(value, bool):
            if value:
                command.append(param)
        elif isinstance(value, list):
            if value:
                command.append(param)
                command.extend(value)
        elif isinstance(value, dict):
            if value:
                command.append(param)
                command.append(" ".join([f"{k}={v}" for k, v in value.items()]))
        elif value is not None:
            command.append(param)
            command.append(str(value))
    
    command_str = " ".join(shlex.quote(arg) for arg in command)
    
    try:
        result = subprocess.run(command_str, shell=True, check=True, capture_output=True, text=True)
        log.debugLog(result.stdout, chatstatus=chatstatus)
    except subprocess.CalledProcessError as e:
        log.debugLog(f"Error running Snakemake: {e.stderr}", chatstatus=chatstatus)
        raise
