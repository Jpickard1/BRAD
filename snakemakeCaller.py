import subprocess
import shlex
from typing import Dict, Any
import json

def callSnakemake(chatstatus, chatlog):
    """
    Call Snakemake with parameters extracted from chatstatus.

    Parameters:
    - chatstatus: A dictionary containing chat status information.
    - chatlog: A log of the chat.

    The function retrieves the user prompt from chatstatus, reads Snakemake
    parameters from a JSON configuration file, and then executes Snakemake
    with the retrieved parameters.

    It also updates chatstatus with information about the Snakemake process.

    Returns:
    - None
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

    Parameters:
    - file_path: Path to the JSON file containing the configuration.

    Returns:
    - A dictionary representing the configuration.
    """
    with open(file_path, 'r') as file:
        config = json.load(file)
    return config

def run_snakemake(params: Dict[str, Any]):
    """
    Run a Snakemake workflow with the given parameters.

    Parameters:
    - params: A dictionary of parameters where the keys are the Snakemake options
              and the values are the corresponding arguments.
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
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error running Snakemake: {e.stderr}")
        raise
