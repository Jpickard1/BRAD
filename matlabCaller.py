import matlab.engine

def callSnakemake(chatstatus, chatlog):
    prompt = chatstatus['prompt']                                        # Get the user prompt
    chatstatus['process'] = {}                                           # Begin saving plotting arguments
    chatstatus['process']['name'] = 'Snakemake'    
    config_file_path = 'configSnakemake.json'
    chatstatus['process']['params'] = read_config(config_file_path)
    chatstatus['process']['params'] = getFunctionArgs(chatstatus)            # Apply t5-small fine tuned to extract plotting args
    run_snakemake(chatstatus['process']['params'])