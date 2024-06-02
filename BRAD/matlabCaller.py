import matlab.engine

def callMatlab(chatstatus, chatlog):
    prompt = chatstatus['prompt']                                        # Get the user prompt
    chatstatus['process'] = {}                                           # Begin saving plotting arguments
    chatstatus['process']['name'] = 'Matlab'    
    config_file_path = 'configMatlab.json' # we could use this to add matlab files to path
