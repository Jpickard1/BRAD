import matlab.engine


def callMatlab(chatstatus, chatlog):
    """
    THIS IS THE INCORRECT DOCUMENTATION FOR THIS FUNCTION IT IS ONLY A TEST DELETE LATER
    IT DOES NOT WORK... WHY?
    Performs a search on Gene Ontology (GO) based on the provided query and allows downloading associated charts and papers.

    :param query: The query list containing gene names or terms for GO search.
    :type query: list

    :return: A dictionary containing the GO search process details.
    :rtype: dict

    """
    prompt = chatstatus['prompt']                                        # Get the user prompt
    chatstatus['process'] = {}                                           # Begin saving plotting arguments
    chatstatus['process']['name'] = 'Matlab'    
    config_file_path = 'configMatlab.json' # we could use this to add matlab files to path
    eng = matlab.engine.start_matlab()


def 