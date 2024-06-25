from langchain import PromptTemplate, LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
import difflib
import matlab.engine
import os
import glob
import re
from BRAD.promptTemplates import matlabPromptTemplate

def callMatlab(chatstatus):
    log.debugLog("Matlab Caller Start", chatstatus=chatstatus)
    prompt = chatstatus['prompt']                                        # Get the user prompt
    llm = chatstatus['llm']                                              # Get the llm
    memory = chatstatus['memory']
    chatstatus['process'] = {}                                           # Begin saving plotting arguments
    chatstatus['process']['name'] = 'Matlab'

    # Turn on matlab engine just before running the code
    chatstatus, mpath = activateMatlabEngine()
    log.debugLog("Matlab PATH Extended, chatstatus=chatstatus)
    
    # Identify matlab files we are adding to the path of BRAD
    matlabFunctions = find_matlab_files(matlabPath)
    log.debugLog(matlabFunctions, chatstatus=chatstatus)
    # Identify which file we should use
    matlabFunction = find_closest_function(prompt, matlabFunctions)
    log.debugLog(matlabFunction, chatstatus=chatstatus)
    # Identify matlab arguments
    matlabFunctionPath = os.path.join(matlabPath, matlabFunction + '.m')
    log.debugLog(matlabFunctionPath, chatstatus=chatstatus)
    # Get matlab docstrings
    matlabDocStrings = read_matlab_docstrings(matlabFunctionPath)
    log.debugLog(matlabDocStrings, chatstatus=chatstatus)
    template = matlabPromptTemplate()
    log.debugLog(template, chatstatus=chatstatus)
    # matlabDocStrings = callMatlab(chatstatus)
    filled_template = template.format(scriptName=matlabFunction, scriptDocumentation=matlabDocStrings) #, history=None, input=None)
    log.debugLog(filled_template, chatstatus=chatstatus)
    PROMPT = PromptTemplate(input_variables=["history", "input"], template=filled_template)
    log.debugLog(PROMPT, chatstatus=chatstatus)
    conversation = ConversationChain(prompt  = PROMPT,
                                     llm     =    llm,
                                     verbose =   chatstatus['config']['debug'],
                                     memory  = memory,
                                    )
    response = conversation.predict(input=chatstatus['prompt'] )
    log.debugLog(debug, chatstatus=chatstatus)
    matlabCode = extract_matlab_code(response, chatstatus)
    log.debugLog(matlabCode, chatstatus=chatstatus)
    execute_matlab_code(matlabCode, chatstatus)
    return chatstatus

def activateMatlabEngine(chatstatus):
    if chatstatus['matlabEng'] is None:
        chatstatus['matlabEng'] = matlab.engine.start_matlab()
    log.debugLog("Matlab Engine On", chatstatus=chatstatus)
    # Add matlab files to the path
    if chatstatus['config']['matlab-path'] == 'matlab-tutorial/':
        matlabPath = chatstatus['config']['matlab-path']
    else:
        base_dir = os.path.expanduser('~')
        matlabPath = os.path.join(base_dir, chatstatus['config']['matlab-path'])
    mpath = chatstatus['matlabEng'].addpath(chatstatus['matlabEng'].genpath(matlabPath))
    return chatstatus, mpath

def execute_matlab_code(matlab_code, chatstatus):
    """
    Safely evaluates the extracted MATLAB code.
    
    Args:
    matlab_code (str): The MATLAB code to execute.

    Returns:
    None
    """
    log.debugLog("EVAL", chatstatus=chatstatus)
    log.debugLog(matlab_code, chatstatus=chatstatus)
    log.debugLog("END EVAL CHECK", chatstatus=chatstatus)
    eng = chatstatus['matlabEng']
    if matlab_code:
        try:
            # Attempt to evaluate the MATLAB code
            eval(matlab_code)
            log.debugLog("Debug: MATLAB code executed successfully.", chatstatus=chatstatus)
        except SyntaxError as se:
            log.debugLog(f"Debug: Syntax error in the MATLAB code: {se}", chatstatus=chatstatus)
        except NameError as ne:
            log.debugLog(f"Debug: Name error, possibly undefined function or variable: {ne}", chatstatus=chatstatus)
        except Exception as e:
            log.debugLog(f"Debug: An error occurred during MATLAB code execution: {e}", chatstatus=chatstatus)
    else:
        log.debugLog("Debug: No MATLAB code to execute.", chatstatus=chatstatus)


def find_matlab_files(path):

    # Construct the search pattern for .m files
    search_pattern = os.path.join(path, '**', '*.m')
    
    # Find all .m files recursively
    matlab_files = glob.glob(search_pattern, recursive=True)

    # Extract only the file names from the full paths
    file_names = [os.path.basename(file)[:-2] for file in matlab_files]
    
    return file_names

def find_closest_function(query, functions):
    # Find the closest matches to the query in the list of functions
    closest_matches = difflib.get_close_matches(query, functions, n=1, cutoff=0.0)
    
    if closest_matches:
        return closest_matches[0]
    else:
        return None

def read_matlab_docstrings(file_path):
    docstrings = []
    first = True
    with open(file_path, 'r') as file:
        for line in file:
            # log.debugLog(file, chatstatus=chatstatus)
            stripped_line = line.strip()
            if stripped_line.startswith('%') or first:
                # Remove the '%' character and any leading/trailing whitespace
                docstrings.append(stripped_line)
                first = False
            else:
                # Stop reading if we encounter a line that doesn't start with '%'
                # assuming docstrings are at the beginning of the file
                break
    return "\n".join(docstrings)

def get_matlab_description(file_path):
    docstrings = read_matlab_docstrings(file_path)
    oneliner = docstrings.split('\n')[1][1:]
    return oneliner

def extract_matlab_code(llm_output, chatstatus):
    """
    Parses the LLM output and extracts the MATLAB code in the final line.
    
    Args:
    llm_output (str): The complete output from the LLM.

    Returns:
    str: The MATLAB code to execute.
    """
    log.debugLog('LLM OUTPUT PARSER', chatstatus=chatstatus)
    log.debugLog(llm_output, chatstatus=chatstatus)
    funcCall = llm_output.split('Execute:')
    log.debugLog(funcCall, chatstatus=chatstatus)
    funcCall = funcCall[len(funcCall)-1].strip()
    if funcCall[:3] != 'eng':
        log.debugLog(funcCall, chatstatus=chatstatus)
        funcCall = 'eng' + funcCall[3:]
        log.debugLog(funcCall, chatstatus=chatstatus)
    return funcCall
    # Define the regex pattern to match the desired line
    #pattern = r'Execute: `(.+)`'

    # Search for the pattern in the LLM output
    #match = re.search(pattern, llm_output)

    # If a match is found, return the MATLAB code, else return None
    #if match:
    #    return match.group(1)
    #else:
    #    return None


def callMatlab_depricated(chatstatus, chatlog):
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


