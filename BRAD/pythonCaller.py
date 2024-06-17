from langchain import PromptTemplate, LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
import difflib
import matlab.engine
import os
import glob
import re
import sys
import subprocess

from BRAD.promptTemplates import pythonPromptTemplate


def callPython(chatstatus):
    print('Python Caller Start')
    prompt = chatstatus['prompt']                                        # Get the user prompt
    llm = chatstatus['llm']                                              # Get the llm
    memory = chatstatus['memory']
    chatstatus['process'] = {}                                           # Begin saving plotting arguments
    chatstatus['process']['name'] = 'Python'

    # Add matlab files to the path
    if chatstatus['config']['py-path'] == 'py-tutorial/': # Admittedly, I'm not sure what's going on here - JP
        pyPath = chatstatus['config']['py-path']
    else:
        base_dir = os.path.expanduser('~')
        pyPath = os.path.join(base_dir, chatstatus['config']['py-path'])
    print('Python scripts added to PATH') if chatstatus['config']['debug'] else None

    # Identify python scripts files we are adding to the path of BRAD
    pyScripts = find_py_files(pyPath)
    print(pyScripts) if chatstatus['config']['debug'] else None
    # Identify which file we should use
    pyScript = find_closest_function(prompt, pyScripts)
    print(pyScript) if chatstatus['config']['debug'] else None
    # Identify pyton script arguments
    pyScriptPath = os.path.join(pyPath, pyScript + '.py')
    print(pyScriptPath) if chatstatus['config']['debug'] else None
    # Get matlab docstrings
    pyScriptDocStrings = read_python_docstrings(pyScriptPath)
    print(pyScriptDocStrings) if chatstatus['config']['debug'] else None
    template = pythonPromptTemplate()
    print(template) if chatstatus['config']['debug'] else None
    # matlabDocStrings = callMatlab(chatstatus)
    filled_template = template.format(scriptName=pyScriptPath, scriptDocumentation=pyScriptDocStrings)
    print(filled_template) if chatstatus['config']['debug'] else None
    PROMPT = PromptTemplate(input_variables=["history", "input"], template=filled_template)
    print(PROMPT) if chatstatus['config']['debug'] else None
    conversation = ConversationChain(prompt  = PROMPT,
                                     llm     =    llm,
                                     verbose =   chatstatus['config']['debug'],
                                     memory  = memory,
                                    )
    response = conversation.predict(input=chatstatus['prompt'] )
    print('START RESPONSE') if chatstatus['config']['debug'] else None
    print(response) if chatstatus['config']['debug'] else None
    print('END RESPONSE') if chatstatus['config']['debug'] else None
    matlabCode = extract_python_code(response, chatstatus)
    print(matlabCode) if chatstatus['config']['debug'] else None
    execute_python_code(matlabCode, chatstatus)
    return chatstatus

def execute_python_code(python_code, chatstatus):
    """
    Safely evaluates the extracted PYTHON code.

    Args:
    python_code (str): The MATLAB code to execute.

    Returns:
    None
    """
    print('EVAL') if chatstatus['config']['debug'] else None
    print(python_code) if chatstatus['config']['debug'] else None
    print('END EVAL CHECK') if chatstatus['config']['debug'] else None
    if python_code:
        try:
            # Attempt to evaluate the MATLAB code
            eval(python_code)
            print("Debug: PYTHON code executed successfully.")
        except SyntaxError as se:
            print(f"Debug: Syntax error in the PYTHON code: {se}")
        except NameError as ne:
            print(f"Debug: Name error, possibly undefined function or variable: {ne}")
        except Exception as e:
            print(f"Debug: An error occurred during PYTHON code execution: {e}")
    else:
        print("Debug: No PYTHON code to execute.")


def find_py_files(path):

    # Construct the search pattern for .m files
    search_pattern = os.path.join(path, '**', '*.py')
    
    # Find all .m files recursively
    py_files = glob.glob(search_pattern, recursive=True)

    # Extract only the file names from the full paths
    file_names = [os.path.basename(file)[:-3] for file in py_files]
    
    return file_names

def find_closest_function(query, functions):
    # Find the closest matches to the query in the list of functions
    closest_matches = difflib.get_close_matches(query, functions, n=1, cutoff=0.0)
    
    if closest_matches:
        return closest_matches[0]
    else:
        return None

def read_python_docstrings(file_path):
    docstring_lines = []
    inside_docstring = False
    
    with open(file_path, 'r') as file:
        for line in file:
            stripped_line = line.strip()
            if stripped_line.startswith('"""') or stripped_line.startswith("'''"):
                if inside_docstring:
                    # This ends the docstring
                    docstring_lines.append(stripped_line)
                    break
                else:
                    # This starts the docstring
                    docstring_lines.append(stripped_line)
                    inside_docstring = True
            elif inside_docstring:
                # Collect lines inside the docstring
                docstring_lines.append(line.rstrip())
            elif not stripped_line:
                # Skip any leading empty lines
                continue
            else:
                # Stop if we encounter a non-docstring line and are not inside a docstring
                break
    
    return "\n".join(docstring_lines)

def get_py_description(file_path):
    docstrings = read_python_docstrings(file_path)
    oneliner = docstrings.split('\n')[1]
    return oneliner

def extract_python_code(llm_output, chatstatus):
    """
    Parses the LLM output and extracts the MATLAB code in the final line.

    Args:
    llm_output (str): The complete output from the LLM.

    Returns:
    str: The MATLAB code to execute.
    """
    print('LLM OUTPUT PARSER') if chatstatus['config']['debug'] else None
    print(llm_output) if chatstatus['config']['debug'] else None
    funcCall = llm_output.split('Execute:')
    print(funcCall) if chatstatus['config']['debug'] else None
    funcCall = funcCall[len(funcCall)-1].strip()
    if funcCall[:32] != 'subprocess.call([sys.executable,':
        print(funcCall) if chatstatus['config']['debug'] else None
        funcCall = 'subprocess.call([sys.executable,' + funcCall[32:]
        print(funcCall) if chatstatus['config']['debug'] else None
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
