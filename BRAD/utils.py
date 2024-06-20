import re
import os
import pandas as pd
import subprocess
import difflib

from langchain import PromptTemplate, LLMChain
from BRAD import log
from BRAD.promptTemplates import fileChooserTemplate, fieldChooserTemplate

def save(chatstatus, data, name):
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: June 19, 2024

    # If this is part of a pipeline, then add the stage number to the printed output
    if len(chatstatus['planned']) != 0:
        stageNum = chatstatus['planned'][0]['order']
        name = 'S' + str(stageNum) + '-' + name
    output_path = os.path.join(chatstatus['output-directory'], name)

    if isinstance(data, pd.DataFrame):
        data.to_csv(output_path, index=False)
    elif output_path.endswith('.tex'):
        with open(output_path, 'w') as file:
            file.write(data)
    else:
        raise ValueError("Unsupported data type or file extension. Use a DataFrame for CSV or a string for .tex files.")
    
    log.debugLog('The information has been saved to: ' + output_path, chatstatus=chatstatus)
    chatstatus['process']['steps'].append(
        {
            'func'     : 'utils.save',
            'new file' : output_path
        }
    )
    return chatstatus

def pdfDownloadPath(chatstatus):
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: June 19, 2024
    path = os.path.join(chatstatus['output-directory'], 'pdf')
    return path

def outputFiles(chatstatus):
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: June 19, 2024
    output_files = []
    for filename in os.listdir(chatstatus['output-directory']):
        output_files.append(filename)
    return output_files

def makeNamesConsistent(chatstatus, files):
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: June 19, 2024
    if len(chatstatus['planned']) != 0:
        stageNum = chatstatus['planned'][0]['order'] + 1
    else:
        return
    renamedFiles = []
    for file in files:
        if file[0] != 'S':
            old_path = os.path.join(chatstatus['output-directory'], file)
            new_path = os.path.join(chatstatus['output-directory'], 'S' + str(stageNum) + '-' + file)
            renamedFiles.append(
                {
                    'old-name' : old_path,
                    'new-name' : new_path
                }
            )
            os.rename(old_path, new_path)
    for file in outputFiles(chatstatus):
        old_path = os.path.join(chatstatus['output-directory'], file)
        new_path = os.path.join(chatstatus['output-directory'], file.replace('/', '').replace('\\', ''))
        renamedFiles.append(
            {
                'old-name' : old_path,
                'new-name' : new_path
            }
        )
        os.rename(old_path, new_path)
    chatstatus['process']['steps'].append(
        {
            'func'  : 'utils.makeNamesConsistent',
            'files' : renamedFiles
        }
    )
    return chatstatus

def loadFromFile(chatstatus):
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: June 19, 2024
    prompt = chatstatus['prompt']
    llm    = chatstatus['llm']
    # Get files to choose from
    availableFiles = outputFiles(chatstatus)
    availableFiles = '\n'.join(availableFiles)
    log.debugLog(availableFiles, chatstatus=chatstatus)
    
    # Build lang chain
    template = fileChooserTemplate()
    template = template.format(files=availableFiles)
    print(template)
    PROMPT   = PromptTemplate(input_variables=["user_query"], template=template)
    chain    = PROMPT | llm

    # Call chain
    print(prompt)
    response = chain.invoke(prompt).content.strip()
    
    # Regular expressions to extract file and fields
    file_pattern = r"File:\s*(\S+)"
    fields_pattern = r"Fields:\s*(.+)"

    # Search for patterns in the response
    file_match = re.search(file_pattern, response)
    fields_match = re.search(fields_pattern, response)

    # Extract the matched values
    file = file_match.group(1) if file_match else None
    fields = fields_match.group(1) if fields_match else None

    log.debugLog('File=' + str(file) + '\n' + 'Fields=' + str(fields), chatstatus=chatstatus)
    chatstatus['process']['steps'].append(log.llmCallLog(llm          = llm,
                                                         prompt       = PROMPT,
                                                         input        = prompt,
                                                         output       = response,
                                                         parsedOutput = {
                                                             'File'   : file,
                                                             'Fields' : fields
                                                         },
                                                         purpose      = 'Select File'
                                                        )
                                        )
    
    # Determine the delimiter based on the file extension
    delimiter = ',' if not file.endswith('.tsv') else '\t'
    
    # Read the file into a DataFrame
    loadfile = os.path.join(chatstatus['output-directory'], file)
    df = pd.read_csv(loadfile, delimiter=delimiter)
    chatstatus['process']['steps'].append(log.loadFileLog(file      = loadfile,
                                                          delimiter = delimiter)
                                         )

    if fields not in df.columns:
        chatstatus, fields = fieldSelectorFromDataFrame(chatstatus, df)

    return chatstatus, list(df[fields].values)

def fieldSelectorFromDataFrame(chatstatus, df):
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: June 19, 2024
    llm      = chatstatus['llm']
    prompt   = chatstatus['prompt']
    template = fieldChooserTemplate()
    template = template.format(columns=', '.join(list(df.columns)))
    PROMPT   = PromptTemplate(input_variables=["user_query"], template=template)
    chain    = PROMPT | llm

    # Call chain
    response = chain.invoke(prompt).content.strip()
    fields = response.split('=')[1].strip()
    chatstatus['process']['steps'].append(log.llmCallLog(llm          = llm,
                                                         prompt       = PROMPT,
                                                         input        = prompt,
                                                         output       = response,
                                                         parsedOutput = {
                                                             'Fields' : fields
                                                         },
                                                         purpose      = 'Select Field'
                                                        )
                                        )

    log.debugLog('field identifier response=\n'+fields, chatstatus=chatstatus)
    return chatstatus, fields

def word_similarity(word1, word2):
    return difflib.SequenceMatcher(None, word1, word2).ratio()

def outputFromPriorStep(chatstatus, step, values=None):
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: June 19, 2024
    print(chatstatus)
    print(step)
    step_output_files = []
    file = None
    for filename in os.listdir(chatstatus['output-directory']):
        if filename.startswith('S'):
            step_output_files.append(filename)
        if filename.startswith('S' + step):
            file = filename
    print(file)
    if file.endswith('.csv'):
        file_path = os.path.join(chatstatus['output-directory'], file)
        df = pd.read_csv(file_path)
        print(df)
        if values is not None:
            df = df[values]
    return df

def compile_latex_to_pdf(chatstatus, tex_file):
    """This function converts .tex files to .pdf files."""
    tex_file_path = os.path.join(chatstatus['output-directory'], tex_file)
    
    # Ensure the file exists
    if not os.path.isfile(tex_file_path):
        raise FileNotFoundError(f"The file {tex_file_path} does not exist.")
    
    # Run the pdflatex command with the specified output directory
    try:
        subprocess.run(
            ['pdflatex', '-output-directory', chatstatus['output-directory'], tex_file_path], 
            check=True
        )
        print(f"PDF generated successfully in {chatstatus['output-directory']}.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")

    return chatstatus