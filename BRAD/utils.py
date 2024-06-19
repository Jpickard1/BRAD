import re
import os
import pandas as pd

from langchain import PromptTemplate, LLMChain

from BRAD.promptTemplates import fileChooserTemplate

def save(chatstatus, df, name):
    # If this is part of a pipeline, then add the stage number to the printed output
    if len(chatstatus['planned']) != 0:
        stageNum = chatstatus['planned'][0]['order']
        name = 'S' + str(stageNum) + '-' + name
    output_path = os.path.join(chatstatus['output-directory'], name)
    df.to_csv(output_path, index=False)
    return chatstatus

def pdfDownloadPath(chatstatus):
    path = os.path.join(chatstatus['output-directory'], 'pdf')
    return path

def outputFiles(chatstatus):
    output_files = []
    for filename in os.listdir(chatstatus['output-directory']):
        output_files.append(filename)
    return output_files

def makeNamesConsistent(chatstatus, files):
    if len(chatstatus['planned']) != 0:
        stageNum = chatstatus['planned'][0]['order']
    else:
        return
    for file in files:
        old_path = os.path.join(chatstatus['output-directory'], file)
        new_path = os.path.join(chatstatus['output-directory'], 'S' + str(stageNum) + '-' + file)
        os.rename(old_path, new_path)
    return

def loadFromFile(chatstatus):
    prompt = chatstatus['prompt']
    llm    = chatstatus['llm']
    # Get files to choose from
    availableFiles = outputFiles(chatstatus)
    print(availableFiles)
    availableFiles = '\n'.join(availableFiles)

    # Build lang chain
    template = fileChooserTemplate()
    template = template.format(files=availableFiles)
    print(template)
    PROMPT   = PromptTemplate(input_variables=["user_query"], template=template)
    chain    = PROMPT | llm

    # Call chain
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

    print('File=' + str(file))
    print('Fields=' + str(fields))
    
    # Determine the delimiter based on the file extension
    delimiter = ',' if not file.endswith('.tsv') else '\t'
    
    # Read the file into a DataFrame
    df = pd.read_csv(os.path.join(chatstatus['output-directory'], file), delimiter=delimiter)

    return list(df[fields].values)

def outputFromPriorStep(chatstatus, step, values=None):
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
