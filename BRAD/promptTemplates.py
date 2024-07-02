from pydantic import BaseModel

# Based on the user's query, create a detailed plan outlining the steps of the analysis. Ensure that each step is clearly defined and makes use of an appropriate method. Clarify which steps are dependent on one another, such as if information from a CODE step is used by DATABASE step, if information from a RAG step is used by a SCRAPE step, or if all previous steps are needed in a WRITE step, or any other dependencies.
def plannerTemplate():
    template = """**INSTRUCTIONS:**
You are planning a bioinformatics analysis pipeline to address a user's query. Your task is to outline a multi-step workflow using the available methods listed below. Each method should be used appropriately to ensure a thorough analysis. For each step in the pipeline, explain the goal of the process as well as provide a prompt to the chatbot that will execute that step. If information is passed between stages, such as from literature or code to databases, indicate the dependencies of steps in the prompt.

**Available Methods:**
1. **RAG**: Look up literature and documents from a text database.
2. **SCRAPE**: Search platforms like arXiv, bioRxiv, and PubMed for the latest research.
3. **DATABASE**: Utilize bioinformatics databases such as Gene Ontology and Enrichr to perform gene set enrichment analyses.
4. **CODE**: Execute bioinformatics pipelines. Utilize prebuilt pipelines or develop new ones as needed.
5. **WRITE**: Synthesize and summarize information. This can include summarizing database searches, code pipeline results, or creating a final report to encapsulate the entire analysis.
6. **ROUTER**: Determine which step we should proceed to next.

Current conversation: {history}\n

User Query: {input}

Plan:
**Step 1 (method, eg. RAG)**:
Prompt: [Description of action for chatbot to do Step 1, i.e. do X, Y, Z]

**Step 2 (method, eg. CODE)**:
Prompt: [Description of action for chatbot to do Step 2, i.e. do A, B, C. Use information from Step X]
...
"""
    return template

def plannerEditingTemplate():
    template = """Based on the most recently proposed Current Plan and the user's new requirements and requested changes, create a revised plan.

Current Plan: {plan}\n

User's requested changes: {{input}}

**Available Methods:**
1. **Retrieval Augmented Generation**(RAG): Look up literature and documents from a text database.
2. **Web Search for New Literature**(SCRAPE): Search platforms like arXiv, bioRxiv, and PubMed for the latest research.
3. **Bioinformatics Databases**(DATABASE): Utilize databases such as Gene Ontology and Enrichr to perform gene set enrichment analyses.
4. **Run Codes**(CODE): Execute bioinformatics pipelines. Utilize prebuilt pipelines or develop new ones as needed.
5. **Write a Report**(WRITE): Synthesize and summarize information. This can include summarizing database searches, code pipeline results, or creating a final report to encapsulate the entire analysis.

**OUTPUT**
Revised Plan:
**Step 1 (method, eg. RAG)**:
Prompt: [Description of action for chatbot to do Step 1, i.e. do X, Y, Z]

**Step 2 (method, eg. CODE)**:
Prompt: [Description of action for chatbot to do Step 2, i.e. do X, Y, Z]
...
"""
    return template

def rerouteTemplate():
    template = """You are executing a bioinformatics pipeline and must decide which step to execute next. We have already seen the following output, and at this point, you must determine which step to run next based on the chat history.

History: {chathistory}

Routing Decisions: {{user_query}}

**OUTPUT**
Next Step=<step number>
REASONING=<why did you choose that step next>
"""
    return template

def scriptSelectorTemplate():
    template="""You must select which code to run to help a user.

**Available Scripts**
{script_list}

**User Query**
{{user_query}}

**Task**
Based on the user's query, select the best script from the available scripts. Provide the script name and explain why it is the best match. If no script is good, replace script with None

**Response Template**
SCRIPT: <selected script>
REASON: <reasoning why the script fits the user prompt>
"""
    return template

# **OUTPUT**
# If output or output files are created, all output files should be named: {output_path}/<output file>

def pythonPromptTemplate():
    template = """Current conversation:\n{{history}}

**PYTHON SCRIPT**
You must run this python script:
{scriptName}

**PYTHON SCRIPT DOCUMENTATION**:
This is the doc string of this python script:
{scriptDocumentation}


**CALL PYTHON SCRIPTS FROM PYTHON**:
Use the `subprocess` module to call any Python script from another Python script. Here are some examples to call a few common Python scripts:

To call a Python script `example_script.py` which has no arguments:

```
Execute: subprocess.call([sys.executable, '<full path to script/> example_script.py', chatstatus['output-directory']])
```

To call a Python script `example_script.py` with one argument:

```
Execute: subprocess.call([sys.executable, '<full path to script/> example_script.py', chatstatus['output-directory'], 'arg1'])
```

To call a Python script `example_script.py` with two arguments:
```
Execute: subprocess.call([sys.executable, '<full path to script/> example_script.py', chatstatus['output-directory'], 'arg1', 'arg2'])
```

Note that chatstatus['output-directory'] is ALWAYS passed as the first argument.

Query:{{input}}

**INSTRUCTIONS**
1. Given the user query and the documentation, identify each of the arguments found in the user's query that should be passed to the Python script.
2. Using the `subprocess` module, provide the one line of code to execute the desired Python script with the given arguments. Assume the necessary modules (`subprocess` and `sys`) are already imported.
3. The last line of your response should say "Execute: <Python code to execute>"
4. Format the response/output as:
    Arguments: 
    Python Code Explanation: <2 sentences maximum>
    Execute: <your code here>

**IMPORTANT**
The code to execute from your response must be formatted as:
    Execute: subprocess.call([sys.executable, '<path to python script>', '<argument 1>', '<argument 2>', ..., '<argument n>'])
This output should be exactly one line and no longer. Stop the response after this line.
"""
    return template

def pythonPromptTemplateWithFiles():
    template = """Current conversation:\n{{history}}

**PYTHON SCRIPT**
You must run this python script:
{scriptName}

**PYTHON SCRIPT DOCUMENTATION**:
This is the doc string of this python script:
{scriptDocumentation}


**CALL PYTHON SCRIPTS FROM PYTHON**:
Use the `subprocess` module to call any Python script from another Python script. Here are some examples to call a few common Python scripts:

To call a Python script `example_script.py` which has no arguments:

```
Execute: subprocess.run([sys.executable, '<full path to script/> example_script.py', chatstatus['output-directory']], capture_output=True, text=True)
)
```

To call a Python script `example_script.py` with one argument:

```
Execute: subprocess.run([sys.executable, '<full path to script/> example_script.py', chatstatus['output-directory'], 'arg1'], capture_output=True, text=True)
)
```

To call a Python script `example_script.py` with two arguments:
```
Execute: subprocess.run([sys.executable, '<full path to script/> example_script.py', chatstatus['output-directory'], 'arg1', 'arg2'], capture_output=True, text=True)
)
```

Note that chatstatus['output-directory'] is ALWAYS passed as the first argument.

The following files were previously created by BRAD and could be used as input to a function if necessary:
{files}

Query:{{input}}

**INSTRUCTIONS**
1. Given the user query and the documentation, identify each of the arguments found in the user's query that should be passed to the Python script.
2. Using the `subprocess` module, provide the one line of code to execute the desired Python script with the given arguments. Assume the necessary modules (`subprocess` and `sys`) are already imported.
3. The last line of your response should say "Execute: <Python code to execute>"
4. Format the response/output as:
    Arguments: 
    Python Code Explanation: <2 sentences maximum>
    Execute: <your code here>

**IMPORTANT**
The code to execute from your response must be formatted as:
    Execute: subprocess.call([sys.executable, '<path to python script>', '<argument 1>', '<argument 2>', ..., '<argument n>'], capture_output=True, text=True))
This output should be exactly one line and no longer. Stop the response after this line.
"""
    return template

def matlabPromptTemplate():
    template = """Current conversation:\n{{history}}

**MATLAB SCRIPT**
You must run this python script: 
{scriptName}

**MATLAB FUNCTION DOCUMENTATION**:
{scriptDocumentation}

**CALL MATLAB FUNCTIONS FROM PYTHON**:
Use MATLAB® Engine API for Python® to call any MATLAB function on the MATLAB path. Some examples to call a few common matlab functions include:

To call the matlab function myFnc which has no arguments:
```
Execute: eng.myFnc()
```

To use the is prime function which requires an input argument:
```
Execute: tf = eng.isprime(37)
```

To determine the greatest common denominator of two numbers, use the gcd function. Set nargout to return the three output arguments from gcd.
```
Execute: t = eng.gcd(100.0,80.0,nargout=3)
```

Query:{{input}}

**OUTPUT**
If output or output files are created, all output files should be named: {output_path}/<output file>

**INSTRUCTIONS**
1. Given the user query and the documentation, identify each of the arguments found in the users query that should be passed to the matlab function.
2. Using the matlab enging eng, provide the one line of code to execute the desired matlab commands. Assume all functions are added to the path and eng already exist.
3. The last line of your response should say "Execute: <MATLAB codes to execute>"
4. Format the response/output as:
Arguments: 
MATLAB Code Explanation: <2 sentences maximum>
Execute: <your code here>

**IMPORTANT**
The code to execute from your response must be formatted as:
    Execute: eng.<function name>(<arguments>)>
This output should be exactly one line and no longer. Stop the response after this line.
"""
    return template

def summarizeDocumentTemplate():
    template = """**INSTRUCTIONS**
You are an assistant responsible for compressing the important information in a document.
You will be given a users query and a piece of text. Summarize the text with the following aims:
1. remove information that is not complete ideas or unrelated to the topic of the user
2. improve the clarity of the writing and information
If there is no relevant information, say "None"

**USER QUERY**
{user_query}

**TEXT**
{text}

**OUTPUT**
<put summary output here>
"""
    return template

def geneDatabaseCallerTemplate():
    template = """Current conversation:\n{{history}}
    
GENEONTOLOGY: The Gene Ontology (GO) is an initiative to unify the representation of gene and gene product attributes across all species via the aims: 1) maintain and develop its controlled vocabulary of gene and gene product attributes; 2) annotate genes and gene products, and assimilate and disseminate annotation data; and 3) provide tools for easy access to all aspects of the data provided by the project, and to enable functional interpretation of experimental data using the GO.
    
ENRICHR: is a tool used to lookup sets of genes and their functional association. ENRICHR has access to many gene-set libraries including Allen_Brain_Atlas_up, ENCODE_Histone_Modifications_2015, Enrichr_Libraries_Most_Popular_Genes, FANTOM6_lncRNA_KD_DEGs, GO_Biological_Process_2023, GTEx, Human_Gene_Atlas, KEGG, REACTOME, Transcription_Factor_PPIs, WikiPathways and many others databases.
    
Available Tables:
{tables}

Query:{{input}}

**INSTRUCTIONS**
1. From the query, decide if GENEONTOLOGY or ENRICHR should be used
2. Identify genes in the users input that should be searched. propose genes with similar names, correct the users spelling, or make small modifications to the list, but do not propose genes that are not found in the humans input.
3. Indicate if genes need to be loaded from a file: True/False
4. Use the following output format with no additional information or explination.

database: <ENRICHR or GENEONTOLOGY>
genes: <None or Gene 1, Gene 2, Gene 3, ...>
load: <True/False>
"""
    return template

def fileChooserTemplate():
    template = """**INSTRUCTIONS**
You are a digital assistant responsible for determining which file we should load based on a users prompt. You many choose either the files listed under AVAILABLE FILES or found in the USER QUERY. Based upon the user query, specify which if any fields should be loaded from the table. Rather than specifying a table name, the user might specify ot load data from a particular step. In this case, files are saved with the prefix `S<number>-<File Name>` to denote which step to load them from.

**EXAMPLES**

Here the files are selected by name:
```
Available Files:
S1-<AAA>.tsv
S2-<BBB>.csv

User Query:
Use the <XXX> output from <BBB> for the this phase of anlaysis.

Output:
File: S2-<BBB>.csv
Fields: <XXX>
```

Here the files are selected by Step:
```
Available Files:
S1-<AAA>.tsv
S2-<BBB>.csv

User Query:
Use the <YYY> identified in Step 1 as input to the codes.

Output:
File: S1-<AAA>.tsv
Fields: <YYY>
```

**USER QUERY**
{{user_query}}

You must select one of the following available files based on that users query:
{files}

**OUTPUT**
File: <which file you selected>
Fields: <which fields in the file are important>
"""
    return template

def fieldChooserTemplate():
    template = """You are a digital assistant responsible for identifying the columns of a spreedsheet based on the users query. Below will be a list of possible columns, and please select the column the user is interested. Use the exact output format and provide no extra informations.

**AVAILABLE COLUMNS**
{columns}

**USER QUERY**
{{user_query}}

Field=<selected field>
"""
    return template

def setReportTitleTemplate():
    template = """You are an assistant responsible for writing a report in response to a users query. Based on the users query, please provide a clear, concise title for the report to answer this question. Keep the title relatively short (i.e. less than 7 words).

**User Query**
{user_query}

Please use this exact title output formatting:
title = <put title here>
"""
    return template

def summarizeAnalysisPipelineTemplate():
    template = """You are an assistant responsible for writing a report of a bioinformatics analysis pipeline, and in this section, you need to write a paragraph summarizing the different steps in the analysis pipeline. The pipeline is run by a bioinformatics chatbot, which has the ability to use RAG (Retrieval Augmented Generation), SCRAPE (web scrape for new literature), CODE (run and execute different software scripts), and DATABASE (look up information on Enrichr and Gene Ontology databases). Each stage of the pipeline is encoded as a prompt to the user. From these prompts and the description below, write a paragraph summarizing the stages of the pipeline.

The report should be concise, somewhere between 3-5 sentences. Also, it should be written in the past tense to signifcy that the steps in the pipeline have already been completed. Please do exclude technical details such as the programming languages, specific database names, input or output file names, directory structures, and anything specific to the implementation of the pipeline.

**Pipeline Steps**
{pipeline}

Please use this exact title output formatting:
paragraph = <put paragraph here>
"""
    return template

def summarizeDatabaseCallerTemplate():
    template = """You are an assistant writing a report of a bioinformatics analysis pipeline, and in this section, you need to write a sentence or two about how a gene database was searched. You summary is based on a conversation between a human and a bioinformatics chatbot. Here is their conversation:

Human: {{input}}
Chatbot: {output}

Additional information:
A table with {numrows} was downloaded. The table will be below your summary in the report.

Please use this exact title output formatting:
summary = <put paragraph here>
"""
    return template

def summarizeRAGTemplate():
    template = """You are an assistant writing a report on the outputs of a bioinformatics pipeline. For this section of the report, I need you to explain the following text clearly and convert it to an acceptable latex format. Here is the text to reformat:

Current Text: {output}

Format your output as:
Latex Version=<put paragraphs here>
"""
    return template

def historyChatTemplate():
    template = """Current conversation: {history}\n\n\nNew Input: \n{input}"""
    return template