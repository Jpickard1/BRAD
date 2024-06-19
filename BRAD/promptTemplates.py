# Based on the user's query, create a detailed plan outlining the steps of the analysis. Ensure that each step is clearly defined and makes use of an appropriate method. Clarify which steps are dependent on one another, such as if information from a CODE step is used by DATABASE step, if information from a RAG step is used by a SCRAPE step, or if all previous steps are needed in a WRITE step, or any other dependencies.
def plannerTemplate():
    template = """**INSTRUCTIONS:**
You are planning a bioinformatics analysis pipeline to address a user's query. Your task is to outline a multi-step workflow using the available methods listed below. Each method should be used appropriately to ensure a thorough analysis. For each step in the pipeline, explain the goal of the process as well as provide a prompt to the chatbot that will execute that step. If information is passed between stages, such as from literature or code to databases, indicate the dependencies of steps in the prompt.

**Available Methods:**
1. **Retrieval Augmented Generation**(RAG): Look up literature and documents from a text database.
2. **Web Search for New Literature**(SCRAPE): Search platforms like arXiv, bioRxiv, and PubMed for the latest research.
3. **Bioinformatics Databases**(DATABASE): Utilize databases such as Gene Ontology and Enrichr to perform gene set enrichment analyses.
4. **Run Codes**(CODE): Execute bioinformatics pipelines. Utilize prebuilt pipelines or develop new ones as needed.
5. **Write a Report**(WRITE): Synthesize and summarize information. This can include summarizing database searches, code pipeline results, or creating a final report to encapsulate the entire analysis.

Current conversation: {history}\n

User Query: {input}

Plan:
**Step 1 (method, eg. RAG)**:
Prompt: [Description of action for chatbot to do Step 1, i.e. do X, Y, Z]

**Step 2 (method, eg. CODE)**:
Prompt: [Description of action for chatbot to do Step 2, i.e. do A, B, C. Use information from Step X]

**Step 3 (method, eg. DATABASE)**:
Prompt: [Description of action for chatbot to do Step 3, i.e. do D, E, F. Use information from Step Y]
...
"""
    return template

def plannerEditingTemplate():
    template = """Based on the most recently proposed Plan and the user's new requirements and requested changes,
create a revised plan.

Current conversation: {history}\n

User's requested changes: {input}

Plan:
**Step 1 (method, eg. RAG)**:
Prompt: [Description of action for chatbot to do Step 1, i.e. do X, Y, Z]

**Step 2 (method, eg. CODE)**:
Prompt: [Description of action for chatbot to do Step 2, i.e. do X, Y, Z]
...
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
Execute: subprocess.call([sys.executable, '<full path to script/> example_script.py'])
```

To call a Python script `example_script.py` with one argument:

```
Execute: subprocess.call([sys.executable, '<full path to script/> example_script.py', 'arg1'])
```

To call a Python script `example_script.py` with two arguments:
```
Execute: subprocess.call([sys.executable, '<full path to script/> example_script.py', 'arg1', 'arg2'])
```
Query:{{input}}

**OUTPUT**
If output or output files are created, all output files should be named: {output_path}/<output file>

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

database: <ENRICHR or GENEONTOLOGY>
genes: <List of genes separated by commas in query or None if code is required>
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
S1-PYTHON.tsv
S2-ENRICHR.csv

User Query:
Use the pathways output from enricher for the this phase of anlaysis.

Output:
File: S2-ENRICHR.csv
Fields: pathways
```

Here the files are selected by Step:
```
Available Files:
S1-PYTHON.tsv
S2-ENRICHR.csv

User Query:
Use the Genes identified in Step 1 as input to the codes.

Output:
File: S1-PYTHON.tsv
Fields: Genes
```

**AVAILABLE FILES**
{files}

**USER QUERY**
{{user_query}}

**OUTPUT**
File: <which file you selected>
Fields: <which fields in the file are important>
"""
    return template

def historyChatTemplate():
    template = """Current conversation: {history}\n\n\nNew Input: \n{input}"""
    return template