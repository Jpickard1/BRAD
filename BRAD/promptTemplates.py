def plannerTemplate():
    template = """
You are planning a bioinformatics analysis pipeline to address a user's query. Your task is to outline a multi-step workflow using the available methods listed below. Each method should be used appropriately to ensure a thorough analysis.
For each step in the pipeline, explain the goal of the process as well as provide a prompt to the chatbot that will execute that step. You can use one method per step and the pipeline can be between 3-12 steps.

Available Methods:
1. **Retrieval Augmented Generation**(RAG): Look up literature and documents from a text database.
2. **Web Search for New Literature**(SCRAPE): Search platforms like arXiv, bioRxiv, and PubMed for the latest research.
3. **Bioinformatics Databases**(DATABASE): Utilize databases such as Gene Ontology and Enrichr to perform gene set enrichment analyses.
4. **Run Codes**(CODE): Execute bioinformatics pipelines. Utilize prebuilt pipelines or develop new ones as needed.
5. **Write a Report**(WRITE): Synthesize and summarize information. This can include summarizing database searches, code pipeline results, or creating a final report to encapsulate the entire analysis.

Based on the user's query, create a detailed plan outlining the steps of the analysis. Ensure that each step is clearly defined and makes use of the appropriate method(s) listed above. Clarify which steps are dependent on one another, such as if information from a CODE step is used by DATABASE step, if information from a RAG step is used by a SCRAPE step, or if all previous steps are needed in a WRITE step, or any other dependencies.

Current conversation: {history}\n

User Query: {input}

Plan:
**Step 1 (method, eg. RAG)**:
Prompt: [Description of action for chatbot to do Step 1, i.e. do X, Y, Z]

**Step 2 (method, eg. CODE)**:
Prompt: [Description of action for chatbot to do Step 2, i.e. do X, Y, Z]

**Step 3 (method, eg. DATABASE)**:
Prompt: [Description of action for chatbot to do Step 3, i.e. do X, Y, Z]
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

def historyChatTemplate():
    template = """Current conversation: {history}\n\n\nNew Input: \n{input}"""
    return template