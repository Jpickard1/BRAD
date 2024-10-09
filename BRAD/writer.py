"""
Module: writer

This module is responsible for compiling chat reports, histories, and logs from various interactions with the 
large language model (LLM) and generating a comprehensive PDF document. The module streamlines the process of 
documenting interactions, making it easier for users to review and analyze conversations.

Usage:
------
This module is intended to be used in scenarios where there is a need to document interactions with the LLM. 
By converting chat histories and logs into a well-organized PDF file, users can maintain a clear record of 
conversations, facilitating further analysis and reflection on the interactions.
"""
import pandas as pd
import os
import re
import json
from tabulate import tabulate
from datetime import datetime

from langchain import PromptTemplate, LLMChain
from langchain.output_parsers.json import SimpleJsonOutputParser
from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
# from langchain_core.pydantic_v1 import BaseModel, Field

from pydantic import BaseModel, Field


from BRAD import log
from BRAD import utils
from BRAD.promptTemplates import setReportTitleTemplate, summarizeAnalysisPipelineTemplate, summarizeDatabaseCallerTemplate, summarizeRAGTemplate
from BRAD.pythonCaller import find_py_files, get_py_description, read_python_docstrings, pythonPromptTemplate, extract_python_code, execute_python_code

def chatReport(state):
    """
    Generates a report of the chat conversation based on the logged interactions 
    and writes it to a PDF file using a LaTeX template.

    Args:
        state (dict): A dictionary containing the language model, user prompt, 
                           output directory, and other relevant information for generating 
                           the chat report.

    Returns:
        dict: The updated state containing the path to the generated PDF report and 
              any updates made during the report generation process.

    Example
    -------
    >>> state = {
    ...     'llm': llm_instance,
    ...     'prompt': "Generate chat report",
    ...     'output-directory': '/path/to/output',
    ... }
    >>> updated_state = chatReport(state)
    """
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: July 2, 2024
    log.debugLog("Chat Report Called", state=state)
    query    = state['prompt']
    llm      = state['llm']

    # TODO: this should be path dependent in the configurations
    # Select the latex template. The specific latex template is hardcoded for now, but a latex-template-path
    # indicator has been added to the config.json file and we can do a selection process similar to how scripts
    # are selected to run
    latexTemplate = getLatexTemplate('/home/jpic/latex-test/templates/dmdBiomarkerEnrichment.tex')
    
    # Read the full chat history
    chatlog = json.load(open(os.path.join(state['output-directory'], 'log.json')))
    chathistory = getChatInputOutputs(state, chatlog)
    log.debugLog(chathistory, state=state)

    # Get fields that are always set
    title = getReportTitle(chathistory, state=state)
    log.debugLog("title= " + str(title), state = state)
    date  = getReportDate()
    log.debugLog("date = " + str(date) , state = state)

    # Fill in the latex template
    latexTemplate = latexTemplate.replace("BRAD-TITLE", title)
    latexTemplate = latexTemplate.replace("BRAD-DATE",  date)

    # Loop over possible keys we put in the latex templates:
    partsOfReport = ['SUMMARY', 'BODY']
    for part in partsOfReport:
        key = 'BRAD-' + part
        if key in latexTemplate:
            if key == 'BRAD-SUMMARY':
                section = getReportSummary(chathistory, state=state)
            if key == 'BRAD-BODY':
                section = getReportBody(state, chatlog)
            latexTemplate = latexTemplate.replace(key, section)

    # After filling in the report template write it to a pdf file
    state, report_file = reportToPdf(state, latexTemplate)
    state = utils.compile_latex_to_pdf(state, report_file)
    
    return state

def getReportDate():
    """
    Returns the current date formatted as a string in the format: "Day of the week, Month Day, Year".

    Returns:
        str: Formatted date string.
    """
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: July 2, 2024
    today = datetime.now()
    formatted_date = today.strftime("%A, %B %d, %Y")
    return formatted_date

def getReportTitle(chathistory, state):
    """
    Uses a language model to generate a descriptive title for a report summarizing the work done in the chatlog.

    Args:
        chathistory (dict): A dictionary containing the chat history with user inputs and model outputs.
        state (dict): A dictionary containing the current chat status, including the language model instance 
                           and user prompt.

    Returns:
        str: The generated report title based on the chatlog and user prompt.
    """
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: July 2, 2024
    llm      = state['llm']
    template = """Given the following chatlog, generate a descriptive title for a report summarizing the work done. Please keep the title concise. This report is in reposnse to the following user request:
{userinput}

**Chatlog Overview:**
{chatlog}
"""
    template = template.format(userinput=state['prompt'], chatlog=chathistory)
    log.debugLog(f"{template=}", state=state)
    prompt = PromptTemplate(template=template, input_variables=["chatlog"])
    json_prompt = PromptTemplate.from_template(
        "Return a JSON object with an `title` and `abstract` keys in response to the following: {question}"
    )
    json_parser = SimpleJsonOutputParser()
    json_chain = json_prompt | llm | json_parser
    response = json_chain.invoke({"question": template})
    report_title = response['title']
#    chain = prompt | llm
#    log.debugLog('Calling getReportTitle', state=state)
#    response = chain.invoke(chathistory)
#    log.debugLog(response, state=state)
#    report_title = response.content.split('=')[1]
    state['process']['steps'].append(log.llmCallLog(llm          = llm,
                                                         prompt       = template,
                                                         input        = chathistory,
                                                         output       = response,
                                                         parsedOutput = {
                                                             'report title' : report_title,
                                                         },
                                                         purpose      = 'generate a title for a pdf (note: using pydantic parser)'
                                                        )
                                         )
    log.debugLog(report_title, state=state)
    return report_title

def getReportSummary(chathistory, state):
    """
    Uses a language model to generate a summary section for a report based on the chat session with BRAD.

    Args:
        chathistory (dict): A dictionary containing the chat history with user inputs and model outputs.
        state (dict): A dictionary containing the current chat status, including the language model instance 
                           and user prompt.

    Returns:
        str: The generated summary section for the report based on the chatlog and user prompt.
    """
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: July 2, 2024
    llm      = state['llm']
    template = """You are in charge of writing a report to summarize a chatbot session that performed a series of steps in a bioinformatics workflow. Given the following chatlog, write a section of to summarize this. Your summary section should be at least three sentences long (minimum) and no longer than twelve sentences.
    
This report is in reposnse to the following user request:
{userinput}

**Chatlog Overview:**
{{chatlog}}

Format your output exactly as follows. Do not include additional text/characters.
**Output:**
Summary=<put the title title here>
"""
    template = template.format(userinput=state['prompt'])
    print(template)
    print(f"{chathistory=}")
    prompt = PromptTemplate(template=template, input_variables=["chatlog"])
    chain = prompt | llm
    log.debugLog('Calling getReportSummary', state=state)
    response = chain.invoke(chathistory)
    log.debugLog(response, state=state)
    report_summary = response.content.split('=')[1]
    state['process']['steps'].append(log.llmCallLog(llm          = llm,
                                                         prompt       = prompt,
                                                         input        = chathistory,
                                                         output       = response,
                                                         parsedOutput = {
                                                             'report summary' : report_summary,
                                                         },
                                                         purpose      = 'generate a summayr of the chat history'
                                                        )
                                         )
    log.debugLog(report_summary, state=state)
    return report_summary


def getChatInputOutputs(state, chatlog):
    """
    Creates a single string to summarize the chat history between a user and BRAD.

    Args:
        state (dict): A dictionary containing the current chat status, not directly used in this function.
        chatlog (dict): A dictionary representing the chat history with keys as indices and values as dictionaries
                        containing 'prompt' and 'output' keys.

    Returns:
        str: A formatted string summarizing the chat history with inputs from the user and outputs from BRAD.
    """
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: July 2, 2024
    chathistory = "**Chat History**\n\n"
    for i in chatlog.keys():
        chathistory += "Human: " + chatlog[i]['prompt'] + '\n\n'
        chathistory += "BRAD: " + chatlog[i]['output'] + '\n\n\n'
    return chathistory



def getLatexTemplate(path2latexTemplate):
    """
    Reads the LaTeX template file and returns it as a string.
    
    Args:
    path2latexTemplate (str): The path to the LaTeX template file.
    
    Returns:
    str: The contents of the LaTeX template file as a string.
    """
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: July 2, 2024
    try:
        with open(path2latexTemplate, 'r') as file:
            latex_template = file.read()
        return latex_template
    except FileNotFoundError:
        return f"Error: The file at {path2latexTemplate} was not found."
    except IOError:
        return f"Error: An error occurred while reading the file at {path2latexTemplate}."


def summarizeSteps(state):
    """
    .. warning: we will remove this soon
    
    This function should write a report about what happened in a pipeline
    """
    query    = state['prompt']
    llm      = state['llm']              # get the llm

    chatlog = json.load(open(os.path.join(state['output-directory'], 'log.json')))

    # Write the report in this outline
    #  - Title
    title = setTitle(state, chatlog)
    
    #  - Driving prompt
    prompt = getPrompt(state, chatlog)

    #  - Summary of process
    processSummary = getProcessSummary(state, chatlog)
    
    #  - Summary of outputs
    reportBody = getReportBody(state, chatlog)
    
    #  - References
    references = getReferences(state, chatlog)

    # Fill in the report
    report = getFirstLatexReportOutline(title=title)

    report = report.format(prompt=prompt,
                           processSummary=processSummary,
                           reportBody=reportBody,
                           references=references)

    # Report to pdf
    state, report_file = reportToPdf(state, report)
    state = utils.compile_latex_to_pdf(state, report_file)
    
    return state #, report

def reportToPdf(state, report):
    """
    Writes the formatted report to a PDF file.

    Args:
        state (dict): A dictionary containing the current chat status, used to track process steps and file paths.
        report (str): The formatted report content in LaTeX format.

    Returns:
        tuple: A tuple containing the updated chat status dictionary and the filename of the generated PDF report.
    """
    report = ensureLatexFormatting(report)
    report_file = 'REPORT.tex'
    state = utils.save(state, report, report_file)
    report_file = state['process']['steps'][-1]['new file']
    return state, report_file

def ensureLatexFormatting(report):
    """
    .. warning: we need to make this function work better
    
    This function ensures appropriate LaTeX formatting.
    
    For instance:
        - any "_" not in a math environment should be immediately proceeded by "\" to make them all "\_". 
          If they are already immediately proceeded by "\", then no change is required.
        - all math should be in an appropriate math environment including $ for intext characters or 
          \begin{equation} <math here> \end{equation} for full lines of math.
        - any file names that have .csv, .tsv, .pkl, .h5ad, .txt, .py, .cpp should be in the \texttt{<file name>} environment.
    """
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: July 2, 2024

    # Function to escape underscores not already escaped
    def escape_underscores(text):
        return re.sub(r'(?<!\\)_', r'\\_', text)

    # Process the report to handle different parts
    parts = re.split(r'(\$.*?\$)', report, flags=re.DOTALL)
    
    processed_parts = []
    for part in parts:
        if part.startswith('$') and part.endswith('$'):
            # Inline math, do not change underscores
            processed_parts.append(part)
        else:
            # Regular text, escape underscores and wrap math expressions in $
            part = escape_underscores(part)
            part = re.sub(r'(\b[a-zA-Z]\w*\s*=\s*[^.]+\b)', r'$\1$', part)  # Heuristic to detect simple equations
            processed_parts.append(part)
    
    report = ''.join(processed_parts)

    # Ensure file names with specific extensions are in \texttt{}
    report = re.sub(r'(\S+\.(csv|tsv|pkl|h5ad|txt|py|cpp))', r'\\texttt{\1}', report)
    
    return report

    

def setTitle(state, chatlog):
    """
    .. warning: we will remove this soon
    
    This function sets the title of the report based on the initial user query
    """

    # Get the pro
    for promptNumber in chatlog.keys():
        log.debugLog(promptNumber, state=state) 
        if promptNumber == 'llm':
            continue
        if chatlog[promptNumber]['process']['module'] == 'PLANNER':
            prompt = chatlog[promptNumber]['prompt']
            break

    # Set the title
    template = setReportTitleTemplate()
    PROMPT = PromptTemplate(input_variables=["user_query"], template=template)
    chain = PROMPT | state['llm']
    res = chain.invoke(prompt)
    log.debugLog('Title Setter Output: \n\n' + str(res.content), state)
    title   = res.content.split('=')[1].strip()
    log.debugLog('Title=' + str(title), state)
    return title

def getPrompt(state, chatlog):
    """
    .. warning: we will remove this soon
    
    This function finds the original prompt used to generate this report
    """
    for promptNumber in chatlog.keys():
        if promptNumber == 'llm':
            continue
        if chatlog[promptNumber]['process']['module'] == 'PLANNER':
            prompt = chatlog[promptNumber]['prompt']
            return prompt
    return ""

def getProcessSummary(state, chatlog):
    """Summarizes each stage of the process"""
    for promptNumber in chatlog.keys():
        if promptNumber == 'llm':
            continue
        if chatlog[promptNumber]['process']['module'] == 'PLANNER':
            fullPlanner = str(chatlog[promptNumber]['status']['queue'])
    log.debugLog(fullPlanner, state=state) 
    template = summarizeAnalysisPipelineTemplate()
    PROMPT = PromptTemplate(input_variables=["pipeline"], template=template)
    chain = PROMPT | state['llm']
    res = chain.invoke(fullPlanner)
    log.debugLog('Process Summary Output: \n\n' + str(res.content), state)
    processSummary   = res.content.split('=')[1].strip()
    log.debugLog('Process Summary=' + str(processSummary), state)
    return processSummary

def getReportBody(state, chatlog):
    """Summarizes individual steps"""
    reportBody = ""
    for promptNumber in chatlog.keys():
        if promptNumber == 'llm':
            continue
        # Add to the report body based on the module
        module = chatlog[promptNumber]['process']['module']
        if module == 'RAG':
            reportBody += ragReporter(chatlog[promptNumber], state)
        elif module == 'CODE':
            reportBody += codeReporter(chatlog[promptNumber], state=state)
        elif module == 'DATABASE':
            reportBody += databaseReporter(chatlog[promptNumber], state)
        # Add to the report body based on the elements
        # reportBody += addFigures(chatlog[promptNumber])
        # reportBody += databaseReporter(chatlog[promptNumber])
    return reportBody

def codeReporter(chatlogStep, currentreport='', state=None):
    """Summarizes running of some code"""
    scriptRun = chatlogStep['process']['steps'][0]['parsedOutput']['scriptName']
    docstring = read_python_docstrings(scriptRun)
    code      = chatlogStep['process']['steps'][1]['parsedOutput']['code']
    output    = chatlogStep['output']

    template = """You are responsible for compiling a report of a bioinformatics pipeline. In the current section you are writing, you must summarize the output of running a piece of code in the pipeline. You will be provided with the current report, the doc strings of the function that was run, the command used to run the script, and the output of the script. Then, please provide the next step of paragraphs(s), in appropriate latex format, to summarize this stage of the pipeline.

**Current Report**
{currentreport}

**Documentation of the function that was run**
{docstring}

**Executed Code**
{code}

**Code Output**
{codeoutput}

**Aditional Instructions**
{{userprompt}}

Please format your output exactly as follows.
**Output**
Summary: <put summary here>
"""
    template = template.format(currentreport=currentreport,
                               docstring=docstring,
                               code=code,
                               codeoutput=output,
    )
    print(template)
    PROMPT = PromptTemplate(input_variables=["userprompt"], template=template)
    chain = PROMPT | state['llm']
    response = chain.invoke(chatlogStep['prompt'])
    print(response)
    return response.content # response.content.split('=')[1]

def ragReporter(chatlogStep, state):
    """Summarize the output of the RAG stage"""
    output = chatlogStep['output']
    template = summarizeRAGTemplate()
    log.debugLog(template, display=True)
    # template = template.format(output=output)
    PROMPT = PromptTemplate(input_variables=["output"], template=template)
    chain = PROMPT | state['llm']
    res = chain.invoke(output)
    processSummary   = res.content.split('=')[1].strip().replace('_', '\_')
    return processSummary
    
def databaseReporter(chatlogStep, state):
    """Summarize the output of the DATABASE lookup stage"""
    databaseTex = ""
    prompt = chatlogStep['prompt']
    output = chatlogStep['output']

    for step in chatlogStep['process']['steps']:
        # Check if the step saved a figure
        if 'func' in step.keys() and step['func'] == 'utils.save':
            filename = step['new file']
            state = log.userOutput(filename, state=state) # 
            df = utils.load_file_to_dataframe(filename)
    if df is not None:
        numrows = df.shape[0]
    else:
        numrows = 0
    
    template = summarizeDatabaseCallerTemplate()
    log.debugLog(template, display=True)
    template = template.format(output=output, numrows=numrows)
    PROMPT = PromptTemplate(input_variables=["input"], template=template)
    chain = PROMPT | state['llm']
    res = chain.invoke(prompt)
    processSummary   = res.content.split('=')[1].strip().replace('_', '\_')
    databaseTex += processSummary
    databaseTex += addFigures(chatlogStep, state)
    databaseTex += addTables(chatlogStep, state)
    
    return databaseTex

def addFigures(chatlogStep, state):
    """Add figures to the latex output file"""
    figureTex = ""
    for step in chatlogStep['process']['steps']:
        # Check if the step saved a figure
        if 'func' in step.keys() and step['func'] == 'utils.savefig':
            figureCode = """\n\n\\begin{figure}
    \\centering
    \\includegraphics[width=\\textwidth]{""" + step['new file'] + """}
\\end{figure}\n\n"""
            figureTex += figureCode
    return figureTex
# \\caption{\\textbf{""" +step['new file'] + """}}
# \\label{fig:enter-label}

def addTables(chatlogStep, state):
    """Add tables to the latex output file"""
    tableTex = ""
    df = None
    # Choose table
    for step in chatlogStep['process']['steps']:
        # Check if the step saved a figure
        if 'func' in step.keys() and step['func'] == 'utils.save':
            filename = step['new file']
            state = log.userOutput(filename, state=state) 
            df = utils.load_file_to_dataframe(filename)
            break
    if df is None:
        return ""
    log.debugLog(type(df), state=state) 
    log.debugLog(df.head(3), state=state) 
    # df = pd.read_csv(filename)
    # Build latex
    tableTex += ('\n\n' + dataframe_to_latex(df) + '\n\n')
    return tableTex

def getReferences(state, chatlog):
    """Finds all documents used in the chat"""
    referenceList = []
    for promptNumber in chatlog.keys():
        if promptNumber == 'llm':
            continue
        if chatlog[promptNumber]['process']['module'] == 'RAG':
            sources = chatlog[promptNumber]['process']['sources']
            for source in sources:
                referenceList.append(source)
    references = '\\begin{itemize}\n'
    for ref in referenceList:
        references += '\\item ' + ref + '\n'
    references += '\\end{itemize}'
    return references

def dataframe_to_latex(df):
    """Converts a dataframe to latex"""
    df = df.head(20)
    for col in df.columns:
        if len(df[col].unique()) == 1:
            df.drop(col,inplace=True,axis=1)
    max_length = 250
    
    # Identify columns to drop
    columns_to_drop = [col for col in df.columns if df[col].astype(str).apply(len).max() > max_length]
    
    # Drop identified columns
    df = df.drop(columns=columns_to_drop)
    
    df['path_name'] = df['path_name'].apply(lambda x: x[:35] + "...")

    # Generate LaTeX table
    latex_table = tabulate(df, tablefmt='latex', showindex=False, headers='keys')


    # Add booktabs and other LaTeX formatting
    latex_table = latex_table.replace("\\\\ ", "\\").replace("toprule", "toprule\nrank & path-name & p-val & z-score & combined-score & adj-p-val \\\\ \\midrule").replace("midrule", "midrule\n").replace("bottomrule", "bottomrule\n")

    # Add table environment
    latex_table = "\\begin{table}[h]\n\\centering\n" + latex_table + "\n\\end{table}"

    return latex_table

def getFirstLatexReportOutline(title='BRAD Output'):
    """A hardcoded latex template for generating a pdf report"""
    report = """\\documentclass{{article}}
\\usepackage[margin=1in]{{geometry}}
\\usepackage{{graphicx}}
\\usepackage{{url}}
\\begin{{document}}

\\title{{""" + title + """}}
\\author{{BRAD}}
\\maketitle

\\section*{{User Prompt}}
{prompt}

\\section*{{Summary of Process}}
{processSummary}

\\section*{{Summary of Outputs}}
{reportBody}

\\section*{{References}}
{references}

\\end{{document}}
"""
    return report