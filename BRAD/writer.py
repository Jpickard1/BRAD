import pandas as pd
import os
import re
import json
from tabulate import tabulate
from datetime import datetime

from langchain import PromptTemplate, LLMChain

from BRAD import log
from BRAD import utils
from BRAD.promptTemplates import setReportTitleTemplate, summarizeAnalysisPipelineTemplate, summarizeDatabaseCallerTemplate, summarizeRAGTemplate
from BRAD.pythonCaller import find_py_files, get_py_description, read_python_docstrings, pythonPromptTemplate, extract_python_code, execute_python_code

def chatReport(chatstatus):
    """This function should write a report of the chat"""
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: July 2, 2024
    log.debugLog("Chat Report Called", chatstatus=chatstatus)
    query    = chatstatus['prompt']
    llm      = chatstatus['llm']

    # Select the latex template. The specific latex template is hardcoded for now, but a latex-template-path
    # indicator has been added to the config.json file and we can do a selection process similar to how scripts
    # are selected to run
    latexTemplate = getLatexTemplate('/home/jpic/latex-test/templates/dmdBiomarkerEnrichment.tex')
    
    # Read the full chat history
    chatlog = json.load(open(os.path.join(chatstatus['output-directory'], 'log.json')))
    chathistory = getChatInputOutputs(chatstatus, chatlog)
    log.debugLog(chathistory, chatstatus=chatstatus)

    # Get fields that are always set
    title = getReportTitle(chathistory, chatstatus=chatstatus)
    log.debugLog("title= " + str(title), chatstatus = chatstatus)
    date  = getReportDate()
    log.debugLog("date = " + str(date) , chatstatus = chatstatus)

    # Fill in the latex template
    latexTemplate = latexTemplate.replace("BRAD-TITLE", title)
    latexTemplate = latexTemplate.replace("BRAD-DATE",  date)

    # Loop over possible keys we put in the latex templates:
    partsOfReport = ['SUMMARY', 'BODY']
    for part in partsOfReport:
        key = 'BRAD-' + part
        if key in latexTemplate:
            if key == 'BRAD-SUMMARY':
                section = getReportSummary(chathistory, chatstatus=chatstatus)
            if key == 'BRAD-BODY':
                section = getReportBody(chatstatus, chatlog)
            latexTemplate = latexTemplate.replace(key, section)

    # After filling in the report template write it to a pdf file
    chatstatus, report_file = reportToPdf(chatstatus, latexTemplate)
    chatstatus = utils.compile_latex_to_pdf(chatstatus, report_file)
    
    return chatstatus

def getReportDate():
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: July 2, 2024
    today = datetime.now()
    formatted_date = today.strftime("%A, %B %d, %Y")
    return formatted_date

def getReportTitle(chathistory, chatstatus):
    """This function uses an llm to determine the title of a report that summarizes the work done by in the chatlog"""
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: July 2, 2024
    llm      = chatstatus['llm']
    template = """Given the following chatlog, generate a descriptive title for a report summarizing the work done. Please keep the title concise. This report is in reposnse to the following user request:
{userinput}

**Chatlog Overview:**
{{chatlog}}

Format your output exactly as follows.
**Output:**
Title=<put the title title here>
"""
    template = template.format(userinput=chatstatus['prompt'])
    print(template)
    prompt = PromptTemplate(template=template, input_variables=["chatlog"])
    chain = prompt | llm
    log.debugLog('Calling getReportTitle', chatstatus=chatstatus)
    response = chain.invoke(chathistory)
    log.debugLog(response, chatstatus=chatstatus)
    report_title = response.content.split('=')[1]
    log.debugLog(report_title, chatstatus=chatstatus)
    return report_title

def getReportSummary(chathistory, chatstatus):
    """This function uses an llm summarize a chat session with BRAD based on the log"""
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: July 2, 2024
    llm      = chatstatus['llm']
    template = """You are in charge of writing a report to summarize a chatbot session that performed a series of steps in a bioinformatics workflow. Given the following chatlog, write a section of to summarize this. Please use appropriate latex formatting, but do not worry about section titles or headings.
    
This report is in reposnse to the following user request:
{userinput}

**Chatlog Overview:**
{{chatlog}}

Format your output exactly as follows.
**Output:**
Summary=<put the title title here>
"""
    template = template.format(userinput=chatstatus['prompt'])
    print(template)
    prompt = PromptTemplate(template=template, input_variables=["chatlog"])
    chain = prompt | llm
    log.debugLog('Calling getReportSummary', chatstatus=chatstatus)
    response = chain.invoke(chathistory)
    log.debugLog(response, chatstatus=chatstatus)
    report_summary = response.content.split('=')[1]
    log.debugLog(report_summary, chatstatus=chatstatus)
    return report_summary


def getChatInputOutputs(chatstatus, chatlog):
    """Creates a single string to summarize the chat history"""
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


def summarizeSteps(chatstatus):
    """This function should write a report about what happened in a pipeline"""
    query    = chatstatus['prompt']
    llm      = chatstatus['llm']              # get the llm

    chatlog = json.load(open(os.path.join(chatstatus['output-directory'], 'log.json')))

    # Write the report in this outline
    #  - Title
    title = setTitle(chatstatus, chatlog)
    
    #  - Driving prompt
    prompt = getPrompt(chatstatus, chatlog)

    #  - Summary of process
    processSummary = getProcessSummary(chatstatus, chatlog)
    
    #  - Summary of outputs
    reportBody = getReportBody(chatstatus, chatlog)
    
    #  - References
    references = getReferences(chatstatus, chatlog)

    # Fill in the report
    report = getFirstLatexReportOutline(title=title)

    report = report.format(prompt=prompt,
                           processSummary=processSummary,
                           reportBody=reportBody,
                           references=references)

    # Report to pdf
    chatstatus, report_file = reportToPdf(chatstatus, report)
    chatstatus = utils.compile_latex_to_pdf(chatstatus, report_file)
    
    return chatstatus #, report

def reportToPdf(chatstatus, report):
    """This function writes the report to a pdf file"""
    report = ensureLatexFormatting(report)
    report_file = 'REPORT.tex'
    chatstatus = utils.save(chatstatus, report, report_file)
    report_file = chatstatus['process']['steps'][-1]['new file']
    return chatstatus, report_file

def ensureLatexFormatting(report):
    """
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

    

def setTitle(chatstatus, chatlog):
    """This function sets the title of the report based on the initial user query"""

    # Get the pro
    for promptNumber in chatlog.keys():
        log.debugLog(promptNumber, chatstatus=chatstatus) 
        if promptNumber == 'llm':
            continue
        if chatlog[promptNumber]['process']['module'] == 'PLANNER':
            prompt = chatlog[promptNumber]['prompt']
            break

    # Set the title
    template = setReportTitleTemplate()
    PROMPT = PromptTemplate(input_variables=["user_query"], template=template)
    chain = PROMPT | chatstatus['llm']
    res = chain.invoke(prompt)
    log.debugLog('Title Setter Output: \n\n' + str(res.content), chatstatus)
    title   = res.content.split('=')[1].strip()
    log.debugLog('Title=' + str(title), chatstatus)
    return title

def getPrompt(chatstatus, chatlog):
    """This function finds the original prompt used to generate this report"""
    for promptNumber in chatlog.keys():
        if promptNumber == 'llm':
            continue
        if chatlog[promptNumber]['process']['module'] == 'PLANNER':
            prompt = chatlog[promptNumber]['prompt']
            return prompt
    return ""

def getProcessSummary(chatstatus, chatlog):
    for promptNumber in chatlog.keys():
        if promptNumber == 'llm':
            continue
        if chatlog[promptNumber]['process']['module'] == 'PLANNER':
            fullPlanner = str(chatlog[promptNumber]['status']['queue'])
    log.debugLog(fullPlanner, chatstatus=chatstatus) 
    template = summarizeAnalysisPipelineTemplate()
    PROMPT = PromptTemplate(input_variables=["pipeline"], template=template)
    chain = PROMPT | chatstatus['llm']
    res = chain.invoke(fullPlanner)
    log.debugLog('Process Summary Output: \n\n' + str(res.content), chatstatus)
    processSummary   = res.content.split('=')[1].strip()
    log.debugLog('Process Summary=' + str(processSummary), chatstatus)
    return processSummary

def getReportBody(chatstatus, chatlog):
    """Summarizes individual steps"""
    reportBody = ""
    for promptNumber in chatlog.keys():
        if promptNumber == 'llm':
            continue
        # Add to the report body based on the module
        module = chatlog[promptNumber]['process']['module']
        if module == 'RAG':
            reportBody += ragReporter(chatlog[promptNumber], chatstatus)
        elif module == 'CODE':
            reportBody += codeReporter(chatlog[promptNumber], chatstatus=chatstatus)
        elif module == 'DATABASE':
            reportBody += databaseReporter(chatlog[promptNumber], chatstatus)
        # Add to the report body based on the elements
        # reportBody += addFigures(chatlog[promptNumber])
        # reportBody += databaseReporter(chatlog[promptNumber])
    return reportBody

def codeReporter(chatlogStep, currentreport='', chatstatus=None):
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
Summary=<put summary here>
"""
    template = template.format(currentreport=currentreport,
                               docstring=docstring,
                               code=code,
                               codeoutput=output,
    )
    print(template)
    PROMPT = PromptTemplate(input_variables=["userprompt"], template=template)
    chain = PROMPT | chatstatus['llm']
    response = chain.invoke(chatlogStep['prompt'])
    print(response)
    return response.content.split('=')[1]

def ragReporter(chatlogStep, chatstatus):
    output = chatlogStep['output']
    template = summarizeRAGTemplate()
    log.debugLog(template, display=True)
    # template = template.format(output=output)
    PROMPT = PromptTemplate(input_variables=["output"], template=template)
    chain = PROMPT | chatstatus['llm']
    res = chain.invoke(output)
    processSummary   = res.content.split('=')[1].strip().replace('_', '\_')
    return processSummary
    
def databaseReporter(chatlogStep, chatstatus):
    databaseTex = ""
    prompt = chatlogStep['prompt']
    output = chatlogStep['output']

    for step in chatlogStep['process']['steps']:
        # Check if the step saved a figure
        if 'func' in step.keys() and step['func'] == 'utils.save':
            filename = step['new file']
            chatstatus = log.userOutput(filename, chatstatus=chatstatus) # 
            df = utils.load_file_to_dataframe(filename)
    if df is not None:
        numrows = df.shape[0]
    else:
        numrows = 0
    
    template = summarizeDatabaseCallerTemplate()
    log.debugLog(template, display=True)
    template = template.format(output=output, numrows=numrows)
    PROMPT = PromptTemplate(input_variables=["input"], template=template)
    chain = PROMPT | chatstatus['llm']
    res = chain.invoke(prompt)
    processSummary   = res.content.split('=')[1].strip().replace('_', '\_')
    databaseTex += processSummary
    databaseTex += addFigures(chatlogStep, chatstatus)
    databaseTex += addTables(chatlogStep, chatstatus)
    
    return databaseTex

def addFigures(chatlogStep, chatstatus):
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

def addTables(chatlogStep, chatstatus):
    """Add tables to the latex output file"""
    tableTex = ""
    df = None
    # Choose table
    for step in chatlogStep['process']['steps']:
        # Check if the step saved a figure
        if 'func' in step.keys() and step['func'] == 'utils.save':
            filename = step['new file']
            chatstatus = log.userOutput(filename, chatstatus=chatstatus) 
            df = utils.load_file_to_dataframe(filename)
            break
    if df is None:
        return ""
    log.debugLog(type(df), chatstatus=chatstatus) 
    log.debugLog(df.head(3), chatstatus=chatstatus) 
    # df = pd.read_csv(filename)
    # Build latex
    tableTex += ('\n\n' + dataframe_to_latex(df) + '\n\n')
    return tableTex

def getReferences(chatstatus, chatlog):
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