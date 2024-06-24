import pandas as pd
import os
import json
from tabulate import tabulate

from langchain import PromptTemplate, LLMChain

from BRAD import log
from BRAD import utils
from BRAD.promptTemplates import setReportTitleTemplate, summarizeAnalysisPipelineTemplate, summarizeDatabaseCallerTemplate, summarizeRAGTemplate

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
    report_file = 'REPORT.tex'
    chatstatus = utils.save(chatstatus, report, report_file)
    report_file = chatstatus['process']['steps'][-1]['new file']
    return chatstatus, report_file

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
            fullPlanner = str(chatlog[promptNumber]['planned'])
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
    reportBody = ""
    for promptNumber in chatlog.keys():
        if promptNumber == 'llm':
            continue
        # Add to the report body based on the module
        module = chatlog[promptNumber]['process']['module']
        if module == 'RAG':
            reportBody += ragReporter(chatlog[promptNumber], chatstatus)
        elif module == 'CODE':
            reportBody += codeReporter(chatlog[promptNumber])
        elif module == 'DATABASE':
            reportBody += databaseReporter(chatlog[promptNumber], chatstatus)
        # Add to the report body based on the elements
        # reportBody += addFigures(chatlog[promptNumber])
        # reportBody += databaseReporter(chatlog[promptNumber])
    return reportBody

def codeReporter(chatlogStep):
    return "\n\nCode Called\n\n"

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
    databaseTex += addFigures(chatlogStep)
    databaseTex += addTables(chatlogStep)
    
    return databaseTex

def addFigures(chatlogStep):
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

def addTables(chatlogStep):
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