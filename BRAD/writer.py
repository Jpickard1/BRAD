import pandas as pd
import os
import json

from langchain import PromptTemplate, LLMChain

from BRAD import log
from BRAD import utils
from BRAD.promptTemplates import setReportTitleTemplate, summarizeAnalysisPipelineTemplate

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
    report = getFirstLatexReportOutline()

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
        print(promptNumber)
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
    print(fullPlanner)
    template = summarizeAnalysisPipelineTemplate()
    PROMPT = PromptTemplate(input_variables=["pipeline"], template=template)
    chain = PROMPT | chatstatus['llm']
    res = chain.invoke(fullPlanner)
    log.debugLog('Process Summary Output: \n\n' + str(res.content), chatstatus)
    processSummary   = res.content.split('=')[1].strip()
    log.debugLog('Process Summary=' + str(processSummary), chatstatus)
    return processSummary

def getReportBody(chatstatus, chatlog):
    reportBody = '\n\n\n REPORT BODY\n\n\n'
    return reportBody

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

def getFirstLatexReportOutline():
    report = """\\documentclass{{article}}
\\usepackage[margin=1in]{{geometry}}
\\begin{{document}}

\\title{{Report on Chat Pipeline Activity}}
\\author{{BRAD}}
\\maketitle

\\section*{{Driving Prompt}}
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