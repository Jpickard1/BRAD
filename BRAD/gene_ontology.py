import requests, sys
import pandas as pd
import json
import csv
from requests_html import HTMLSession
import requests
from requests.exceptions import ConnectionError
import os
import copy

def geneOntology(chatstatus, goQuery):
    """
    Performs Gene Ontology (GO) search for specified genes and updates the chat status with the results.

    :param goQuery: The query string containing gene names or terms for GO search.
    :type goQuery: str
    :param chatstatus: The current status of the chat, including the prompt, configuration, and process details.
    :type chatstatus: dict

    :raises FileNotFoundError: If the gene list file is not found.

    :return: The updated chat status dictionary containing the GO search results and process details.
    :rtype: dict

    """
    current_script_path = os.path.abspath(__file__)
    current_script_dir = os.path.dirname(current_script_path)
    file_path = os.path.join(current_script_dir, 'helperData', 'gene_list.txt')
    with open(file_path, 'r') as file:
        contents = file.read()
    gene_list = contents.split('\n')
    real_list = goQuery # []
    #for words in goQuery.split(' '):
    #    words = words.upper()
    #    if words in gene_list:
    #        real_list.append(words)
    if len(real_list) > 0:
        print(real_list)
        #chatstatus['output'] += '\n would you search Gene Ontology for these terms [Y/N]?'
        #print('\n would you search Gene Ontology for these terms [Y/N]?')
        #go = input().strip().upper()
        #chatstatus['process']['search'] = (go == 'Y')
        #if go == 'Y':
        go_process = goSearch(real_list)
        chatstatus['process']['GO'] = go_process
    return chatstatus
            
def goSearch(query):
    """
    Performs a search on Gene Ontology (GO) based on the provided query and allows downloading associated charts and papers.

    :param query: The query list containing gene names or terms for GO search.
    :type query: list

    :return: A dictionary containing the GO search process details.
    :rtype: dict

    """
    process = {}
    output = {}
    for terms in query:
        output, geneStatus = textGO(terms)
        process['output'] = output
        if geneStatus == True:
            #print('\n would you like to download charts associated with these genes [Y/N]?')
            #download = input().strip().upper()
            #process['chart_download'] = (download == 'Y')
            #if download == 'Y':   
            for term in output:
                go_id = str(term[0])
                chartGO(go_id)
                # print('\n would you like to download the paper associated with these genes [Y/N]?')
                # download2 = input().strip().upper()
                # process['paper_download'] = (download2 == 'Y')
                # if download2 == 'Y':
                pubmedPaper(go_id)
                    
        else:
            # print('\n would you like to download the gene product annotation [Y/N]?')
            # download = input().strip().upper()
            # process['annotation_download'] = (download == 'Y')
            # if download == 'Y':
            for term in query:
                print(term)
                annotations(term)
    return process


def textGO(query):
    """
    Performs a text-based Gene Ontology (GO) search for a specified query and returns the extracted data and gene status.

    :param query: The query string containing gene names or terms for GO search.
    :type query: str

    :raises requests.HTTPError: If the HTTP request to the GO API fails.

    :return: A tuple containing the extracted data and a boolean indicating whether the query is a gene.
    :rtype: tuple

    """
    gene = True
    requestURL = "https://www.ebi.ac.uk/QuickGO/services/ontology/go/search?query="+query+"&limit=25&page=1"
    r = requests.get(requestURL, headers={ "Accept" : "application/json"})

    if not r.ok:
        r.raise_for_status()
        sys.exit()
    extracted_data = []

    responseBody = r.text
    data = json.loads(responseBody)
    if data['numberOfHits'] == 0:
        gene=False
        print("No Gene Ontology Available - Searching Gene Products")
        requestURL = requestURL = "https://www.ebi.ac.uk/QuickGO/services/geneproduct/search?query="+query
        r = requests.get(requestURL, headers={ "Accept" : "application/json"})

        if not r.ok:
            r.raise_for_status()
            sys.exit()

        responseBody = r.text
        data = json.loads(responseBody)
        for result in data['results']:
            id = result['id']
            extracted_data.append(id)
        for id in extracted_data:
            print(f"ID: {id}")
    else:
        for result in data['results']:
            id = result['id']
            text = result['definition']['text']
            extracted_data.append((id, text))
            # Print the extracted data
        for id, text in extracted_data:
            print(f"ID: {id}")
            print(f"Text: {text}")
    return extracted_data, gene

    
#Input is a GO:----- identification for a gene        
def chartGO(identifier):
    """
    Downloads a chart for a specified Gene Ontology (GO) identifier.

    :param identifier: The GO identifier for which the chart is to be downloaded.
    :type identifier: str

    :raises requests.HTTPError: If the HTTP request to download the chart fails.

    :return: None

    """
    try: 
        path = os.path.abspath(os.getcwd()) + '/go_charts'
        os.makedirs(path, exist_ok = True) 
        print("Directory '%s' created successfully" % path) 
    except OSError as error: 
        print("Directory '%s' can not be created" % path)
    requestURL = "https://www.ebi.ac.uk/QuickGO/services/ontology/go/terms/{ids}/chart?ids=GO%3A"+identifier[3:]
    img_data = requests.get(requestURL).content
    # save chart
    with open(os.path.join(path, identifier[3:] + '.jpg'), 'wb') as handler:
        handler.write(img_data)

#Input is a GO:----- identification for a gene     
def pubmedPaper(identifier):
    """
    Downloads PubMed papers associated with a specified Gene Ontology (GO) identifier.

    :param identifier: The GO identifier for which the associated PubMed papers are to be downloaded.
    :type identifier: str

    :raises requests.HTTPError: If the HTTP request to the PubMed API fails.

    :return: None

    """
    requestURL2 = "https://www.ebi.ac.uk/QuickGO/services/ontology/go/terms/GO%3A"+identifier[3:]
    r = requests.get(requestURL2, headers={ "Accept" : "application/json"})
    if not r.ok:
        r.raise_for_status()
        sys.exit()
    responseBody = r.text
    data = json.loads(responseBody)
    # Extract the dbId from the first result
    dbId = []
    print(data)
    if data['numberOfHits'] > 0:
        xrefs = data['results'][0].get('definition', {}).get('xrefs', [])
        if xrefs:
            for db in xrefs:
                dbId.append(db['dbId'])
        s = HTMLSession()
        print(dbId)
        headers = {'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36'}
        if len(dbId) > 0:
            try: 
                path = os.path.abspath(os.getcwd()) + '/specialized_docs'
                os.makedirs(path, exist_ok = True) 
                print("Directory '%s' created successfully" % path) 
            except OSError as error: 
                print("Directory '%s' can not be created" % path)
            for idname in dbId:
                print(idname)
                try:
                    base_url = 'https://pubmed.ncbi.nlm.nih.gov/'
                    r = s.get(base_url + idname + '/', headers = headers, timeout = 5)
                    try:
                        pdf_url = r.html.find('a.id-link', first=True).attrs['href']
                        print(pdf_url)
                        if "doi" in pdf_url:
                            print('Not public')
                            continue
                        r = s.get(pdf_url, headers = headers, timeout = 5)
                        pdf_real = 'https://ncbi.nlm.nih.gov'+r.html.find('a.int-view', first=True).attrs['href']
                        print(pdf_real)
                        r = s.get(pdf_real, stream=True)
                        with open(os.path.join(path, idname + '.pdf'), 'wb') as f:
                            for chunk in r.iter_content(chunk_size = 1024):
                                if chunk:
                                    f.write(chunk)
                    except AttributeError:
                        pass
                        print(f"{idname} could not be gathered.")

                except ConnectionError as e:
                    pass
                    print(f"{idname} could not be gathered.")
        else:
            print(f"No paper associated with {identifier} found on PubMed")
#Input is a gene-product  
def annotations(ids):
    """
    Downloads annotations for a specified gene product.

    :param ids: The gene product identifier for which the annotations are to be downloaded.
    :type ids: str

    :raises requests.HTTPError: If the HTTP request to download the annotations fails.

    :return: None

    """
    try: 
        path = os.path.abspath(os.getcwd()) + '/go_annotations'
        os.makedirs(path, exist_ok = True) 
        print("Directory '%s' created successfully" % path) 
    except OSError as error: 
        print("Directory '%s' can not be created" % path)
        
    requestURL = "https://www.ebi.ac.uk/QuickGO/services/annotation/downloadSearch?geneProductId="+ids

    r = requests.get(requestURL, headers={ "Accept" : "text/tsv"})

    if not r.ok:
        r.raise_for_status()
        sys.exit()
    responseBody = r.text
    lines = responseBody.strip().split('\n')
    with open(os.path.join(path, ids + '.tsv'), 'w', newline='') as file:
        writer = csv.writer(file, delimiter='\t')
    
        for line in lines:
            # Split each line by tab and write to the TSV file
            writer.writerow(line.split('\t'))
