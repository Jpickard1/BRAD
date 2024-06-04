import requests, sys
import pandas as pd
import json
import csv
from requests_html import HTMLSession
import requests
from requests.exceptions import ConnectionError
import os
import copy

def geneOntology(goQuery, chatstatus):
    """
    Searches for gene terms in a provided query using a predefined gene list, 
    and optionally initiates a Gene Ontology search based on user input.

    Parameters:
    goQuery (str): The query string containing potential gene terms to be searched.
    chatstatus (dict): A dictionary containing the status of the chat and process information.

    Returns:
    dict: Updated chatstatus dictionary with the results of the Gene Ontology search if initiated.
    """
    current_script_path = os.path.abspath(__file__)
    current_script_dir = os.path.dirname(current_script_path)
    file_path = os.path.join(current_script_dir, 'helperData', 'gene_list.txt')
    with open(file_path, 'r') as file:
        contents = file.read()
    gene_list = contents.split('\n')
    real_list = []
    for words in goQuery.split(' '):
        words = words.upper()
        if words in gene_list:
            real_list.append(words)
    if len(real_list) > 0:
        print(real_list)
        chatstatus['output'] += '\n would you search Gene Ontology for these terms [Y/N]?'
        print('\n would you search Gene Ontology for these terms [Y/N]?')
        go = input().strip().upper()
        chatstatus['process']['search'] = (go == 'Y')
        if go == 'Y':
            go_process = gonto.goSearch(real_list)
            chatstatus['process']['GO'] = go_process
    return chatstatus
            
def goSearch(query):
    """
    Performs a Gene Ontology search for the given query terms and prompts the user
    for various actions based on the search results.

    Parameters:
    query (list of str): A list of gene terms to search for in the Gene Ontology database.

    Returns:
    dict: A dictionary containing the process status and user decisions on downloading charts,
          papers, and gene product annotations.
    """
    process = {}
    output = {}
    for terms in query:
        output, geneStatus = textGO(terms)
        process['output'] = output
        if geneStatus == True:
            print('\n would you like to download charts associated with these genes [Y/N]?')
            download = input().strip().upper()
            process['chart_download'] = (download == 'Y')
            if download == 'Y':   
                for term in output:
                    go_id = str(term[0])
                    chartGO(go_id)
                    print('\n would you like to download the paper associated with these genes [Y/N]?')
                    download2 = input().strip().upper()
                    process['paper_download'] = (download2 == 'Y')
                    if download2 == 'Y':
                        pubmedPaper(go_id)
                    
        else:
            print('\n would you like to download the gene product annotation [Y/N]?')
            download = input().strip().upper()
            process['annotation_download'] = (download == 'Y')
            if download == 'Y':
                for term in query:
                    print(term)
                    annotations(term)
    return process


def textGO(query):
    """
    Searches the Gene Ontology (GO) database or Gene Products database for the given query and extracts relevant information.

    This function first searches the GO database for the specified query. If no results are found, it searches the Gene Products database.
    The extracted data includes IDs and definitions (if available) from the search results.

    Args:
        query (str): The search query string.

    Returns:
        tuple: A tuple containing:
            - extracted_data (list): A list of tuples or strings with the extracted IDs and definitions from the search results.
            - gene (bool): A boolean indicating whether the search results were found in the GO database (True) or the Gene Products database (False).

    Raises:
        HTTPError: If an HTTP error occurs during the requests.
        SystemExit: If the request fails and the program needs to exit.

    Examples:
        >>> textGO("kinase")
        ID: GO:0004672
        Text: Catalysis of the transfer of a phosphate group, usually from ATP, to a substrate molecule.
        ...
        ([(GO:0004672, 'Catalysis of the transfer of a phosphate group, usually from ATP, to a substrate molecule.'), ...], True)
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
    Retrieves a chart image for a given Gene Ontology (GO) term identifier and saves it as a JPEG file.

    This function constructs a request URL using the provided GO term identifier, retrieves the chart image from the GO database, 
    and saves it to a file named after the GO term ID.

    Args:
        identifier (str): The Gene Ontology (GO) term identifier in the format 'GO:XXXXXXX'.

    Returns:
        None

    Raises:
        HTTPError: If an HTTP error occurs during the request.
        IOError: If there is an error saving the image file.

    Examples:
        >>> chartGO("GO:0008150")
        # This will save the chart image for GO:0008150 as '0008150.jpg' in the current directory.
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
    Retrieves the first PubMed paper associated with a given Gene Ontology (GO) term identifier and saves it as a PDF file.

    This function constructs a request URL using the provided GO term identifier, retrieves the relevant information from the QuickGO database,
    extracts the first cross-reference ID (dbId), and then attempts to download the associated PubMed paper PDF. The PDF is saved in a 
    directory named 'specialized_docs' within the current working directory.

    Args:
        identifier (str): The Gene Ontology (GO) term identifier in the format 'GO:XXXXXXX'.

    Returns:
        None

    Raises:
        HTTPError: If an HTTP error occurs during the requests.
        SystemExit: If the request fails and the program needs to exit.
        IOError: If there is an error creating the directory or saving the PDF file.

    Examples:
        >>> pubmedPaper("GO:0008150")
        # This will download the first PubMed paper associated with GO:0008150 and save it as a PDF.
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
    Downloads annotation data for a given gene product ID from the QuickGO database and saves it as a TSV file.

    This function constructs a request URL using the provided gene product ID, retrieves the annotation data in TSV format from the QuickGO database, 
    and saves it to a file named after the gene product ID.

    Args:
        ids (str): The gene product ID.

    Returns:
        None

    Raises:
        HTTPError: If an HTTP error occurs during the request.
        SystemExit: If the request fails and the program needs to exit.
        IOError: If there is an error saving the TSV file.

    Examples:
        >>> download_annotations("P12345")
        # This will save the annotation data for gene product ID P12345 as 'P12345.tsv' in the current directory.
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
