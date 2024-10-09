"""

Gene Ontology (GO)
------------------

This module provides functions to perform `Gene Ontology (GO) <https://geneontology.org/>`_ searches, download charts, and retrieve associated publications and annotations based on gene terms. The module interacts with external APIs, such as QuickGO and PubMed, to gather the relevant information.

Available Methods
~~~~~~~~~~~~~~~~~

This module has the following methods:

"""

import requests, sys
import pandas as pd
import json
import csv
from requests_html import HTMLSession
import requests
from requests.exceptions import ConnectionError
import os
import copy
from io import StringIO

from BRAD import log

def geneOntology(state, goQuery):
    """
    Performs Gene Ontology (GO) search for specified genes and updates the chat status with the results.

    :param goQuery: The query string containing gene names or terms for GO search.
    :type goQuery: str
    
    :param state: The current status of the chat, including the prompt, configuration, and process details.
    :type state: dict

    :raises FileNotFoundError: If the gene list file is not found.

    :return: The updated chat status dictionary containing the GO search results and process details.
    :rtype: dict

    """
    # Auth: Marc Choi
    #       machoi@umich.edu
    current_script_path = os.path.abspath(__file__)
    current_script_dir = os.path.dirname(current_script_path)
    file_path = os.path.join(current_script_dir, 'helperData', 'gene_list.txt')
    with open(file_path, 'r') as file:
        contents = file.read()
    gene_list = contents.split('\n')
    if len(goQuery) > 0:
        go_process = goSearch(goQuery, state)
        state['process']['GO'] = go_process
    return state
            
def goSearch(query, state):
    """
    Performs a search on Gene Ontology (GO) based on the provided query and allows downloading associated charts and papers.

    :param query: The query list containing gene names or terms for GO search.
    :type query: list

    :return: A dictionary containing the GO search process details.
    :rtype: dict

    """
    # Auth: Marc Choi
    #       machoi@umich.edu
    process = {}
    output = {}
    for terms in query:
        output, geneStatus = textGO(terms, state)
        process['output'] = output
        if geneStatus == True:
            state = log.userOutput('\n would you like to download charts associated with these genes [Y/N]?', state=state)
            for term in output:
                go_id = str(term[0])
                chartGO(go_id, state)
                state = log.userOutput('\n would you like to download the paper associated with these genes [Y/N]?', state=state)
                # download2 = input().strip().upper()
                # process['paper_download'] = (download2 == 'Y')
                # if download2 == 'Y':
                pubmedPaper(go_id, state)

                    
        else:
            state = log.userOutput('\n would you like to download the gene product annotation [Y/N]?', state=state)
            for term in query:
                state = log.userOutput(term, state=state)
                annotations(term, state)
    return process


def textGO(query, state):
    """
    Performs a text-based Gene Ontology (GO) search for a specified query and returns the extracted data and gene status.

    :param query: The query string containing gene names or terms for GO search.
    :type query: str

    :raises requests.HTTPError: If the HTTP request to the GO API fails.

    :return: A tuple containing the extracted data and a boolean indicating whether the query is a gene.
    :rtype: tuple

    """
    # Auth: Marc Choi
    #       machoi@umich.edu
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
        state = log.userOutput("No Gene Ontology Available - Searching Gene Products", state=state)
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
            state = log.userOutput(f"ID: {id}", state=state)
    else:
        for result in data['results']:
            id = result['id']
            text = result['definition']['text']
            extracted_data.append((id, text))
            # Print the extracted data
        for id, text in extracted_data:
            state = log.userOutput(f"ID: {id}", state=state)
            state = log.userOutput(f"Text: {text}", state=state)
    return extracted_data, gene

    

#Input is a GO:----- identification for a gene        
def chartGO(identifier, state):

    """
    Downloads a chart for a specified Gene Ontology (GO) identifier.

    :param identifier: The GO identifier for which the chart is to be downloaded.
    :type identifier: str

    :raises requests.HTTPError: If the HTTP request to download the chart fails.

    """
    # Auth: Marc Choi
    #       machoi@umich.edu
    try: 
        path = os.path.abspath(os.getcwd()) + '/go_charts'
        os.makedirs(path, exist_ok = True)
        state = log.userOutput("Directory '%s' created successfully" % path, state=state)
    except OSError as error:
        state = log.userOutput("Directory '%s' can not be created" % path, state=state)
    requestURL = "https://www.ebi.ac.uk/QuickGO/services/ontology/go/terms/{ids}/chart?ids=GO%3A"+identifier[3:]
    img_data = requests.get(requestURL).content

    # save chart
    with open(os.path.join(path, identifier[3:] + '.jpg'), 'wb') as handler:
        handler.write(img_data)

#Input is a GO:----- identification for a gene     
def pubmedPaper(identifier, state):
    """
    Downloads PubMed papers associated with a specified Gene Ontology (GO) identifier.

    :param identifier: The GO identifier for which the associated PubMed papers are to be downloaded.
    :type identifier: str

    :raises requests.HTTPError: If the HTTP request to the PubMed API fails.

    """
    # Auth: Marc Choi
    #       machoi@umich.edu
    
    requestURL2 = "https://www.ebi.ac.uk/QuickGO/services/ontology/go/terms/GO%3A"+identifier[3:]
    r = requests.get(requestURL2, headers={ "Accept" : "application/json"})
    if not r.ok:
        r.raise_for_status()
        sys.exit()
    responseBody = r.text
    data = json.loads(responseBody)
    # Extract the dbId from the first result
    dbId = []
    
    #state = log.userOutput(data, state=state)
    if data['numberOfHits'] > 0:
        xrefs = data['results'][0].get('definition', {}).get('xrefs', [])
        if xrefs:
            for db in xrefs:
                dbId.append(db['dbId'])
        s = HTMLSession()
        headers = {'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36'}
        if len(dbId) > 0:
            try: 
                path = os.path.abspath(os.getcwd()) + '/specialized_docs'
                os.makedirs(path, exist_ok = True) 
                state = log.userOutput("Directory '%s' created successfully" % path, state=state)
            except OSError as error: 
                state = log.userOutput("Directory '%s' can not be created" % path, state=state)
            for idname in dbId:
                state = log.userOutput(idname, state=state)
                try:
                    base_url = 'https://pubmed.ncbi.nlm.nih.gov/'
                    r = s.get(base_url + idname + '/', headers = headers, timeout = 5)
                    try:
                        pdf_url = r.html.find('a.id-link', first=True).attrs['href']
                        state = log.userOutput(pdf_url, state=state)
                        if "doi" in pdf_url:
                            state = log.userOutput("Not public", state=state)
                            continue
                        r = s.get(pdf_url, headers = headers, timeout = 5)
                        pdf_real = 'https://ncbi.nlm.nih.gov'+r.html.find('a.int-view', first=True).attrs['href']
                        state = log.userOutput(pdf_real, state=state)
                        r = s.get(pdf_real, stream=True)
                        with open(os.path.join(path, idname + '.pdf'), 'wb') as f:
                            for chunk in r.iter_content(chunk_size = 1024):
                                if chunk:
                                    f.write(chunk)
                    except AttributeError:
                        pass
                        state = log.userOutput(f"{idname} could not be gathered.", state=state)

                except ConnectionError as e:
                    pass
                    state = log.userOutput(f"{idname} could not be gathered.", state=state)
        else:
            state = log.userOutput(f"No paper associated with {identifier} found on PubMed", state=state)


#THIS ONE WORKS BETTER
def annotations(ids, state):
    """
    Downloads annotations for a specified gene product.

    :param ids: The gene product identifier for which the annotations are to be downloaded.
    :type ids: str

    :raises requests.HTTPError: If the HTTP request to download the annotations fails.

    """
    # Auth: Marc Choi
    #       machoi@umich.edu
    try: 
        path = os.path.abspath(os.getcwd()) + '/go_annotations'
        os.makedirs(path, exist_ok = True) 
        state = log.userOutput("Directory '%s' created successfully" % path, state=state)
    except OSError as error: 
        state = log.userOutput("Directory '%s' can not be created" % path, state=state)
        
        
    requestURL = "https://amigo.geneontology.org/amigo/search/annotation?q="+ids
    real_id = ids[3:]
    fetch_annotation(real_id, path)


def fetch_annotation(id, path):
    # URL to fetch data from
    url = 'https://golr-aux.geneontology.io/solr/select?defType=edismax&qt=standard&indent=on&wt=csv&rows=100000&start=0&fl=source%2Cbioentity_internal_id%2Cbioentity_label%2Cqualifier%2Cannotation_class%2Creference%2Cevidence_type%2Cevidence_with%2Caspect%2Cbioentity_name%2Csynonym%2Ctype%2Ctaxon%2Cdate%2Cassigned_by%2Cannotation_extension_class%2Cbioentity_isoform&facet=true&facet.mincount=1&facet.sort=count&json.nl=arrarr&facet.limit=25&hl=true&hl.simple.pre=%3Cem%20class%3D%22hilite%22%3E&hl.snippets=1000&csv.encapsulator=&csv.separator=%09&csv.header=false&csv.mv.separator=%7C&fq=document_category:%22annotation%22&facet.field=aspect&facet.field=taxon_subset_closure_label&facet.field=type&facet.field=evidence_subset_closure_label&facet.field=regulates_closure_label&facet.field=isa_partof_closure_label&facet.field=annotation_class_label&facet.field=qualifier&facet.field=annotation_extension_class_closure_label&facet.field=assigned_by&facet.field=panther_family_label&q=GO%3A'+id+'&qf=annotation_class%5E2&qf=annotation_class_label_searchable%5E1&qf=bioentity%5E2&qf=bioentity_label_searchable%5E1&qf=bioentity_name_searchable%5E1&qf=annotation_extension_class%5E2&qf=annotation_extension_class_label_searchable%5E1&qf=reference_searchable%5E1&qf=panther_family_searchable%5E1&qf=panther_family_label_searchable%5E1&qf=bioentity_isoform%5E1&qf=isa_partof_closure%5E1&qf=isa_partof_closure_label_searchable%5E1'
    print(url)
# Fetch data from the URL
    response = requests.get(url)

# Check if the request was successful
    if response.status_code == 200:
    # Read the data into a pandas DataFrame
        data = StringIO(response.text)
        df = pd.read_csv(data, sep='\t')  # '\t' is the tab separator as specified in the URL
    
    # Define the new column names
        new_column_names = [
            'Source',
            'Bioentity Internal ID',
            'Bioentity Label',
            'Qualifier',
            'Annotation Class',
            'Reference',
            'Evidence Type',
            'Evidence With',
            'Aspect',
            'Bioentity Name',
            'Synonym',
            'Type',
            'Taxon',
            'Date',
            'Assigned By',
            'Annotation Extension Class',
            'Bioentity Isoform'
        ]
    
    # Rename the columns directly
        df.columns = new_column_names
    
    # Save the updated DataFrame to a CSV file
        df.to_csv(path+'/annotations_go_'+id+'.csv', index=False)
        print("Data with updated column names has been saved to "+path+"/annotations_go_"+id+".csv")
    else:
        print(f"Failed to retrieve data. Status code: {response.status_code}")



