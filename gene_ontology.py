import requests, sys
import pandas as pd
import json
import csv

def textGO(query):
    requestURL = "https://www.ebi.ac.uk/QuickGO/services/ontology/go/search?query="+query+"&limit=25&page=1"
    r = requests.get(requestURL, headers={ "Accept" : "application/json"})

    if not r.ok:
        r.raise_for_status()
        sys.exit()
    extracted_data = []

    responseBody = r.text
    data = json.loads(responseBody)
    if data['numberOfHits'] == 0:
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
    return extracted_data

    
        
def chartGO(identifier):
    requestURL = "https://www.ebi.ac.uk/QuickGO/services/ontology/go/terms/{ids}/chart?ids=GO%3A"+identifier[3:]
    img_data = requests.get(requestURL).content
    # save chart
    with open(identifier+'.jpg', 'wb') as handler:
        handler.write(img_data)
        
def pubmedPaper(identifier):
    requestURL2 = "https://www.ebi.ac.uk/QuickGO/services/ontology/go/terms/GO%3A"+identifier[3:]
    r = requests.get(requestURL2, headers={ "Accept" : "application/json"})
    if not r.ok:
        r.raise_for_status()
        sys.exit()
    responseBody = r.text
    data = json.loads(responseBody)
    # Extract the dbId from the first result
    dbId = None
    if data['numberOfHits'] > 0:
        xrefs = data['results'][0].get('definition', {}).get('xrefs', [])
        if xrefs:
            dbId = xrefs[0].get('dbId')
    print(dbId)

def annotations(ids):
    requestURL = "https://www.ebi.ac.uk/QuickGO/services/annotation/downloadSearch?geneProductId="+ids

    r = requests.get(requestURL, headers={ "Accept" : "text/tsv"})

    if not r.ok:
        r.raise_for_status()
        sys.exit()

    responseBody = r.text
    lines = responseBody.strip().split('\n')
    with open(ids+".tsv", 'w', newline='') as file:
        writer = csv.writer(file, delimiter='\t')
    
        for line in lines:
            # Split each line by tab and write to the TSV file
            writer.writerow(line.split('\t'))
