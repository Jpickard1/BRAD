import subprocess
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
import chromadb
import pandas as pd
from requests_html import HTMLSession
import requests
from requests.exceptions import ConnectionError
import os
from Bio import Entrez
import math
import pandas as pd
import datetime
import time
import sys
import string
import gc
from bs4 import BeautifulSoup as bs
from urllib.parse import urljoin
from requests_html import HTMLSession
import requests
from requests.exceptions import ConnectionError
from langchain import PromptTemplate, LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

def webScraping(chatstatus):
    query    = chatstatus['prompt']
    llm      = chatstatus['llm']              # get the llm
    memory   = chatstatus['memory']           # get the memory of the model
    
    # Define the mapping of keywords to functions
    scraping_functions = {
        'ARXIV'   : arxiv,
        'BIORXIV' : biorxiv,
        'PUBMED'  : pubmed
    }
    
    # Identify the database and the search terms
    template = """Current conversation:\n{history}
    
    Query:{input}
    
    From the query, decide if ARXIV, PUBMED, or BIORXIV should be searched, and propose no more than 10 search terms for this query and database. Separate each term with a comma, and provide no extra information/explination for either the database or search terms. Format your output as follows with no additions:
    
    Database: <ARXIV, PUBMED, or BIORXIV>
    Search Terms: <improved search terms>
    """
    PROMPT = PromptTemplate(input_variables=["history", "input"], template=template)
    conversation = ConversationChain(prompt  = PROMPT,
                                     llm     = llm,
                                     verbose = chatstatus['config']['debug'],
                                     memory  = memory,
                                    )
    response = conversation.predict(input=query)
    llmResponse = parse_llm_response(response)
    llmKey, searchTerms = llmResponse['database'], llmResponse['search_terms']

    # Determine the target source
    source = next((key for key in scraping_functions if key == llmKey), 'PUBMED')
    process = {'searched': source}
    scrape_function = scraping_functions[source]
    
    # Execute the scraping function and handle errors
    try:
        output = f'searching on {source}...'
        print(output)
        print('Search Terms: ' + str(searchTerms)) if chatstatus['config']['debug'] else None
        for st in searchTerms:
            scrape_function(st)
        searchTerms = ' '.join(searchTerms)
        scrape_function(searchTerms)
    except Exception as e:
        output = f'Error occurred while searching on {source}: {e}'
        print(output)
        process = {'searched': 'ERROR'}

    chatstatus['process'] = process
    chatstatus['output']  = output
    return chatstatus

def webScraping_depricated(chatstatus):
    query = chatstatus['prompt']
    
    # Define the mapping of keywords to functions
    scraping_functions = {
        'ARXIV'   : arxiv,
        'BIORXIV' : biorxiv,
        'PUBMED'  : pubmed
    }
    
    # Normalize the query
    query_tokens = query.upper().split()
    query = query.lower()
    query = query.replace('arxiv', '')
    query = query.replace('biorxiv', '')
    query = query.replace('pubmed', '')
    query = ' '.join(query.split())
    remove_punctuation = str.maketrans('', '', string.punctuation)
    query = query.translate(remove_punctuation)
    
    # Determine the target source
    source = next((key for key in scraping_functions if key in query_tokens), 'PUBMED')
    process = {'searched': source}
    scrape_function = scraping_functions[source]
    
    # Execute the scraping function and handle errors
    try:
        output = f'searching on {source}'
        print(output)
        scrape_function(query)
    except Exception as e:
        output = f'Error occurred while searching on {source}: {e}'
        print(output)
        process = {'searched': 'ERROR'}

    chatstatus['process'] = process
    chatstatus['output']  = output
    return chatstatus

def arxiv(query):
    """
    Search for articles on arXiv, display results, and optionally download the articles as PDFs.

    Args:
        query (str): The search query for arXiv.

    Returns:
        tuple: A tuple containing the output message (str) and a process dictionary (dict).
    """
    process = {}
    output = 'searching the following on arxiv: ' + query
    print(output)
    df, pdfs = arxiv_search(query, 10)
    process['search results'] = df
    displayDf = df[['Title', 'Authors', 'Abstract']]
    display(displayDf)
    output += '\n would you like to download these articles [Y/N]?'
    print('would you like to download these articles [Y/N]?')
    download = input().strip().upper()
    process['download'] = (download == 'Y')
    if download == 'Y':
        #id_list = df['id'].to_list()
        output += arxiv_scrape(pdfs)
    return output, process

def arxiv_search1(query):
    """
    DEPRECATED
    Search for articles on arXiv based on the given query.

    Args:
        query (str): The search query for arXiv.

    Returns:
        pd.DataFrame: A DataFrame containing the search results with columns 'id', 'title', 'categories', 
                      'abstract', 'doi', 'created', 'updated', and 'authors'.
    """
    pd.set_option('display.max_colwidth', None)
    #date_from='2024-04-01',date_until='2024-05-01',
    #
    scraper = arxivscraper.Scraper(category='q-bio', date_from='2023-05-01',date_until='2024-05-01', filters={'abstract':[query]})
    output = scraper.scrape()
    cols = ('id', 'title', 'categories', 'abstract', 'doi', 'created', 'updated', 'authors')
    df = pd.DataFrame(output,columns=cols)
    return df

def arxiv_scrape1(id_list):
    """
    DEPRECATED
    Download articles from arXiv as PDFs based on the given list of IDs.

    Args:
        id_list (list): A list of article IDs to download from arXiv.

    Returns:
        str: A string indicating the outcome of the download process.
    """
    output = ''
    session = HTMLSession()
    headers = {'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36'}
    try:
        path = os.path.join(os.getcwd(), 'specialized_docs')
        os.makedirs(path, exist_ok = True) 
        output = f"Directory '{path}' created successfully"
    except OSError:
        output = f"Directory '{path}' could not be created"
    print(output)

    #Scrape arxiv and
    for arxiv_id  in id_list:
        try:
            pdf_url = f'https://arxiv.org/pdf/{arxiv_id}'
            local_path = os.path.join(path, f'{arxiv_id}.pdf')
            print(f'{pdf_url} --> {local_path}')
            output += f'\n{pdf_url} --> {local_path}'

            response = session.get(pdf_url, stream=True)
            response.raise_for_status()

            with open(os.path.join(path, ids + '.pdf'), 'wb') as f:
                for chunk in response.iter_content(chunk_size = 1024):
                    f.write(chunk)
        except (ConnectionError, requests.exceptions.RequestException) as e:
            print(f"{arxiv_id} could not be gathered: {e}")
            output += f"\n{arxiv_id} could not be gathered."
    return output

def search_pubmed_article(query, number_of_articles=10):
    """
    Searches PubMed for articles matching the specified query and retrieves their PMIDs.

    This function uses the NCBI Entrez E-utilities to search the PubMed database for articles 
    that match the given query. It returns a list of PMIDs for the articles found.

    Parameters:
    query (str): The keyword or phrase to search for in PubMed articles.
    number_of_articles (int, optional): The maximum number of article PMIDs to return. Defaults to 10.

    Returns:
    list: A list of PMIDs (str) for articles matching the query.

    Example:
    >>> pmids = search_pubmed_article('cancer research')
    >>> print(pmids)
    ['12345678', '23456789', ...]

    Notes:
    - Ensure you have set up your email address with Entrez before making requests to NCBI.
    - The search results are sorted by relevance by default.

    """
    Entrez.email = 'inhyak@gmail.com'
    handle = Entrez.esearch(db='pubmed', term = query, retmax=number_of_articles, sort='relevance')
    record = Entrez.read(handle)
    #record = Entrez.ecitmatch()
    handle.close()
    return record['IdList']
    #return record

def pubmed(query):
    """
    Scrapes PubMed for articles matching the query, retrieves their PMIDs, and downloads available PDFs.

    Parameters:
    query (str): The keyword to search for in PubMed articles.

    Returns:
    None

    Example:
    >>> pubmedscrape('cancer research')
    This will search for PubMed articles related to 'cancer research', retrieve their PMIDs,
    and download available PDFs to the 'specialized_docs' directory.

    """
    pmid_list = search_pubmed_article(query)
    citation_arr = []
    if pmid_list:
        for pmid in pmid_list:
            citation = pmid
            citation_arr.append(citation)
        s = HTMLSession()

        headers = {'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36'}
        try: 
            path = os.path.abspath(os.getcwd()) + '/specialized_docs'
            os.makedirs(path, exist_ok = True) 
            print("Directory '%s' created successfully" % path) 
        except OSError as error: 
            print("Directory '%s' can not be created" % path) 
        for pmc in citation_arr:
            try:
                base_url = 'https://pubmed.ncbi.nlm.nih.gov/'
                r = s.get(base_url + pmc + '/', headers = headers, timeout = 5)
                pdf_url = r.html.find('a.id-link', first=True).attrs['href']
                if "doi" in pdf_url:
                    continue
                r = s.get(pdf_url, headers = headers, timeout = 5)
                try:
                    pdf_real = 'https://ncbi.nlm.nih.gov'+r.html.find('a.int-view', first=True).attrs['href']
                    r = s.get(pdf_real, stream=True)
                    with open(os.path.join(path, pmc + '.pdf'), 'wb') as f:
                        for chunk in r.iter_content(chunk_size = 1024):
                            if chunk:
                                f.write(chunk)
                except AttributeError as e:
                    pass
                    print(f"{pmc} could not be gathered.")
                
                    
            except ConnectionError as e:
                pass
                print(f"{pmc} could not be gathered.")

    else:
        print("no articles found")
    print("pdf collection complete!")

def biorxiv(query):
    """
    Scrapes the bioRxiv preprint server for articles matching a specific query.

    This function uses the `biorxiv_real_search` method to search bioRxiv for articles
    based on the provided query. The search is conducted within the current 
    month by default, and retrieves a maximum of 75 records within 300 seconds.
    
    Parameters:
    query (str): The keyword to search for in bioRxiv articles.

    Returns:
    None

    Example:
    >>> biorxiv_scrape('machine learning')
    This will search for bioRxiv articles related to 'machine learning' published 
    within the current month and print the titles, authors, and URLs of the articles found.

    Notes:
    - The search is restricted to articles in the 'biorxiv' journal.
    - The search is conducted within the time frame from the first day of the current 
      month to the current date.
    - By default, the function does not include abstracts in the output.
    - The columns retrieved for each article include the title, authors, and URL.
    - The `kwd_type` parameter is set to 'all', meaning all specified keywords must be 
      present in the search results.
    - Adjusting parameters like `subjects`, `athr`, and `abstracts` might be necessary 
      for more refined searches.
    """
    biorxiv_real_search(start_date  = datetime.date.today().replace(year=2015), 
                        end_date    = datetime.date.today(),
                        subjects    = [], 
                        journal     = 'biorxiv',
                        kwd         = [query], 
                        kwd_type    = 'all', 
                        athr        = [], 
                        max_records = 75, 
                        max_time    = 300,
                        cols        = ['title', 'authors', 'url'],
                        abstracts   = False
                       )

def biorxiv_real_search(start_date  = datetime.date.today().replace(year=2015), 
                        end_date    = datetime.date.today(), 
                        subjects    = [], 
                        journal     = 'biorxiv',
                        kwd         = [], 
                        kwd_type    = 'all', 
                        athr        = [], 
                        max_records = 75, 
                        max_time    = 300,
                        cols        = ['title', 'authors', 'url'],
                        abstracts   = False):
    """
    Searches a specified journal for articles matching given criteria and returns their details.

    This function builds a search query URL based on the given parameters, fetches the search results
    from the specified journal, and returns a DataFrame with the details of the articles found.

    Parameters:
    start_date (datetime.date, optional): The start date for the search. Defaults to the first day of the current month.
    end_date (datetime.date, optional): The end date for the search. Defaults to today.
    subjects (list, optional): List of subjects to filter the search by. Defaults to an empty list.
    journal (str, optional): The journal to search. Defaults to 'biorxiv'.
    kwd (list, optional): List of keywords to filter the search by. Defaults to an empty list.
    kwd_type (str, optional): Type of keyword matching ('all', 'any'). Defaults to 'all'.
    athr (list, optional): List of author names to filter the search by. Defaults to an empty list.
    max_records (int, optional): Maximum number of records to fetch. Defaults to 75.
    max_time (int, optional): Maximum time (in seconds) to spend on the search. Defaults to 300.
    cols (list, optional): List of columns to include in the returned DataFrame. Defaults to ['title', 'authors', 'url'].
    abstracts (bool, optional): Whether to fetch abstracts for the articles. Defaults to False.

    Returns:
    pd.DataFrame: A DataFrame containing the details of the articles found.

    Example:
    >>> df = biorxiv_real_search(start_date=datetime.date(2023, 1, 1), end_date=datetime.date(2023, 1, 31), subjects=['neuroscience'], kwd=['brain'], max_records=10)
    >>> print(df)
       title        authors                                                url
    0  Title1  [Author1, Author2]  http://www.biorxiv.org/content/10.1101/2023...

    Notes:
    - The function assumes that the BeautifulSoup, pandas, and requests libraries are installed and properly configured.
    - The search is performed on the specified journal's website, and the articles' details are extracted from the search results.
    - If `abstracts` is set to True, the function will also fetch the abstracts of the articles, which can increase the execution time.
    """

    ## keep track of timing
    overall_time = time.time()

    ## url
    BASE = 'http://{:s}.org/search/'.format(journal)
    url = BASE
    ## format dates
    start_date = str(start_date)
    end_date = str(end_date)

    ## format inputs
    journal = journal.lower()
    kwd_type = kwd_type.lower()

    ### build the url string

    ## journal selection
    journal_str = 'jcode%3A' + journal
    url += journal_str

    ## subject selection
    if len(subjects) > 0:
        first_subject = ('%20').join(subjects[0].split())
        subject_str = 'subject_collection_code%3A' + first_subject
        for subject in subjects[1:]:
            subject_str = subject_str + '%2C' + ('%20').join(subject.split())
        url += '%20' + subject_str
        
    ## keyword selection
    if len(kwd) > 0:
        kwd_str = 'abstract_title%3A' + ('%252C%2B').join([kwd[0]] + [('%2B').join(keyword.split()) for keyword in kwd[1:]])
        kwd_str = kwd_str + '%20abstract_title_flags%3Amatch-' + kwd_type
        url += '%20' + kwd_str
	## author selection
    if len(athr) == 1:
        athr_str = 'author1%3A' + ('%2B').join(athr[0].split())
        url += '%20' + athr_str
    if len(athr) == 2:
        athr_str = 'author1%3A' + ('%2B').join(athr[0].split()) + '%20author2%3A' + ('%2B').join(athr[1].split())
        url += '%20' + athr_str

	## date range string
    date_str = 'limit_from%3A' + start_date + '%20limit_to%3A' + end_date
    url += '%20' + date_str

	## fixed formatting
    num_page_results = 75
    url += '%20numresults%3A' + str(num_page_results) + '%20format_result%3Acondensed' + '%20sort%3Arelevance-rank'

    print(url)
	## lists to store date
    titles = []
    author_lists = []
    urls = []

	### once the string has been built, access site

	# initialize number of pages to loop through
    page = 0
	## loop through other pages of search if they exist
    while True:
        # keep user aware of status
        print('Fetching search results {:d} to {:d}...'.format(num_page_results*page+1, num_page_results*(page+1)))
        # access url and pull html data
        if page == 0:
            url_response = requests.post(url)
            html = bs(url_response.text, features='html.parser')
            # find out how many results there are, and make sure don't pull more than user wants
            num_results_text = html.find('div', attrs={'class': 'highwire-search-summary'}).text.strip().split()[0]
            if num_results_text == 'No':
                print('No results found matching search criteria.')
                return()
                
            num_results = int(num_results_text)
            num_fetch_results = min(max_records, num_results)
        else:
            page_url = url + '?page=' + str(page)
            url_response = requests.post(page_url)
            html = bs(url_response.text, features='html.parser')
        # list of articles on page
        articles = html.find_all(attrs={'class': 'search-result'})
        
        ## pull details from each article on page
        titles += [article.find('span', attrs={'class': 'highwire-cite-title'}).text.strip() if article.find('span', attrs={'class': 'highwire-cite-title'}) is not None else None for article in articles]
        author_lists += [[author.text for author in article.find_all('span', attrs={'class': 'highwire-citation-author'})] for article in articles]
		
        urls = ['http://www.{:s}.org'.format(journal) + article.find('a', href=True)['href'] for article in articles]
		## see if too much time has passed or max number of records reached or no more pages
        if time.time() - overall_time > max_time or (page+1)*num_page_results >= num_fetch_results:
            break

        page += 1

	## only consider desired number of results
    records_data = list(zip(*list(map(lambda dummy_list: dummy_list[0:num_fetch_results], [titles, author_lists, urls]))))
    full_records_df = pd.DataFrame(records_data,columns=['title', 'authors', 'url'])

	## keep user informed on why task ended
    if num_results > max_records:
        print('Max number of records ({:d}) reached. Fetched in {:.1f} seconds.'.format(max_records, time.time() - overall_time))
    elif time.time() - overall_time > max_time:
        print('Max time ({:.0f} seconds) reached. Fetched {:d} records in {:.1f} seconds.'.format(max_time, num_fetch_results, time.time() - overall_time))
    else:
        print('Fetched {:d} records in {:.1f} seconds.'.format(num_fetch_results, time.time() - overall_time))
		## check if abstracts are to be pulled
    if abstracts:
        print('Fetching abstracts for {:d} papers...'.format(len(full_records_df)))
        full_records_df['abstract'] = [bs(requests.post(paper_url).text, features='html.parser').find('div', attrs={'class': 'section abstract'}).text.replace('Abstract','').replace('\n','') for paper_url in full_records_df.url]
        cols += ['abstract']
        print('Abstracts fetched.')

    try: 
        path = os.path.abspath(os.getcwd()) + '/specialized_docs'
        os.makedirs(path, exist_ok = True) 
        print("Directory '%s' created successfully" % path) 
    except OSError as error: 
        print("Directory '%s' can not be created" % path) 

    print('Downloading {:d} PDFs to {:s}...'.format(len(full_records_df), path))
    pdf_urls = [''.join(url) + '.full.pdf' for url in full_records_df.url] # list of urls to pull pdfs from

	# create filenames to export pdfs to
	# currently setup in year_lastname format
    pdf_lastnames_full = ['_'.join([name.split()[-1] for name in namelist]) for namelist in full_records_df.authors] # pull out lastnames only
    pdf_lastnames = [name if len(name) < 200 else name.split('_')[0] + '_et_al' for name in pdf_lastnames_full] # make sure file names don't get longer than ~200 chars
    pdf_paths = [''.join(lastname) + '.pdf' for lastname in zip(pdf_lastnames)] # full path for each file
    # export pdfs
    for paper_idx in range(len(pdf_urls)):
        response = requests.get(pdf_urls[paper_idx])
        file = open(os.path.join(path,pdf_paths[paper_idx]), 'wb')
        file.write(response.content)
        file.close()
        gc.collect()
    print('Download complete.')

	## create dataframe to be returned
    records_df = full_records_df[cols]
	
    print('Total time to fetch and manipulate records was {:.1f} seconds.'.format(time.time() - overall_time))

	## return the results
    return(records_df)

#Parsers
def create_db(query, query2):
    """
    Creates a database from scraped articles and PDFs based on given queries.

    This function performs the following steps:
    1. Scrapes articles from arXiv, bioRxiv, and PubMed using the provided queries.
    2. Loads the scraped documents.
    3. Splits the loaded documents into chunks.
    4. Initializes a Chroma database with the chunked documents and embeddings.

    Parameters:
    query (str): The keyword to search for in PubMed articles.
    query2 (str): The keyword to search for in arXiv and bioRxiv articles.

    Returns:
    None

    Example:
    >>> create_db('machine learning', 'artificial intelligence')
    This will scrape articles related to 'machine learning' from PubMed and 'artificial intelligence' from arXiv and bioRxiv,
    load the documents, split them into chunks, and create a Chroma database.

    Notes:
    - The function assumes that the arxivscrape, biorxiv_scrape, and pubmedscrape functions are defined and properly importable.
    - Ensure that the HuggingFaceEmbeddings and Chroma configurations are correct for your use case.
    - The Chroma database is created with cosine similarity as the distance metric.

    """
    print('creating database (this might take a while)')
    arxivscrape(query2)
    biorxiv_scrape(query2)
    pubmedscrape(query)
    
    local = os.getcwd()  ## Get local dir
    os.chdir(local)      ## shift the work dir to local dir
    print('\nWork Directory: {}'.format(local))

    #%% Phase 1 - Load DB
    embeddings_model = HuggingFaceEmbeddings(
        model_name='BAAI/bge-base-en-v1.5')

#%% Phase 1 - Load documents
    path_docs = './specialized_docs/'

    print('\nDocuments loading from:',path_docs)
    text_loader_kwargs={'autodetect_encoding': True}
    loader = DirectoryLoader(path_docs, glob="**/*.pdf", loader_cls=UnstructuredPDFLoader, 
                          loader_kwargs=text_loader_kwargs, show_progress=True,
                          use_multithreading=True)
    docs_data = loader.load()  
    print('\nDocuments loaded...')

#%% Phase 2 - Split the text
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    persist_directory = "./custom_dbs_fullScale_cosine/"

## User input ::
    arr_chunk_size = [700] #Chunk size 
    arr_chunk_overlap = [200] #Chunk overlap

    for i in range(len(arr_chunk_size)):
        for j in range(len(arr_chunk_overlap)):
        
            text_splitter = RecursiveCharacterTextSplitter(chunk_size = arr_chunk_size[i], 
                                                        chunk_overlap = arr_chunk_overlap[j], 
                                                        separators=[" ", ",", "\n", ". "])
            data_splits = text_splitter.split_documents(docs_data)
            
            print('\nDocuments split into chunks...')
        
        #%% Phase 2 - Split the text
            print('\nInitializing Chroma Database...')
            db_name = "custom_DB_cosine_cSize_%d_cOver_%d" %(arr_chunk_size[i], arr_chunk_overlap[j])
        
            p2_2 = subprocess.run('mkdir  %s/*'%(persist_directory+db_name), shell=True)
            _client_settings = chromadb.PersistentClient(path=(persist_directory+db_name))
        
            vectordb = Chroma.from_documents(documents = data_splits, 
                                    embedding = embeddings_model, 
                                    client = _client_settings,
                                    collection_name = db_name,
                                    collection_metadata={"hnsw:space": "cosine"})
        
            print('Completed Chroma Database: ', db_name)
            del vectordb, text_splitter, data_splits


# This code appears to be unused - JP
def fetch_details(pmid):
    """
    Fetches detailed information for a given PubMed article using its PMID.

    This function uses the NCBI Entrez E-utilities to fetch detailed information 
    about a PubMed article specified by its PMID. The details are returned in XML format.

    Parameters:
    pmid (str): The PubMed ID of the article to fetch details for.

    Returns:
    dict: A dictionary containing detailed information about the PubMed article.

    Example:
    >>> details = fetch_details('12345678')
    >>> print(details)
    {'PubmedArticle': [...], 'PubmedBookArticle': [...], ...}

    Notes:
    - Ensure you have set up your email address with Entrez before making requests to NCBI.
    - The returned dictionary contains various fields with detailed information about the article,
      including title, authors, abstract, and more.

    """
    handle = Entrez.efetch(db='pubmed', id=pmid, retmode = 'xml')
    records = Entrez.read(handle)
    handle.close()
    return records

def webScraping2(query):
    process = {}
    if 'ARXIV' in query.upper().split(' '):
        output = 'searching on arxiv'
        print(output)
        arxivscrape(query)
        process['searched'] = 'ARXIV'
    elif 'BIORXIV' in query.upper().split(' '):
        output = 'searching on bioRxiv'
        print(output)
        biorxiv_scrape(query)
        process['searched'] = 'BIORXIV'
    elif 'PUBMED' in query.upper().split(' '):
        output = 'searching on PubMed'
        print(output)
        pubmedscrape(query)
        process['searched'] = 'PUBMED'
    else:
        output = 'by default, searching on PUBMED'
        print(output)
        pubmedscrape(query)
        process['searched'] = 'PUBMED'
    return output, process['searched']


'''
def arxiv(query):
    """
    Scrape research papers from the arXiv repository based on a query and download the PDFs.

    Parameters:
    query (str): The search term to filter papers by their abstracts.

    Functionality:
    1. Initializes the scraper with specific parameters for the q-bio category and a specified date range.
    2. Uses the query to filter papers by their abstracts.
    3. Scrapes the metadata of the filtered papers and stores it in a DataFrame.
    4. Extracts the IDs of the papers from the DataFrame.
    5. Creates a directory named 'specialized_docs' in the current working directory to store the downloaded PDFs.
    6. Iterates over the list of paper IDs, constructs the PDF URLs, and downloads the PDFs.
    7. Saves each PDF in the 'specialized_docs' directory.
    8. Handles connection errors during the download process.

    Requirements:
    - arxivscraper: A module to scrape arXiv metadata.
    - requests_html: Used for handling HTTP sessions and downloading PDFs.
    - pandas: For handling data in DataFrame format.
    - os: For creating directories and handling file paths.

    Example:
    arxivscrape('machine learning')
    This will scrape papers related to 'machine learning' in their abstracts within the specified category and date range, and download their PDFs.

    Notes:
    - Ensure you have the necessary packages installed: arxivscraper, requests_html, pandas.
    - Modify the category and date range in the Scraper initialization as needed.
    """
    scraper = arxivscraper.Scraper(category='q-bio', date_from='2024-04-01',date_until='2024-05-01',t=10, filters={'abstract':[query]})
    output = scraper.scrape()
    cols = ('id', 'title', 'categories', 'abstract', 'doi', 'created', 'updated', 'authors')
    df = pd.DataFrame(output,columns=cols)
    id_list = df['id'].to_list()
    s = HTMLSession()

    headers = {'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36'}

    #Create directory to store pdfs
    try:
        path = os.path.abspath(os.getcwd()) + '/specialized_docs'
        os.makedirs(path, exist_ok = True) 
        print("Directory '%s' created successfully" % path)
    except OSError as error:
        print("Directory '%s' can not be created" % path)

    #Scrape arxiv and
    for ids in id_list:
        try:
            base_url = 'https://arxiv.org/pdf/'
            pdf_url = base_url+ids
            print(pdf_url)
            r = s.get(pdf_url, stream=True)
            print(r)
            with open(os.path.join(path, ids + '.pdf'), 'wb') as f:
                for chunk in r.iter_content(chunk_size = 1024):
                    if chunk:
                        f.write(chunk)
                    
        except ConnectionError as e:
            pass
            print(f"{ids} could not be gathered.")
'''

def arxiv_search(query, count):
    #get the url
    split_query = query.split()
    url = "https://arxiv.org/search/?searchtype=all&query="
    for term in split_query:
        url = url+term+"+"
    url = url[:-1]+"&abstracts=show&size=50&order="
    print(url)
    try: 
        path = os.path.abspath(os.getcwd()) + '/arxiv'
        os.makedirs(path, exist_ok = True) 
        print("Directory '%s' created successfully" % path) 
    except OSError as error: 
        print("Directory '%s' can not be created" % path) 

    # query the website and return the html to the variable 'page'
    page = requests.get(url)
    # parse the html using beautiful soup and store in variable 'soup'
    soup = bs(page.content, 'html.parser')
    paper_block = soup.find_all(class_='arxiv-result')
    paper_list = []
    arxiv_urls = []
    i = 0
    for paper in paper_block:
        arxiv_title = paper.find_all(class_='title is-5 mathjax')
        arxiv_authors = paper.find_all(class_='authors')
        paper_authors = [author.get_text(strip=True) for author in arxiv_authors]
        arxiv_abstracts = paper.find_all(class_ = 'abstract-short has-text-grey-dark mathjax')
        arxiv_results = paper.find_all(class_='list-title is-inline-block')
        # Assuming URLs are within <a> tags inside the arxiv-result class
        for result in arxiv_results:
        # Assuming URLs are within <a> tags inside the arxiv-result class
            for a_tag in result.find_all('a', href=True):
                full_url = urljoin(url, a_tag['href'])
                arxiv_urls.append(full_url)
        arxiv_title = result_set_to_string(arxiv_title)
        arxiv_abstracts = result_set_to_string(arxiv_abstracts)
        arxiv_authors = ', '.join(paper_authors)
        arxiv_authors = arxiv_authors[8:]
        paper_list.append({'Title': arxiv_title, 'Authors': arxiv_authors, 'Abstract': arxiv_abstracts})
        i += 1
        if i >= count:
            break
    df = pd.DataFrame(paper_list)
    pdf_urls = [attempt for attempt in arxiv_urls if 'pdf' in attempt.lower()]
    return df, pdf_urls
    
def arxiv_scrape(pdf_urls):
    s = HTMLSession()

    headers = {'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36'}
    for papers in pdf_urls:
        try:
            r = s.get(papers, stream=True)
            paper_id = papers[-10:]
            with open(os.path.join(path, paper_id + '.pdf'), 'wb') as f:
                for chunk in r.iter_content(chunk_size = 1024):
                    if chunk:
                        f.write(chunk)
                    
        except ConnectionError as e:
            pass
            print(f"{pmc} could not be gathered.")
            
def result_set_to_string(result_set):
    """
    Converts a BeautifulSoup ResultSet to a single string.
    
    Args:
        result_set (bs4.element.ResultSet): The ResultSet to convert.
        
    Returns:
        str: A string containing the text content of all elements in the ResultSet.
    """
    return ' '.join([element.get_text(strip=True) for element in result_set])

def parse_llm_response(response):
    """
    Parses the LLM response to extract the database name and search terms.
    
    Parameters:
    response (str): The response from the LLM.
    
    Returns:
    dict: A dictionary with the database name and a list of search terms.
    """
    # Initialize an empty dictionary to hold the parsed data
    parsed_data = {}

    # Split the response into lines
    lines = response.strip().split('\n')

    # Extract the database name
    database_line = lines[0].replace("Database:", "").strip()
    parsed_data["database"] = database_line

    # Extract the search terms
    search_terms_line = lines[1].replace("Search Terms:", "").strip()
    search_terms = [term.strip() for term in search_terms_line.split(',')]
    parsed_data["search_terms"] = search_terms

    return parsed_data
