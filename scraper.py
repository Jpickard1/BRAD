import subprocess
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
import chromadb
import arxivscraper
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
import requests
from bs4 import BeautifulSoup as bs
from requests_html import HTMLSession
import requests
from requests.exceptions import ConnectionError

def webScraping(query):
    searched = None
    if 'ARXIV' in query.upper().split(' '):
        print('searching on arxiv')
        arxivscrape(query)
        searched = 'ARXIV'
    elif 'BIORXIV' in query.upper().split(' '):
        print('searching on bioRxiv')
        biorxiv_scrape(query)
        searched = 'BIORXIV'
    elif 'PUBMED' in query.upper().split(' '):
        print('searching on PubMed')
        pubmedscrape(query)
        searched = 'PUBMED'
    else:
        print('by default, searching on PUBMED')
        pubmedscrape(query)
        searched = 'PUBMED'
    return searched
    
def arxivscrape(query):
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
        #r = s.get(base_url + ids + '/', headers = headers, timeout = 5)
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

def pubmedscrape(query):
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
    abstract_arr = []
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
                pdf_real = 'https://ncbi.nlm.nih.gov'+r.html.find('a.int-view', first=True).attrs['href']
                print(pdf_real)
                r = s.get(pdf_real, stream=True)
                with open(os.path.join(path, pmc + '.pdf'), 'wb') as f:
                    for chunk in r.iter_content(chunk_size = 1024):
                        if chunk:
                            f.write(chunk)
                    
            except ConnectionError as e:
                pass
                print(f"{pmc} could not be gathered.")

    else:
        print("no articles found")
    print("pdf collection complete!")


def real_search(start_date  = datetime.date.today().replace(day=1), 
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
    >>> df = real_search(start_date=datetime.date(2023, 1, 1), end_date=datetime.date(2023, 1, 31), subjects=['neuroscience'], kwd=['brain'], max_records=10)
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
            print(page_url)
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

def biorxiv_scrape(query):
    """
    Scrapes the bioRxiv preprint server for articles matching a specific query.

    This function uses the `real_search` method to search bioRxiv for articles
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
    real_search(start_date  = datetime.date.today().replace(day=1), 
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