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

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
import chromadb
import subprocess

from langchain_community.document_loaders import PyPDFLoader
from langchain.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from BRAD.promptTemplates import scrapeTemplate
from BRAD import utils
from BRAD import log

def webScraping(chatstatus):
    """
    Performs web scraping based on the provided chat status, executing specific scraping functions for different sources like arXiv, bioRxiv, and PubMed.

    :param chatstatus: The status of the chat, containing information about the current prompt and configuration.
    :type chatstatus: dict

    :raises None: This function does not raise any specific errors.

    :return: The updated chat status after executing the web scraping process.
    :rtype: dict

    """
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: May 20, 2024
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
    template = scrapeTemplate()
    template = template.format(search_terms=chatstatus['search']['used terms'])
    PROMPT = PromptTemplate(input_variables=["history", "input"], template=template)
    conversation = ConversationChain(prompt  = PROMPT,
                                     llm     = llm,
                                     verbose = chatstatus['config']['debug'],
                                     memory  = memory,
                                    )
    response = conversation.predict(input=query)
    llmResponse = parse_llm_response(response)
    log.debugLog(llmResponse, chatstatus=chatstatus)
    try:
        llmType = str(llm.model)
    except:
        try:
            llmType = str(llm.model_name)
        except:
            llmType = str(llm)
    chatstatus['process']['steps'].append(log.llmCallLog(llm             = llmType,
                                                         prompt          = PROMPT,
                                                         memory          = memory,
                                                         input           = query,
                                                         output          = response,
                                                         parsedOutput    = llmResponse,
                                                         purpose         = 'identify how to web scrape'
                                                        )
                                         )
    llmKey, searchTerms = llmResponse['database'].upper(), llmResponse['search_terms']

    # Determine the target source
    source = next((key for key in scraping_functions if key == llmKey), 'PUBMED')
    process = {'searched': source}
    scrape_function = scraping_functions[source]
    
    # Execute the scraping function and handle errors
    if chatstatus['config']['SCRAPE']['perform_search']:
        try:
            output = f'searching on {source}...'
            log.debugLog(output, chatstatus=chatstatus)
            log.debugLog('Search Terms: ' + str(searchTerms), chatstatus=chatstatus)
            for numTerm, st in enumerate(searchTerms):
                if numTerm == chatstatus['config']['SCRAPE']['max_search_terms']:
                    break
                scrape_function(st, chatstatus)
        except Exception as e:
            output = f'Error occurred while searching on {source}: {e}'
            log.debugLog(output, chatstatus=chatstatus)
            process = {'searched': 'ERROR'}
    
        if chatstatus['config']['SCRAPE']['add_from_scrape']:
            chatstatus = updateDatabase(chatstatus)
        
        chatstatus['process']['steps'].append(process)
        chatstatus['output'] = "Articles were successfully downloaded."
    else:
        chatstatus['output'] = "No articles were searched."        
    return chatstatus


  
def arxiv(query, chatstatus):
    """
    Searches for articles on the arXiv repository based on the given query, displays search results, and optionally downloads articles as PDFs.

    :param query: The search query for arXiv.
    :type query: str

    :raises None: This function does not raise any specific errors.

    :return: A tuple containing the output message and a process dictionary.
    :rtype: tuple

    """
    # Auth: Marc Choi
    #       machoi@umich.edu
    process = {}
    output = 'searching the following on arxiv: ' + query
    chatstatus = log.userOutput(output, chatstatus=chatstatus)
    df, pdfs = arxiv_search(query, 10, chatstatus=chatstatus)
    process['search results'] = df
    displayDf = df[['Title', 'Authors', 'Abstract']]
    display(displayDf)
    if chatstatus['config']['SCRAPE']['save_search_results']:
        utils.save(chatstatus, df, "arxiv-search-" + str(query) + '.csv')
    if len(chatstatus['queue']) == 0:
        if chatstatus['interactive']:
            output += '\n Would you like to download these articles [Y/N]?'
            chatstatus = log.userOutput('Would you like to download these articles [Y/N]?', chatstatus=chatstatus)
            download = input().strip().upper()
            chatstatus['process']['steps'].append(
                {
                    'func'           : 'scraper.arxiv',
                    'prompt to user' : 'Do you want to proceed with this plan? [Y/N/edit]',
                    'input'          : download,
                    'purpose'        : 'decide to download pdfs or not'
                }
            )
        else:
            download = chatstatus['SCRAPE']['download_search_results']
    else:
        download = 'Y'
    process['download'] = (download == 'Y')
    if download == 'Y':
        output += arxiv_scrape(pdfs, chatstatus)
    return output, process



def search_pubmed_article(query, number_of_articles=10, chatstatus=None):
    """
    Searches PubMed for articles matching the specified query and retrieves their PMIDs.

    :param query: The keyword or phrase to search for in PubMed articles.
    :type query: str
    :param number_of_articles: The maximum number of article PMIDs to return. Defaults to 10.
    :type number_of_articles: int

    :raises None: This function does not raise any specific errors.

    :return: A list of PMIDs for articles matching the query.
    :rtype: list

    """
    # Auth: Marc Choi
    #       machoi@umich.edu
    Entrez.email = 'inhyak@gmail.com'
    log.debugLog("search_pubmed_article", chatstatus=chatstatus)
    log.debugLog(f'term={query}', chatstatus=chatstatus)
    handle = Entrez.esearch(db='pubmed', term = query, retmax=number_of_articles, sort='relevance')
    log.debugLog(f'handle={str(handle)}', chatstatus=chatstatus)
    record = Entrez.read(handle)
    log.debugLog(f'record={str(record)}', chatstatus=chatstatus)
    handle.close()
    return record['IdList']

def pubmed(query, chatstatus):
    """
    Scrapes PubMed for articles matching the query, retrieves their PMIDs, and downloads available PDFs.

    :param query: The keyword to search for in PubMed articles.
    :type query: str

    :raises None: This function does not raise any specific errors.

    :return: None
    :rtype: None
    """
    # Auth: Marc Choi
    #       machoi@umich.edu

    pmid_list = search_pubmed_article(query, chatstatus=chatstatus)
    citation_arr = []
    if pmid_list:
        for pmid in pmid_list:
            citation = pmid
            citation_arr.append(citation)
        s = HTMLSession()

        headers = {'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36'}

        try: 
            path = utils.pdfDownloadPath(chatstatus) # os.path.abspath(os.getcwd()) + '/specialized_docs'
            os.makedirs(path, exist_ok = True)
            log.debugLog("Directory '%s' created successfully" % path, chatstatus=chatstatus)
        except OSError as error: 
            log.debugLog("Directory '%s' can not be created" % path, chatstatus=chatstatus)
        for pmc in citation_arr:
            try:
                base_url = 'https://pubmed.ncbi.nlm.nih.gov/'
                r = s.get(base_url + pmc + '/', headers = headers, timeout = 5)
                if r.html.find('a.id-link', first=True) is not None:
                    pdf_url = r.html.find('a.id-link', first=True).attrs['href']
                    if 'ncbi.nlm.nih.gov' not in pdf_url:
                        continue
                    r = s.get(pdf_url, headers = headers, timeout = 5)
                    try:
                        ending = r.html.find('a.int-view', first=True).attrs['href']
                        pdf_real = 'https://ncbi.nlm.nih.gov'+ending
                        r = s.get(pdf_real, stream=True, timeout = 5)
                        with open(os.path.join(path, pmc + '.pdf'), 'wb') as f:
                            for chunk in r.iter_content(chunk_size = 1024):
                                if chunk:
                                    f.write(chunk)
                    except AttributeError as e:
                        print(e)
                        log.debugLog(f"{pmc} could not be gathered.", chatstatus=chatstatus)
                        pass                
                    
            except ConnectionError as e:
                pass
                log.debugLog(f"{pmc} could not be gathered.", chatstatus=chatstatus)

    else:
        log.debugLog("no articles found", chatstatus=chatstatus)
    log.debugLog("pdf collection complete!", chatstatus=chatstatus)

def biorxiv(query, chatstatus):
    """
    Scrapes the bioRxiv preprint server for articles matching a specific query.

    :param query: The keyword to search for in bioRxiv articles.
    :type query: str

    :raises None: This function does not raise any specific errors.

    :return: None
    :rtype: None

    """
    # Auth: Marc Choi
    #       machoi@umich.edu

    biorxiv_real_search(chatstatus  = chatstatus,
                        start_date  = datetime.date.today().replace(year=2015), 
                        end_date    = datetime.date.today(),
                        subjects    = [], 
                        journal     = 'biorxiv',
                        kwd         = [query], 
                        kwd_type    = 'all', 
                        athr        = [], 
                        max_records = 10, 
                        max_time    = 300,
                        cols        = ['title', 'authors', 'url'],
                        abstracts   = False
                        )

def biorxiv_real_search(chatstatus,
                        start_date  = datetime.date.today().replace(year=2015), 
                        end_date    = datetime.date.today(), 
                        subjects    = [], 
                        journal     = 'biorxiv',
                        kwd         = [], 
                        kwd_type    = 'all', 
                        athr        = [], 
                        max_records = 10, 
                        max_time    = 300,
                        cols        = ['title', 'authors', 'url'],
                        abstracts   = False
                        ):

    """
    Searches for articles on arXiv, bioRxiv, or PubMed based on the given queries and creates a database from the scraped articles and PDFs.

    :param start_date: The start date for the search query. Defaults to today's date.
    :type start_date: datetime.date
    :param end_date: The end date for the search query. Defaults to today's date.
    :type end_date: datetime.date
    :param subjects: The subjects to search for in the specified journal. Defaults to an empty list.
    :type subjects: list
    :param journal: The journal to search for articles. Defaults to 'biorxiv'.
    :type journal: str
    :param kwd: The keywords to search for in the abstract or title. Defaults to an empty list.
    :type kwd: list
    :param kwd_type: The type of keyword search to perform. Defaults to 'all'.
    :type kwd_type: str
    :param athr: The authors to search for in the articles. Defaults to an empty list.
    :type athr: list
    :param max_records: The maximum number of records to fetch. Defaults to 75.
    :type max_records: int
    :param max_time: The maximum time (in seconds) to spend fetching records. Defaults to 300.
    :type max_time: int
    :param cols: The columns to include in the database. Defaults to ['title', 'authors', 'url'].
    :type cols: list
    :param abstracts: Whether to include abstracts in the database. Defaults to False.
    :type abstracts: bool

    :raises None: This function does not raise any specific errors.

    :return: The DataFrame containing the records fetched and processed.
    :rtype: pd.DataFrame

    """
    # Auth: Marc Choi
    #       machoi@umich.edu

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
        kwd_string = ' '.join(kwd)
        intermediate_string = kwd_string.replace(' ', '%2B')
        kwd_str = 'abstract_title%3A' + intermediate_string
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
    num_page_results = max_records
    url += '%20numresults%3A' + str(num_page_results) + '%20format_result%3Acondensed' + '%20sort%3Arelevance-rank'
    
    log.debugLog(url, chatstatus=chatstatus)

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
        log.debugLog('Fetching search results {:d} to {:d}...'.format(num_page_results*page+1, num_page_results*(page+1)), chatstatus=chatstatus)
        # access url and pull html data
        if page == 0:
            url_response = requests.post(url)
            html = bs(url_response.text, features='html.parser')
            # find out how many results there are, and make sure don't pull more than user wants
            num_results_text = html.find('div', attrs={'class': 'highwire-search-summary'}).text.strip().split()[0]
            if num_results_text == 'No':
                log.debugLog("No results found matching search criteria.", chatstatus=chatstatus)
                return()

            num_results_text = num_results_text.replace(',', '')
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
        log.debugLog('Max number of records ({:d}) reached. Fetched in {:.1f} seconds.'.format(max_records, time.time() - overall_time), chatstatus=chatstatus)
    elif time.time() - overall_time > max_time:
        log.debugLog('Max time ({:.0f} seconds) reached. Fetched {:d} records in {:.1f} seconds.'.format(max_time, num_fetch_results, time.time() - overall_time), chatstatus=chatstatus)
    else:
        log.debugLog('Fetched {:d} records in {:.1f} seconds.'.format(num_fetch_results, time.time() - overall_time), chatstatus=chatstatus)
		## check if abstracts are to be pulled
    if abstracts:
        log.debugLog('Fetching abstracts for {:d} papers...'.format(len(full_records_df)), chatstatus=chatstatus)
        full_records_df['abstract'] = [bs(requests.post(paper_url).text, features='html.parser').find('div', attrs={'class': 'section abstract'}).text.replace('Abstract','').replace('\n','') for paper_url in full_records_df.url]
        cols += ['abstract']
        log.debugLog('Abstracts fetched.', chatstatus=chatstatus)

    try: 
        path = utils.pdfDownloadPath(chatstatus) # os.path.abspath(os.getcwd()) + '/specialized_docs'
        os.makedirs(path, exist_ok = True) 
        log.debugLog("Directory '%s' created successfully" % path, chatstatus=chatstatus)
    except OSError as error: 
        log.debugLog("Directory '%s' can not be created" % path, chatstatus=chatstatus)
    log.debugLog('Downloading {:d} PDFs to {:s}...'.format(len(full_records_df), path), chatstatus=chatstatus)
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
    chatstatus = log.userOutput("Download complete.", chatstatus=chatstatus)

	## create dataframe to be returned
    records_df = full_records_df[cols]
	
    chatstatus = log.userOutput('Total time to fetch and manipulate records was {:.1f} seconds.'.format(time.time() - overall_time), chatstatus=chatstatus)

	## return the results
    return(records_df)

#Parsers
def create_db(query, query2):
    """
    Creates a database from scraped articles and PDFs based on given queries.

    :param query: The keyword to search for in PubMed articles.
    :param query2: The keyword to search for in arXiv and bioRxiv articles.
    :type query: str
    :type query2: str

    :raises None: This function does not raise any specific errors.

    :return: None
    :rtype: None

    """
    # Auth: Marc Choi
    #       machoi@umich.edu

    log.debugLog('creating database (this might take a while)', chatstatus=chatstatus)
    arxivscrape(query2)
    biorxiv_scrape(query2)
    pubmedscrape(query)
    
    local = os.getcwd()  ## Get local dir
    os.chdir(local)      ## shift the work dir to local dir
    log.debugLog('\nWork Directory: {}'.format(local), chatstatus=chatstatus)

    #%% Phase 1 - Load DB
    embeddings_model = HuggingFaceEmbeddings(
        model_name='BAAI/bge-base-en-v1.5')

#%% Phase 1 - Load documents
    path_docs = utils.pdfDownloadPath(chatstatus) # './specialized_docs/'
    log.debugLog('\nDocuments loading from:',path_docs, chatstatus=chatstatus)
    text_loader_kwargs={'autodetect_encoding': True}
    loader = DirectoryLoader(path_docs, glob="**/*.pdf", loader_cls=UnstructuredPDFLoader, 
                          loader_kwargs=text_loader_kwargs, show_progress=True,
                          use_multithreading=True)
    docs_data = loader.load()
    log.debugLog('\nDocuments loaded...', chatstatus=chatstatus)

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
            log.debugLog('\nDocuments split into chunks...', chatstatus=chatstatus)
        
        #%% Phase 2 - Split the text
            log.debugLog('\nInitializing Chroma Database...', chatstatus=chatstatus)
            db_name = "custom_DB_cosine_cSize_%d_cOver_%d" %(arr_chunk_size[i], arr_chunk_overlap[j])
        
            p2_2 = subprocess.run('mkdir  %s/*'%(persist_directory+db_name), shell=True)
            _client_settings = chromadb.PersistentClient(path=(persist_directory+db_name))
        
            vectordb = Chroma.from_documents(documents = data_splits, 
                                    embedding = embeddings_model, 
                                    client = _client_settings,
                                    collection_name = db_name,
                                    collection_metadata={"hnsw:space": "cosine"})
            log.debugLog('Completed Chroma Database: ' + str(db_name), chatstatus=chatstatus)
            del vectordb, text_splitter, data_splits



def arxiv_search(query, count, chatstatus=None):
    """
    Searches for articles on arXiv based on the given query and retrieves a specified number of results.

    :param query: The search query for arXiv.
    :param count: The number of search results to retrieve.
    :type query: str
    :type count: int

    :raises None: This function does not raise any specific errors.

    :return: A tuple containing a DataFrame with the search results and a list of PDF URLs.
    :rtype: tuple

    """
    # Auth: Marc Choi
    #       machoi@umich.edu

    #get the url
    split_query = query.split()
    url = "https://arxiv.org/search/?searchtype=all&query="
    for term in split_query:
        url = url+term+"+"
    url = url[:-1]+"&abstracts=show&size=50&order="
    log.debugLog(url, chatstatus=chatstatus)
    try: 
        path = os.path.abspath(os.getcwd()) + '/arxiv'
        os.makedirs(path, exist_ok = True) 
        log.debugLog("Directory '%s' created successfully" % path, chatstatus=chatstatus)
    except OSError as error: 
        log.debugLog("Directory '%s' can not be created" % path, chatstatus=chatstatus)

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
    if chatstatus['config']['debug']:
        display(df.head())
    pdf_urls = [attempt for attempt in arxiv_urls if 'pdf' in attempt.lower()]
    return df, pdf_urls
    
def arxiv_scrape(pdf_urls, chatstatus):
    """
    Downloads PDFs from a list of URLs pointing to arXiv articles.

    :param pdf_urls: A list of URLs pointing to arXiv articles in PDF format.
    :type pdf_urls: list

    :raises None: This function does not raise any specific errors.

    :return: None
    :rtype: None

    """
    # Auth: Marc Choi
    #       machoi@umich.edu

    s = HTMLSession()
    try: 
        path = utils.pdfDownloadPath(chatstatus) # os.path.abspath(os.getcwd()) + '/specialized_docs'
        os.makedirs(path, exist_ok = True) 
        log.debugLog("Directory '%s' created successfully" % path, chatstatus=chatstatus) 
    except OSError as error: 
        log.debugLog("Directory '%s' can not be created" % path, chatstatus=chatstatus)

    headers = {'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36'}
    pdf_string = ""
    for papers in pdf_urls:
        pdf_string += papers+"\n"
        try:
            r = s.get(papers, stream=True)
            paper_id = papers[-10:]
            with open(os.path.join(path, paper_id + '.pdf'), 'wb') as f:
                for chunk in r.iter_content(chunk_size = 1024):
                    if chunk:
                        f.write(chunk)
                    
        except ConnectionError as e:
            pass
            log.debugLog(f"{pmc} could not be gathered.", chatstatus=chatstatus)
    return pdf_string
            
def result_set_to_string(result_set):
    """
    Converts a BeautifulSoup result set to a string.

    :param result_set: The result set to convert to a string.
    :type result_set: bs4.element.ResultSet

    :raises None: This function does not raise any specific errors.

    :return: The string representation of the result set.
    :rtype: str
    """
    # Auth: Marc Choi
    #       machoi@umich.edu

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

def updateDatabase(chatstatus):
    """
    .. warning: This function contains hardcoded values related to text chunking
    
    Update the database with new documents based on the given chat status.

    This function determines which documents need to be added to the database, downloads them,
    splits them into chunks, and adds the formatted chunks to the specified database.

    Args:
        chatstatus (dict): The current chat status containing database information and other parameters.

    Returns:
        dict: The updated chat status after adding new documents to the database.
    """
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: June 27, 2024

    # Determine which documents need to be added to the database
    new_docs_path = utils.pdfDownloadPath(chatstatus)

    if chatstatus['databases']['RAG'] is None:
        return chatstatus
    
    if not os.path.isdir(new_docs_path):
        return chatstatus

    # Warning! these values are hard coded
    chunk_size=[700]
    chunk_overlap=[200]
    
    # Load all the documents
    text_loader_kwargs = {'autodetect_encoding': True}
    new_loader = DirectoryLoader(new_docs_path,
                                 glob="**/*.pdf",
                                 loader_cls=PyPDFLoader,
                                 show_progress=True,
                                 use_multithreading=True)
    new_docs_data = new_loader.load()
    print('\nNew documents loaded...')
    
    # Split the new document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size[0],
                                                    chunk_overlap=chunk_overlap[0],
                                                    separators=[" ", ",", "\n", ". "])
    new_data_splits = text_splitter.split_documents(new_docs_data)
    print("New document split into chunks...")

    # Format the document splits to be placed into the database
    new_data_splits
    docs, meta = [], []
    for doc in new_data_splits:
        docs.append(doc.page_content)
        meta.append(doc.metadata)

    # Add to the database
    log.debugLog('Adding texts to database', chatstatus)
    log.debugLog(f'len(docs)={len(docs)}', chatstatus)
    if len(docs) == 0:
        log.debugLog('exiting updateDatabase() because no new docs were found', chatstatus)
        return chatstatus
    chatstatus['databases']['RAG'].add_texts(texts = docs,
                                             meta  = meta)
    log.debugLog('Done adding texts to database', chatstatus)
    return chatstatus
