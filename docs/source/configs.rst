Each `Agent` is configured to interact with core and tool modules according to certain specifications. These configurations
control different features, such as how the RAG pipeline works with different search or compression mechanisms, or how many
articles are downloaded from each online repository. The default configuration parameters for each module are in `config.json`,
but the default values can be overwritten for individual agents by supplying the parameters in the `Agent` constructor.

The configuration parameters should be stored in a `JSON` file organized as:

>>> {
...     "log_path": "<path/to/output-directories>/BRAD",  // output location
...     "image-path-extension": "images"                  // output location for images
...     "RAG": {  // LAB NOTEBOOK
...         "num_articles_retrieved": 3,   // number of articles during retrieval
...         "multiquery": false,           // retrieval with single user quiery or
...                                        // multiple llm generated multiqueries
...         "contextual_compression": false,  // using contextual compression or raw 
...                                           // chunks during generation
...         "rerank": false,    // reranking documents after retrieval
...         "similarity": true, // search algorithm for selecting document 
...                             // during retrieval
...         "mmr": true,        // one of similarity or mmr must be true
...         ...
...     },
...     "SCRAPE": { // DIGITAL LIBRARY: web scraping from arXiv, bioRxiv, and PubMed
...         "add_from_scrape": true,     // add documents to the LAB NOTEBOOK database
...         "max_search_terms": 10       // number of terms to search on the archives
...         "max_articles_download": 10, // how many articles to download
...         ...
...     },
...     "DATABASE": { // DIGITAL LIBRARY: searching enrichr, gene ontology, etc.
...         "max_search_terms": 100   // maximum number of search terms (genes or
...                                   // other) to query at once
...         "max_enrichr_pval": 0.5,  // database specific parameters for search
...         ...
...     },
...     "SOFTWARE": { // SOFTWARE: path to find available software for BRAD to run
...         "py-path": "/home/jpic/BRAD-Tools/",
...     },
...     "display": {  // General display parameters
...         "num_df_rows_display": 3,
...         "dpi": 300,
...         "figsize": [ 5, 3 ],
...         "colormap": "viridis",
...         ...
...     },
>>> }
