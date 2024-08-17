import pandas as pd
import numpy as np
from copy import deepcopy
import os
import sys
from importlib import reload
from scipy.stats import zscore
from scipy.stats import entropy
import scipy.io
import scipy
import textwrap
from scipy import sparse
import importlib
from itertools import product
from datetime import datetime
from IPython.display import display # displaying dataframes
import string
import warnings
import re
import matplotlib.pyplot as plt

# Bioinformatics
import gget

from BRAD import utils
from BRAD import log

# gene enrichment


def queryBlast(chatstatus):
    """
    Queries the BLAST database for DNA sequences provided in the `chatstatus` and updates the status with results.

    This function extracts a prompt from the `chatstatus` dictionary, cleans the prompt by removing 
    unnecessary punctuation, and then performs BLAST searches on DNA sequences found in the cleaned prompt. 
    It compiles the results into a DataFrame and appends it to the `chatstatus` under the 'process' key.

    Parameters:
    chatstatus (dict): A dictionary containing the current status of the chat and other metadata. 
                       It should have the following structure:
                       {
                           'prompt': str,  # A string containing DNA sequences to query
                           'process': {   # A dictionary containing processing steps
                               'steps': list  # A list of steps performed during processing
                           }
                       }
    query (str): A query string. This parameter is currently unused in the function but is part of the function signature.

    Returns:
    dict: The updated `chatstatus` dictionary, which includes the results of the BLAST queries appended to the 'process' key.

    Raises:
    None

    Warnings:
    - If no results are found after querying, a warning is issued with the message 'No Results Found'.
    """
    # Auth: Marc Andrew Choi
    #       machoi@umich.edu
    # Date: July 30, 2024
    prompt = chatstatus['prompt']
    

    # Remove any punctuation except for - and _, which are used in gget database names
    punctuation_to_remove = string.punctuation.replace('-', '').replace('_', '')
    translator = str.maketrans('', '', punctuation_to_remove)
    prompt = prompt.translate(translator)
    result_df = pd.DataFrame()
    for dnaseq in prompt.split(' '):
        if dnaseq.upper():
            try:
                dna_df = gget.blast(dnaseq)
                dna_df['search'] = dnaseq
                result_df = pd.concat([result_df, dna_df], ignore_index=True)
            except:
                warnings.warn(dnaseq + " is not a valid search term")

    if result_df.empty is None:
        warnings.warn('No Results Found')

    log.userOutput(result_df, chatstatus=chatstatus)
    chatstatus['process']['steps'].append(
                                            {
                                                  'result db' : result_df,
                                            }
                                         )
    
    return chatstatus
