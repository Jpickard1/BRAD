'''Find the first order interactions of a gene in the Hardwired Genome

Usage:
    python <path/to/script/>hwgFirstOrderInteractions.py <gene name>

This script uses the Hardwired Genome construction. Note: Cooper recently found a bug in how the data are integrated for the Hardwired Genome.
The code here should side step that by using a different download from Ensembl (GRCh38.p13.tsv) for mapping between gene names and ENSG IDs, but
the issue may persist.

Arguments:
    gene name: this can either be a gene name in its symbol or common name or it can be a ENSG ID
'''
import pandas as pd
import scipy.io as sio
import numpy as np
from scipy.sparse import csc_matrix
import sys

def main():
    geneName = sys.argv[1]
    HWG = sio.loadmat('/home/jpic/RAG-DEV/tutorials/planner/HWG.mat')            # Load Hardwired Genome A matrix
    A = HWG['HWG']['A']
    A = A[0,0].todense()
    df = pd.read_csv('/home/jpic/RAG-DEV/tutorials/planner/geneIndexTable.csv')  # Load gene index table
    df2 = pd.read_csv('/home/jpic/RAG-DEV/tutorials/planner/GRCh38.p13.tsv', delimiter='\t')                          # Load GRCH38 to map between gene names and ENSG IDs
    if df[df['Stable ID'].isin([geneName])].shape[0] == 0:
        temp0 = df2[df2['Gene name'] == geneName]                                    # Find ENSG of Gene Name
        ENSG1 = temp0['Gene stable ID'].values[0]
    else:
        ENSG1 = geneName
    idx = df[df['Stable ID'] == ENSG1].index[0]                                  # Find position of target gene in HWG
    idxs = np.where(A[idx,:] == 1)[1]                                            # Find first order interactions in network
    firstOrderEnsg = list(df.iloc[idxs]['Stable ID'].values)                     # Get ENSG of first order interactions
    firstOrderGenes = list(set(df2[df2['Gene stable ID'].isin(firstOrderEnsg)]['Gene name'].values)) # Get gene names of first order contacts
    dfSave = pd.DataFrame({'Genes':firstOrderGenes})
    dfSave.to_csv('HWG_first_order_interactions.csv', index=False)

if __name__ == "__main__":
    main()
