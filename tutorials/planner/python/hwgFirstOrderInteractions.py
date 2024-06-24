'''
Find the first order interactions of a gene in the Hardwired Genome

Arguments (three arguments):
    1. output directory: chatstatus['output-directory']
    2. output file: <name of output file>
    3. gene name: this can either be a gene name in its symbol or common name or it can be a ENSG ID

Usage:
    Command Line:
    ```
    python <path/to/script/>hwgFirstOrderInteractions.py <output path> <output file> <gene name>
    ```
                                                               |              |           |
                                                          Argument 1     Argument 2   Argument 3
    BRAD Line:
    ```
    subprocess.call([sys.executable, '<path/to/script/>/hwgFirstOrderInteractions.py', chatstatus['output-directory'], <output file>, <gene name>])
    ```

Output file name should be `S<Step number>-HWG-1stOrder-<gene name>.csv`

This script uses the Hardwired Genome construction. Note: Cooper recently found a bug in how the data are integrated for the Hardwired Genome.
The code here should side step that by using a different download from Ensembl (GRCh38.p13.tsv) for mapping between gene names and ENSG IDs, but
the issue may persist.
'''
import pandas as pd
import scipy.io as sio
import numpy as np
from scipy.sparse import csc_matrix
import sys
import os

def main():
    outputPath = sys.argv[1]
    outputFile = sys.argv[2]
    outputFile = os.path.join(outputPath, outputFile)
    geneName = sys.argv[3]

    print('******************************')
    print('Finding 1st Order Interactions')
    print('******************************')
    
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
    dfSave.to_csv(outputFile, index=False)

if __name__ == "__main__":
    main()
