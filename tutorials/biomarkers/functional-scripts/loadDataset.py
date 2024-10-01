"""
loadDataset.py is the script for selecing datasets by name. This must be the first script run. This script is a file chooser to load a gene expression dataset from the lab's turbo partition to give access to the data to BRAD.

Arguments (three arguments):
    1. output directory: chatstatus['output-directory']
    2. output file: <name of output file>
    3. dataset: this can be any of the following:
        - "2015": a bulk RNAseq time series dataset of synchronized Fibroblast proliferation
        - "2018": a bulk RNAseq time series dataset of Weintraubs Myogenic reprogramming experiment

Usage:
```
subprocess.run([sys.executable, '<path/to/script/>/loadDataset.py', chatstatus['output-directory'], <output file>, <dataset>], capture_output=True, text=True)
```

**OUTPUT FILE NAME INSTRUCTIONS**
1. Output path should be chatstatus['output-directory']
2. Output file name should be `S0-<data set name>.pkl`
"""
import os
import sys
import pandas as pd
import anndata as an
import numpy as np
import pickle

def main():
    outputPath = sys.argv[1] # chatstatus['output-directory']
    outputFile = sys.argv[2] # output file name
    dataset    = sys.argv[3]

    outputFile = os.path.join(outputPath, outputFile)
    
    print('******************************')
    print('       Dataset Selector       ')
    print('******************************')

    out_path = "/nfs/turbo/umms-indikar/shared/projects/geneformer/data/rajapakse_lab_data_jpic.h5ad"
    ad = an.read(out_path)
    
    if dataset == '2015':
        ds = 'chen_2015'
    elif dataset == '2018':
        ds = 'liu_2018'
    else:
        print('Dataset is invalid!')

    ad = ad[ad.obs['dataset'] == ds]
    print('The correct dataset was selected')

    print('******************************')
    print('  Convert to Gene Coordinates ')
    print('******************************')
    
    # Build the object in which to save the trajectories. This will be state variables by time by trajectory
    n = ad.shape[1]
    T = ad.obs['order'].nunique()
    R = ad.obs['replicate'].nunique()
    X = np.zeros((n, T, R))
    print(f'After reformatting, the data will be shaped as={X.shape}')
    print('where the modes are genes by time points by experimental replicates')

    # List possible timepoints & replicates
    timepoints = ad.obs['order'].unique()
    replicates = ad.obs['replicate'].unique()

    # Loop over dataframe
    for i in range(ad.obs.shape[0]):
        order = ad.obs['order'][i]
        replicate = ad.obs['replicate'][i]
    
        time = np.where(timepoints == order)[0][0]
        rep = np.where(replicates == replicate)[0][0]
    
        X[:, time, rep] = ad.X[i, :]
    states = list(ad.var.index)
    metadata = ad.var
    
    output_file = os.path.join(outputPath, outputFile)
    with open(output_file, 'wb') as f:
        pickle.dump({'X': X, 'states': states, 'metadata': metadata}, f)
    
    print(f"Data saved to {output_file}")

if __name__ == "__main__":
    main()
