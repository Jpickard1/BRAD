"""
This script builds linear models of time series data using Dynamic Mode Decomposition (DMD).

Arguments (four arguments):
    1. output directory: chatstatus['output-directory']
    2. output file: <name of output file>
    3. input file: the file created previously in the pipeline with formatted time series data
    4. DMD Rank: this parameter controls the amount reduced order modeling. A value of -1 indicates no model reduction

Usage:
Command Line:
```
python <path/to/script/>datasetSelector.py <output path> <output file> <input file> <dmd rank>
```
                                                 |              |            |           |
                                             Argument 1     Argument 2   Argument 3  Argument 4
BRAD Line:
```
subprocess.run([sys.executable, '<path/to/script/>/buildLinearModel.py', chatstatus['output-directory'], <output file>, <input file>, <dmd rank>], capture_output=True, text=True)
```

*Always replace <path/to/script> with the correct path given above.*

Dynamic Mode Decomposition (DMD):
--------------------------------
DMD is a data-driven modeling method that decomposes a given dataset into a collection of dynamic modes, each associated with a fixed frequency, growth/decay rate, and amplitude. The key idea is to approximate the linear dynamics of the system using snapshots of data.

Given a sequence of snapshots X = [x1, x2, ..., xm] and Y = [x2, x3, ..., xm+1], where each snapshot xi represents the state of the system at time ti, DMD seeks to find a matrix A such that:

    Y ≈ AX

To achieve this, DMD solves the following optimization problem:

    A = Y X^†

where X^† is the pseudoinverse of X.

To compute the DMD modes and eigenvalues, we perform the following steps:
1. Compute the Singular Value Decomposition (SVD) of X:
   
    X = UΣV*

2. Approximate A using the reduced-order model:
   
    Ã = U^* Y V Σ^(-1)

3. Compute the eigendecomposition of Ã:
   
    ÃW = WΛ

where Λ contains the eigenvalues and W contains the eigenvectors (DMD modes).

The DMD rank parameter controls the amount of reduced-order modeling. A value of -1 indicates no model reduction, using the full rank of the data matrix.


**OUTPUT FILE NAME INSTRUCTIONS**
1. Output path should be chatstatus['output-directory']
2. Output file name should be `S2-<descriptive name>.pkl`
"""

import pickle
import os
import sys
sys.path.append('/home/jpic/RAG-DEV/tutorials/biomarkers/functional-scripts/')
from model import Model # LinearTimeInvariant

def main():
    outputPath = sys.argv[1] # chatstatus['output-directory']
    outputFile = sys.argv[2] # output file name
    inputFile  = sys.argv[3]
    dmdRank    = int(sys.argv[4])
    if dmdRank == -1:
        dmdRank = None
    
    print('******************************')
    print('  Dynamic Mode Decomposition  ')
    print('******************************')
    print(f"Output Path: {outputPath}")
    print(f"Output File: {outputFile}")
    print(f"Input File: {inputFile}")
    print(f"DMD Rank: {dmdRank}")


    # load the input file
    with open(inputFile, 'rb') as file:
        data = pickle.load(file)
    X = data['X']
    states = data['states']
    LTI = model.LinearTimeInvariant(data = X, states = states, dmdRank = dmdRank)
    
    output_file_path = os.path.join(outputPath, outputFile)
    with open(output_file_path, 'wb') as file:
        pickle.dump(LTI, file)
    
    print(f"Model saved to {output_file_path}")

if __name__ == "__main__":
    main()
