"""
This script performs sensor selection on models of dynamical systems. This is applied to models of cellular processes for biomarker selection.

Arguments (four arguments):
    1. output directory: chatstatus['output-directory']
    2. output file: <name of output file>
    3. input file: the file created previously with the LTI or LTV model
    4. time horizon for the gramian (an integer used as the upper bound of the summation)

Usage:
Command Line:
```
python <path/to/script/>biomarkerSelection.py <output path> <output file> <input file> <integer between 2 and 10: Gramian summation bound>
```
                                                   |              |            |                   |
                                                Argument 1     Argument 2   Argument 3          Argument 4
Example BRAD Line:
```
subprocess.run([sys.executable, '<path/to/script/>/buildLinearModel.py', chatstatus['output-directory'], S3-sensors.csv, S2-model.pkl, 5], capture_output=True, text=True)
```

Template BRAD Line:
```
subprocess.run([sys.executable, '<path/to/script/>/buildLinearModel.py', chatstatus['output-directory'], S3-<descriptive name>.csv, <Input file from S2>, <integer between 2 and 10: Gramian summation bound>], capture_output=True, text=True)
```

*Always replace <path/to/script> with the correct path given above.*

*Always replace anything in < > with descriptive names/variables.*


**OUTPUT FILE NAME INSTRUCTIONS**
1. Output path should be chatstatus['output-directory']
2. Output file name should be `S3-<descriptive name>.csv`
"""

import pickle
import os
import sys
import numpy as np
sys.path.append('/home/jpic/BRAD-Tools-Imports/')
from classes import model # LinearTimeInvariant
import pandas as pd
from scipy.sparse.linalg import eigs

def obsvEnergyMaximization(MODEL, gramT=10):
    """
    Perform observer energy maximization on a LinearTimeInvariant or LinearTimeVariant model.

    Params:
    --------------
    model (Model):
        An instance of LinearTimeInvariant or LinearTimeVariant.

    gramT (int, optional):
        Number of timepoints over which to compute the Gram matrix. Default is 10.

    Returns:
    --------------
    result (dict):
        The result of observer energy maximization specific to the input model. The result includes:
        - 'sensors': DataFrame containing information about the selected sensors.
        - 'dmd': Dictionary with Dynamic Mode Decomposition (DMD) results.
        - 'G': Gram matrix.
        - 'evals': Eigenvalues of the Gram matrix.
        - 'evecs': Eigenvectors of the Gram matrix.

    References:
    --------------
    Hasnain, A., Balakrishnan, S., Joshy, D. M., Smith, J., Haase, S. B., & Yeung, E. (2023).
    Learning perturbation-inducible cell states from observability analysis of transcriptome dynamics.
    Nature Communications, 14(1), 3148. [Nature Publishing Group UK London]
    """
    # Auth: Joshua Pickard
    #       jpic@umich.edu
    # Date: January 21, 2024
    #
    # Note: This developed in the github.com/Jpickard1/bioObsv and github.com/CooperStansbury/DMD_gene
    # repositories based upon the AqibHasnain/transcriptome-dynamics-dmd-observability
    if isinstance(MODEL, model.LinearTimeInvariant):
        # Perform observer energy maximization for LinearTimeInvariant model
        # You can access model-specific attributes and methods here
        # A = model.dmd_res['Atilde']
        # u = model.dmd_res['u_r']
        # x0_embedded = model.dmd_res['data_embedded'][:,0,:]
        G = MODEL.gram_matrix(T=gramT, reduced=True)
    elif isinstance(MODEL, model.LinearTimeVariant):
        # Perform observer energy maximization for LinearTimeVariant model
        # You can access model-specific attributes and methods here
        # need to implement for time invariant models
        G = MODEL.gram_matrix(T=gramT)
    else:
        raise ValueError("Unsupported model type. Supported types: LinearTimeInvariant, LinearTimeVariant")
        
    D, V = eigs(G, k=1)
    V = np.abs(V) # this line was missing. it is used in the original Hasnain code.

    # this line we will change based on the annotated data object
    obs = pd.DataFrame({'state'  : MODEL.states,
                        'ev1'    : V[:,0],
                        'weight' : np.real(V[:,0])})

    obs['rank'] = obs['weight'].rank(ascending=False)
    obs = obs.sort_values(by='rank', ascending=True)
    obs = obs.reset_index(drop=True)

    return {'sensors': obs,
            'G'      : G,
            'evals'  : D,
            'evecs'  : V}

def main():
    outputPath = sys.argv[1]      # chatstatus['output-directory']
    outputFile = sys.argv[2]      # output file name
    inputFile  = sys.argv[3]      # name of file where the model was previusly built
    gramT      = sys.argv[4]      # name of file where the model was previusly built
    print('******************************')
    print('       Sensor Selection       ')
    print('******************************')
    print(f"Output Path: {outputPath}")
    print(f"Output File: {outputFile}")
    print(f"Input File : {inputFile}")
    print(f"Gram T     : {gramT}")
    gramT = int(gramT)

    # Load the model from the input file
    with open(inputFile, 'rb') as file:
        MODEL = pickle.load(file)

    print('..............................')
    print('         Model Loaded         ')
    
    sensors = obsvEnergyMaximization(MODEL, gramT=gramT)['sensors']
    
    print('..............................')
    print('      Sensors Selected        ')
    
    
    df = pd.DataFrame(sensors)
    output_file_path = os.path.join(outputPath, outputFile)
    df.to_csv(output_file_path)    
    print(f"Model saved to {output_file_path}")

    print('..............................')
    print('      Sensors To File         ')
    

if __name__ == "__main__":
    main()