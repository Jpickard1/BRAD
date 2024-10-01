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
python <path/to/script/>evaluateBiomarkers.py <output path> <output file> <model file> <biomarker .csv file> <number of sensors>
```
                                                   |              |            |                   |                    |
                                                Argument 1     Argument 2   Argument 3          Argument 4        Argument 5
BRAD Line:
```
subprocess.run([sys.executable, '<path/to/script/>/evaluateBiomarkers.py', chatstatus['output-directory'], <output file>, <model file>, <biomarker .csv file>, <number of sensors>], capture_output=True, text=True)
```

*Always replace <path/to/script> with the correct path given above.*

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

def main():
    outputPath      = sys.argv[1]      # chatstatus['output-directory']
    outputFile      = sys.argv[2]      # output file name
    modelInputFile  = sys.argv[3]      # name of file where the model was previusly built
    sensorInputFile = sys.argv[4]      # name of file where the model was previusly built
    numberOfSensor  = sys.argv[5]      # name of file where the model was previusly built
    numberOfSensor  = int(numberOfSensor)

    print('******************************')
    print('       State Estimation       ')
    print('******************************')
    print(f"Output Path: {outputPath}")
    print(f"Output File: {outputFile}")
    print(f"Model Input File: {modelInputFile}")
    print(f"Sensor Input File: {sensorInputFile}")
    print(f"Number of Sensors: {numberOfSensor}")
    

    # Load the model from the input file
    with open(inputFile, 'rb') as file:
        MODEL = pickle.load(file)

    # Load the model from the input file
    df = pd.read_csv(sensorInputFile)
    MODEL.set_sensors(df[

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