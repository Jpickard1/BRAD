"""
This script executes a series of custom commands to build a model for predicting the contribution to reprogramming obtained by the inclusion or exclusion of individual TFs using an AnnData object or single cell datasets loaded from a .h5ad file.

Arguments (four arguments):
    1. output directory: chatstatus['output-directory']
    2. output file: <name of output file>
    3. input file: <file created in previous step>
    4. custom commands: a string containing custom commands separated by semicolons

Based on the arguments, the input file will be loaded, the custom commands will be executed, and the model and results will be saved to the output directory.

Additionally, the following imports are already provided for you and can be used in your code:
```
import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import pickle
```

**Usage**
BRAD Line:
```
subprocess.run([sys.executable, "<path/to/script/>/custom_modeling_brad.py", chatstatus['output-directory'], <output file>, <input file>, "<custom commands>"], capture_output=True, text=True)
```

**Examples**
User Prompt: Run custom code to build a model and save the results
Response Code:
```
response = subprocess.run([sys.executable, "<path/to/script/>/custom_modeling_brad.py", chatstatus['output-directory'], "model_results.pkl", "<path/to/data>/HSC-reprogrammed-DE.h5ad", "<custom commands>"], capture_output=True, text=True)
```
"""
import argparse
import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import pickle

def main(output_directory, output_file, input_file, custom_commands):

    chatstatus = {
        'output-directory': output_directory
    }
    sc.settings.figdir = output_directory
    sc.set_figure_params(dpi=300)
    
    # Load the h5ad file using scanpy
    adata = sc.read_h5ad(input_file)
    print(f'adata loaded from {input_file}')
    print(f'adata.shape={adata.shape}')
    print(f'adata.obs.columns={adata.obs.columns}')
    print(f'adata.obs.head()={adata.obs.head()}')
    print(f'adata.var.head()={adata.var.head()}')

    # Define the context in which the custom code will run
    context = {
        'adata': adata,
        'cdist': cdist,
        'train_test_split': train_test_split,
        'RandomForestRegressor': RandomForestRegressor,
        'r2_score': r2_score,
        'mean_squared_error': mean_squared_error,
        'sns': sns,
        'plt': plt,
        'np': np,
        'pd': pd,
        'os': os,
        'pickle': pickle,
        'output_directory': output_directory
    }

    # Deserialize the custom commands
    commands = custom_commands.split(';')

    # Execute the list of custom commands in memory
    print("****************************")
    print("      EXECUTE COMMANDS      ")
    print("****************************")
    for command in commands:
        command = command.strip()
        print(command)
        exec(command, context)

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    output_path = os.path.join(output_directory, output_file)

    # Save the results
    with open(output_path, 'wb') as f:
        pickle.dump(context.get('results', {}), f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run custom commands for modeling.')
    parser.add_argument('output_directory', type=str, help='The output directory.')
    parser.add_argument('output_file', type=str, help='The output file name.')
    parser.add_argument('input_file', type=str, help='The input file name.')
    parser.add_argument('custom_commands', type=str, help='The custom commands as a string.')
    args = parser.parse_args()
    main(args.output_directory, args.output_file, args.input_file, args.custom_commands)
