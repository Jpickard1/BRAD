"""
This script executes a series of commands on an AnnData object or single cell datasets loaded from a .h5ad file. You can write your own code to manipulate, visualize, or otherwise study these data. The results will be saved back to disk. You can use the scanpy, seaborn, matplotlib.pyplot, and other standard libraries to process .h5ad files, create visualizations, and explore a dataset with this file.

Arguments (four arguments):
    1. output directory: chatstatus['output-directory']
    2. output file: <name of output file>
    3. input file: <file created in previous step>
    4. scanpy commands: a list of scanpy commands to be executed on the AnnData object (provided as a single string with commands separated by a delimiter, e.g., ';')

Based on the arguments, the input file will be loaded, then your commands will be executed, and finally, the output or resulting ann data object will be saved to the corrrectou output file and directory. Your code is not responsible for loading the .h5ad object, that will happen automatically, and when loaded, the object will be called adata. Your scanpy commands can operate directly on the adata object that will be loaded for you.

Additionally, the following imports are already provided for you and can be used in your code:
```
import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
```

**Usage**
BRAD Line:
```
subprocess.run([sys.executable, "<path/to/script/>/scanpy_brad.py", chatstatus['output-directory'], <output file>, <input file>, "<scanpy commands>"], capture_output=True, text=True)
```

**Examples**
Use the below examples to help generate your code.

*Example 1*
User Prompt: Run scanpy preprocessing and UMAP visualization on XXX.h5ad and save the UMAP plot
Response Code:
```
response = subprocess.run([sys.executable, "<path/to/script/>/scanpy_brad.py", chatstatus['output-directory'], "XXX-modified.h5ad", "<path/to/data>/XXX.h5ad", "sc.pp.neighbors(adata); sc.tl.umap(adata); sc.pl.umap(adata, save='umap.png')"], capture_output=True, text=True)
```
Explination: the adata object will be loaded in memory already. The command "sc.pp.neighbors(adata)" will preprocess the data, then the command "sc.tl.umap(adata)" will perform UMAP, and finally the command "sc.pl.umap(adata, 'umap.png')" will save the UMAP to a well named file.

*Example 2*
User Prompt: Perform PCA and clustering on the dataset YYY.h5ad and save the PCA plot
Response Code:
```
response = subprocess.run([sys.executable, "<path/to/script/>/scanpy_brad.py", chatstatus['output-directory'], "YYY-modified.h5ad", "<path/to/data>/YYY.h5ad", "sc.pp.pca(adata); sc.tl.leiden(adata); sc.pl.pca(adata, save='pca.png')"], capture_output=True, text=True)
```
Explination: the adata object will be loaded in memory already. The command "sc.pp.pca(adata)" will preprocess the data, then the command "sc.tl.leiden(adata)" will perform the leiden algorithm, and finally the command "sc.pl.pca(adata, save='pca.png')" will save the PCA to a well named file.

*Example 3*
User Prompt: compute the distance between all cells that are not source or target to all source and target cells.
Reponse Code:
```
adata_X = adata.X
mask = ~adata.obs['recipe'].isin(['source', 'target'])
recipe_embeddings = adata[mask]
source_embeddings = adata[adata.obs['recipe'] == 'source']
target_embeddings = adata[adata.obs['recipe'] == 'target']
recipe_distances_source = cdist(recipe_embeddings.X, source_embeddings.X)
recipe_distances_target = cdist(recipe_embeddings.X, target_embeddings.X)
adata_dist = sc.AnnData(X=np.concatenate((recipe_distances_source, recipe_distances_target), axis=1), var=pd.concat((source_embeddings.obs, target_embeddings.obs)), obs=recipe_embeddings.obs)
adata_dist.write(os.path.join(chatstatus['output-directory'], <file name>.h5ad))
```

Each plot should have a title. Also if a legend is used, please place the legend on the right hand side.

**OUTPUT FILE NAME INSTRUCTIONS**
1. Output path should be chatstatus['output-directory']
2. Output file name should be `<descriptive name>.h5ad`
"""

import argparse
import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

def main(output_directory, output_file, input_file, scanpy_commands):

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

    # Deserialize the scanpy commands
    commands = scanpy_commands.split(';')

    # Execute the list of scanpy commands in memory
    print("****************************")
    print("      EXECUTE COMMANDS      ")
    print("****************************")
    for command in commands:
        command = command.strip()
        print(command)
        exec(command)

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    output_path = os.path.join(output_directory, output_file)
    adata.write(output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Execute scanpy commands on an AnnData object.')
    parser.add_argument('output_directory', type=str, help='The output directory.')
    parser.add_argument('output_file', type=str, help='The output file name.')
    parser.add_argument('input_file', type=str, help='The input file name.')
    parser.add_argument('scanpy_commands', type=str, help='The scanpy commands to be executed, separated by semicolons.')
    args = parser.parse_args()
    main(args.output_directory, args.output_file, args.input_file, args.scanpy_commands)
