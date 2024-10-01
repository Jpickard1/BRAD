"""
This script builds and trains a neural network to predict elements in the `adata.obs` columns using selected biomarkers.

Arguments (eight arguments):
    1. output directory: chatstatus['output-directory']
    2. output file: <name of output model file>
    3. input .csv file: a CSV file containing a 'States' column with a list of genes
    4. number of sensors (int): an integer specifying the number of biomarkers to use
    5. hidden nodes (int): number of nodes in the hidden layers
    6. epochs (int): number of training epochs, choose less than 500
    7. batch size (int): number of data points per training batch, choose less than 400
    8. number of hidden layers (int): number of hidden layers in the network

Usage:
Template BRAD Line:
subprocess.run([sys.executable, '<path/to/script/>/trainNeuralNetwork.py', chatstatus['output-directory'], <output model file>, <input CSV>, <num sensors>, <hidden nodes>, <epochs>, <batch size>, <number of hidden layers>], capture_output=True, text=True)

Example BRAD Line:
subprocess.run([sys.executable, '<path/to/script/>/trainNeuralNetwork.py', chatstatus['output-directory'], <output model file>, <input CSV>, "25", "10", "100", "50", "3"], capture_output=True, text=True)

Neural Network Training:
------------------------
This script loads a .h5ad AnnData object, extracts biomarkers (features) from the provided list of genes, and trains a neural network to predict a target from `adata.obs`. The model can be customized with parameters like number of hidden layers, learning rate, and epochs.

Steps:
1. Load the input CSV file with a 'States' column to select biomarkers (genes).
2. Extract the biomarker data from the .h5ad AnnData object.
3. Build a neural network model based on the given parameters.
4. Train the model to predict target variables from `adata.obs`.
5. Save the model and output training metrics such as total loss and error.

**OUTPUT FILE NAME INSTRUCTIONS**
1. Output path should be chatstatus['output-directory'].
2. Output file name should be NN-<descriptive name>.pkl
"""

import sys
import os
import numpy as np
import pandas as pd
import anndata
import pickle
import tensorflow as tf
from tensorflow.keras import layers, models
import seaborn as sns
import matplotlib.pyplot as plt

def main():
    # Parse command line arguments
    outputPath        = sys.argv[1]  # chatstatus['output-directory']
    outputFile        = sys.argv[2]  # output model file name
    inputFile         = sys.argv[3]   # input CSV file with biomarkers
    numSensors        = int(sys.argv[4])  # number of biomarkers (sensors) to use
    hidden_units      = int(sys.argv[5])  # number of hidden units
    epochs            = int(sys.argv[6])  # number of epochs for training
    batch_size        = int(sys.argv[7])  # batch size for training
    num_hidden_layers = int(sys.argv[8])  # number of hidden layers

    print("Input parameters")
    print(f"{outputPath       =}") 
    print(f"{outputFile       =}") 
    print(f"{inputFile        =}") 
    print(f"{numSensors       =}") 
    print(f"{hidden_units     =}") 
    print(f"{epochs           =}") 
    print(f"{batch_size       =}") 
    print(f"{num_hidden_layers=}") 

    # Load dataset
    print("Load Dataset")
    adata_file_path = '/nfs/turbo/umms-indikar/shared/projects/public_data/time_series_RNA/mitoticExit/krenning.pkl'
    with open(adata_file_path, 'rb') as file:
        adata = pickle.load(file)
    
    adata.var['Gene'] = adata.var.index.str.split('__').str[0]  # Extract gene names
    adata.var.index = adata.var['Gene']

    # Load biomarkers from CSV file
    print("Load Biomarkers")
    df = pd.read_csv(inputFile)
    biomarkers = df['state'].head(numSensors).tolist()  # Select the top numSensors biomarkers

    # Format dataset as X (features) and y (labels)
    print("Restructure Features")
    idxs = [list(adata.var['Gene']).index(bmkr) for bmkr in biomarkers if bmkr in adata.var['Gene']]
    print(f"Number of valid biomarkers used: {len(idxs)}")
    X = adata[:, idxs].X
    y = adata.obs['time'].values  # Modify as needed to select appropriate target column
    X = X.astype(np.float32)
    y = pd.get_dummies(adata.obs['phase'], prefix='phase')
    
    # Build Feed-Forward Neural Network
    print("Build Model")
    model = models.Sequential()
    model.add(layers.InputLayer(input_shape=(X.shape[1],)))
    for _ in range(num_hidden_layers):
        model.add(layers.Dense(hidden_units, activation='relu'))
    model.add(layers.Dense(y.shape[1]))
    model.compile(optimizer='adam', loss='mse', metrics=['mae', 'accuracy', 'Precision', 'Recall'])

    # Train the model
    print("Fit Model")
    history = model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=0.2)

    # Plot training history
    print("Plot")
    plt.figure(figsize=(8, 6), dpi=300)
    plt.plot(history.history['accuracy'], label='Training Accuracy', color='blue', linestyle='-', marker='o', markersize=4)
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='green', linestyle='--', marker='s', markersize=4)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.grid(True)
    plt.legend(loc='lower right', fontsize=10)
    sns.despine()
    plt.savefig(os.path.join(outputPath, 'training_accuracy.png'))
    plt.close()
    print("Image path: " + str(os.path.join(outputPath, 'training_accuracy.png')))

    # Save the model
    print("Save")
    model_save_path = os.path.join(outputPath, outputFile)
    model.save(model_save_path)
    print("Model path: " + str(model_save_path))
    
    # Save training history to a pickle file
    history_save_path = os.path.join(outputPath, 'training_history.pkl')
    with open(history_save_path, 'wb') as f:
        pickle.dump(history.history, f)

    print(f"Model saved to {model_save_path}")
    print(f"Training history saved to {history_save_path}")

    print("Job complete!")

if __name__ == "__main__":
    main()
