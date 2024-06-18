import os
import pandas as pd

def save(chatstatus, df, name):
    # If this is part of a pipeline, then add the stage number to the printed output
    if 'stage' in chatstatus['process'].keys() and len(chatstatus['process']['stage']) != 0:
        stageNum = chatstatus['process']['stage'][0]['order']
        name += 'S' + str(stageNum) + '-' + name
    output_path = os.path.join(chatstatus['output-directory'], name)
    df.to_csv(output_path, index=False)
    return chatstatus

def pdfDownloadPath(chatstatus):
    path = os.path.join(chatstatus['output-directory'], 'pdf')
    return path