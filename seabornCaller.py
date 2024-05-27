import seaborn as sns
import inspect
import re
import matplotlib.pyplot as plt
import inspect
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from seaborn.palettes import color_palette
import random
import json
import matplotlib

from functionCaller import *

import seaborn as sns
import matplotlib.pyplot as plt
import logging
import random

def callSnsV3(chatstatus):
    prompt = chatstatus['prompt']                                                    # Get the user prompt
    df = chatstatus['current table']['tab']                                          # Get the data to plot
    df.dropna(inplace=True)
    newPrompt = True
    plot_functions = plottingMethods()                                       # Identify the plotting function from keyword
    for key in plot_functions:
        if key in prompt:
            plot_func, sns_func = plot_functions[key]
            break
    else:
        plot_func, sns_func = random.choice(list(plot_functions.values()))   # Choose a random plotting function is none is clear
        logging.info("No matching plot type found in the prompt.")
    while True:
        try:
            # Parse the parameters of the plot to be generated
            print('Step -1') if chatstatus['config']['debug'] else None
            if 'process' not in chatstatus.keys() or chatstatus['process'] is None:
                chatstatus['process'] = {}  # Begin saving plotting arguments
                chatstatus['process']['name'] = 'SNS'
                chatstatus['process']['type'] = str(plot_func)
            if 'params' not in chatstatus['process'].keys():
                chatstatus['process']['params'] = functionArgs(sns_func)             # Extract arguments from a seaborn function based on function handle
            if newPrompt:
                chatstatus = checkPlottingConfigurations(chatstatus, df)                 # Apply a rules based parser to extract plotting args
                chatstatus['process']['params'] = getFunctionArgs(chatstatus)            # Apply t5-small fine tuned to extract plotting args
                chatstatus['process']['params']['data'] = df                             # Explicitly save the data we are plotting
            print('Step 0') if chatstatus['config']['debug'] else None
            validate_arguments(sns_func, chatstatus['process']['params'])            # Perform crude argument validation to see if the arguments were extracted well

            print('Step 1') if chatstatus['config']['debug'] else None
                
            # Update missing values
            setSeabornConfigurations(chatstatus)  # Check if there are missing values in the parameters that had default values
            print('Step 2') if chatstatus['config']['debug'] else None
            for k in chatstatus['process']['params'].keys():  # Check if there are missing values in the args that were recently used
                if chatstatus['process']['params'][k] is None and k in chatstatus['plottingParams'].keys():
                    chatstatus['process']['params'][k] = chatstatus['plottingParams'][k]
            print('Step 3') if chatstatus['config']['debug'] else None
            if 'figsize' not in chatstatus['process']['params'].keys() or chatstatus['process']['params']['figsize'] is None:
                chatstatus['process']['params']['figsize'] = chatstatus['config']['display']['figsize']
            print('Step 4') if chatstatus['config']['debug'] else None
            if chatstatus['process']['params']['dpi'] is None:
                chatstatus['process']['params']['dpi'] = chatstatus['config']['display']['dpi']
            if chatstatus['process']['params']['colormap'] is None:
                chatstatus['process']['params']['colormap'] = chatstatus['config']['display']['colormap']

            if chatstatus['config']['debug']:  # Show the plotting arguments extracted from the prompt
                display(chatstatus['process'])
            print('Step 5') if chatstatus['config']['debug'] else None
            # Attempt to create the plot
            fig, chatstatus['process']['params']['ax'] = plt.subplots(figsize=chatstatus['process']['params']['figsize'], dpi=chatstatus['process']['params']['dpi'])
            chatstatus['process']['params']['ax'] = plot_func(chatstatus['process']['params'])
            print('Step 6') if chatstatus['config']['debug'] else None
            plt.xlim(chatstatus['process']['params']['xlim'])
            plt.ylim(chatstatus['process']['params']['ylim'])
            print('Step 7') if chatstatus['config']['debug'] else None
            if chatstatus['process']['params']['despine']:
                sns.despine()
            if chatstatus['process']['params']['title'] is not None:
                plt.title(chatstatus['process']['params']['title'])
            if chatstatus['process']['params']['xlabel'] is not None:
                plt.xlabel(chatstatus['process']['params']['xlabel'])
            if chatstatus['process']['params']['ylabel'] is not None:
                plt.ylabel(chatstatus['process']['params']['ylabel'])
            if chatstatus['process']['params']['ax'] is not None:
                plt.show()

            # Save parameters for future use
            for k in chatstatus['process']['params'].keys():
                chatstatus['plottingParams'][k] = chatstatus['process']['params'][k]

        except Exception as e:
            print(f"An error occurred: {e}")
        user_input = input("Would you like to enter new parameters? (y/n): ").strip()
        if user_input.lower() in ['y', 'yes', 'sure', 'continue']:
            print('How would you like to modify this plot?')
            prompt = input().strip()
            if '=' in prompt:
                newPrompt = False
                keyValuePairs = prompt.split(',')
                for kv in keyValuePairs:
                    k, v = kv.split('=')
                    try:
                        chatstatus['process']['params'][k] = eval(v)
                    except:
                        chatstatus['process']['params'][k] = v                        
            else:
                newPrompt = True
                chatstatus['prompt'] = prompt
        else:
            break  # Exit the loop if the user doesn't want to enter new parameters
    return chatstatus

'''
for param in chatstatus['process']['params'].keys():
    user_value = input(f"Enter value for {param} (current: {chatstatus['process']['params'][param]}): ").strip()
    if user_value:
        # Try to convert user input to the correct type, if possible
        try:
            # Evaluate the input to handle different types (e.g., numbers, lists)
            chatstatus['process']['params'][param] = eval(user_value)
        except:
            chatstatus['process']['params'][param] = user_value
'''

def callSnsV2(chatstatus, chatlog):
    prompt = chatstatus['prompt']                                                                     # Get the user prompt
    df = chatstatus['current table']['tab']                                                           # Get the data to plot    

    # Parse the parameters of the plot to be generated
    plot_functions = plottingMethods()                                                                # Identify the plotting function from keyword
    for key in plot_functions:
        if key in prompt:
            plot_func, sns_func = plot_functions[key]
            break
    else:
        plot_func, sns_func = random.choice(list(plot_functions.values()))                            # Choose a random plotting function is none is clear
        logging.info("No matching plot type found in the prompt.")
    
    chatstatus['process'] = {}                                                                        # Begin saving plotting arguments
    chatstatus['process']['name'] = 'SNS'
    chatstatus['process']['type'] = str(plot_func)
    chatstatus['process']['params'] = functionArgs(sns_func)                                          # Extract arguments from a seaborn function based on function handle
    chatstatus['process']['params'] = getFunctionArgs(chatstatus)                                     # Identify necessary arguments for the searbon function within the prompt
    chatstatus['process']['params']['data'] = df                                                      # Explicitly save the data we are plotting
    validate_arguments(sns_func, chatstatus['process']['params'])                                     # Perform crude argument validation to see if the arguments were extracted well
    
    # Update missing values
    setSeabornConfigurations(chatstatus)                                                              # Check if there are missing values in the parameters that had default values
    for k in chatstatus['process']['params'].keys():                                                  # Check if there are missing values in the args that were recently used
        if chatstatus['process']['params'][k] is None and k in chatstatus['plottingParams'].keys():
            chatstatus['process']['params'][k] = chatstatus['plottingParams'][k]
    if chatstatus['process']['params']['figsize'] == None:
        chatstatus['process']['params']['figsize'] = chatstatus['config']['display']['figsize']
    if chatstatus['process']['params']['dpi'] == None:
        chatstatus['process']['params']['dpi'] = chatstatus['config']['display']['dpi']
    if chatstatus['process']['params']['colormap'] == None:
        chatstatus['process']['params']['colormap'] = chatstatus['config']['display']['colormap']
        
    if chatstatus['config']['debug']:                                                                 # Show the plotting arguments extracted from the prompt
        display(chatstatus['process'])
    
    try:
        fig, ax = plt.figure(figsize = chatstatus['process']['params']['figsize'], dpi= chatstatus['process']['params']['dpi'])
        ax = plot_func(chatstatus['process']['params'])
        plt.xlim(chatstatus['process']['params']['xlim'])
        plt.ylim(chatstatus['process']['params']['ylim'])
        if chatstatus['process']['params']['despine']:
            sns.despine()
        if chatstatus['process']['params']['title'] is not None:
            plt.title(chatstatus['process']['params']['title'])
        if chatstatus['process']['params']['xlabel'] is not None:
            plt.xlabel(chatstatus['process']['params']['xlabel'])
        if chatstatus['process']['params']['ylabel'] is not None:
            plt.ylabel(chatstatus['process']['params']['ylabel'])
        if ax is not None:
            plt.show()
        for k in chatstatus['process']['params'].keys():
            chatstatus['plottingParams'][k] = chatstatus['process']['params'][k]
    except Exception as e:
        print(f"An error occurred: {e}")

    return chatstatus

def callSns(chatstatus, chatlog):
    prompt = chatstatus['prompt'].lower()
    
    # Determine the plotting function to use (the name must be given)
    plot_functions = plottingMethods()
    for key in plot_functions:
        if key in prompt:
            plot_func, sns_func = plot_functions[key]
            break
    else:
        plot_func, sns_func = random.choice(list(plot_functions.values()))
        logging.info("No matching plot type found in the prompt.")
    
    chatstatus['process'] = {}
    chatstatus['process']['name'] = 'SNS'
    chatstatus['process']['params'] = functionArgs(sns_func)
    chatstatus['process']['params'] = getFunctionArgs(chatstatus)
    if chatstatus['config']['debug']:
        display(chatstatus['process']['params'])    
    validate_arguments(sns_func, chatstatus['process']['params'])
    # populate any empty arguments with the previous arguments
    # i = len(chatlog) - 1
    for k in chatstatus['process']['params'].keys():
        if chatstatus['process']['params'][k] is None and k in chatstatus['plottingParams'].keys():
            chatstatus['process']['params'][k] = chatstatus['plottingParams'][k]
    
    if chatstatus['config']['debug']:
        display(chatstatus['process']['params'])
    
    if chatstatus['process']['params']['figsize'] == None:
        chatstatus['process']['params']['figsize'] = chatstatus['config']['display']['figsize']
    if chatstatus['process']['params']['dpi'] == None:
        chatstatus['process']['params']['dpi'] = chatstatus['config']['display']['dpi']
    if chatstatus['process']['params']['colormap'] == None:
        chatstatus['process']['params']['colormap'] = chatstatus['config']['display']['colormap']
    
    setSeabornConfigurations(chatstatus)
    sns.set_palette(chatstatus['process']['params']['colormap'].lower())
    plt.figure(figsize = chatstatus['process']['params']['figsize'],
               dpi     = chatstatus['process']['params']['dpi'],
              )
    ax = None
    df = chatstatus['current table']['tab']
    chatstatus['process']['params']['data'] = df

    try:
        print('Trying to plot')
        ax = plot_func(chatstatus['process']['params'])
    except Exception as e:
        print(f"An error occurred: {e}")
        ax = None  # or any other fallback you want to use in case of an error

    plt.xlim(chatstatus['process']['params']['xlim'])
    plt.ylim(chatstatus['process']['params']['ylim'])
    if chatstatus['process']['params']['despine']:
        sns.despine()
    if chatstatus['process']['params']['title'] is not None:
        plt.title(chatstatus['process']['params']['title'])
    if chatstatus['process']['params']['xlabel'] is not None:
        plt.xlabel(chatstatus['process']['params']['xlabel'])
    if chatstatus['process']['params']['ylabel'] is not None:
        plt.ylabel(chatstatus['process']['params']['ylabel'])
    if ax is not None:
        plt.show()
    for k in chatstatus['process']['params'].keys():
        chatstatus['plottingParams'][k] = chatstatus['process']['params'][k]
    return chatstatus

def setSeabornConfigurations(chatstatus):
    '''this function currently does nothing'''
    # load the configurations
    with open('configSeaborn.json', 'r') as f:
        configSns = json.load(f)
            
    # check palette
    configSns['palette'] = chatstatus['process']['params'].get('palette', configSns['palette'])
    chatstatus['process']['params']['palette'] = configSns['palette']
    
    # check palette
    configSns['default_marker'] = chatstatus['process']['params'].get('marker', configSns['default_marker'])
    chatstatus['process']['params']['marker'] = configSns['default_marker']
    
    # check palette
    configSns['despine'] = chatstatus['process']['params'].get('despine', configSns['despine'])
    chatstatus['process']['params']['despine'] = configSns['despine']
    
    # check context
    configSns['context'] = chatstatus['process']['params'].get('context', configSns['context'])
    chatstatus['process']['params']['context'] = configSns['context']
    
    # check style
    configSns['snsStyle'] = chatstatus['process']['params'].get('snsStyle', configSns['snsStyle'])
    chatstatus['process']['params']['snsStyle'] = configSns['snsStyle']
    
    # check font_scale
    configSns['font_scale'] = chatstatus['process']['params'].get('font_scale', configSns['font_scale'])
    chatstatus['process']['params']['font_scale'] = configSns['font_scale']

    # check sizes
    configSns['rc']['figure.figsize'] = chatstatus['process']['params'].get('figsize', configSns['rc']['figure.figsize'])
    chatstatus['process']['params']['figsize'] = configSns['rc']['figure.figsize']
    
    configSns['rc']['axes.titlesize'] = chatstatus['process']['params'].get('axes.titlesize', configSns['rc']['axes.titlesize'])
    chatstatus['process']['params']['axes.titlesize'] = configSns['rc']['axes.titlesize']
    
    configSns['rc']['axes.labelsize'] = chatstatus['process']['params'].get('axes.labelsize', configSns['rc']['axes.labelsize'])
    chatstatus['process']['params']['axes.labelsize'] = configSns['rc']['axes.labelsize']
    
    configSns['rc']['xtick.labelsize'] = chatstatus['process']['params'].get('xtick.labelsize', configSns['rc']['xtick.labelsize'])
    chatstatus['process']['params']['xtick.labelsize'] = configSns['rc']['xtick.labelsize']
    
    configSns['rc']['ytick.labelsize'] = chatstatus['process']['params'].get('ytick.labelsize', configSns['rc']['ytick.labelsize'])
    chatstatus['process']['params']['ytick.labelsize'] = configSns['rc']['ytick.labelsize']

    # Apply the seaborn configurations
    # sns.set_context(context=configSns['context'])
    # sns.set_style(style=configSns['snsStyle'], rc=configSns['rc'])
    # sns.set_palette(palette=configSns['palette'])
    # sns.set(font_scale=configSns['font_scale'])

    # save all to a file again
    with open('configSeaborn.json', 'w') as f:
        json.dump(configSns, f, indent=4)


    return chatstatus
    

def plottingMethods():
    plot_functions = {
        "line"         : [lineplotCaller   , sns.lineplot   ],
        "bar"          : [barplotCaller    , sns.barplot    ],
        "histogram"    : [histplotCaller   , sns.histplot   ],
        "box"          : [boxplotCaller    , sns.boxplot    ],
        "violin"       : [violinplotCaller , sns.violinplot ],
        "scatter"      : [scatterplotCaller, sns.scatterplot],
        "pair"         : [PairGridCaller   , sns.pairplot   ],
        "heatmap"      : [heatmapCaller    , sns.heatmap    ],
        "pair"         : [pairplotCaller   , sns.pairplot   ],
        "kde"          : [kdeplotCaller    , sns.kdeplot    ],
        "joint"        : [JointGridCaller  , sns.JointGrid  ],
#        "swarm"        : [swarmplotCaller  , sns.swarmplot]
#        "count"        : [countplotCaller  , sns.countplot]
#        "cat"          : [catplotCaller    , sns.plot]
#        "reg"          : [regplotCaller    , sns.plot]
    }
    return plot_functions

#        "pie"          : pie_chart   ,
#        "area"         : area_plot   ,
#        "hexbin"       : hexbin_plot ,
#        "facet grid"   : facet_grid  ,
#        "strip"        : strip_plot  ,
#        "dist"         : dist_plot   ,
#        "distribution" : dist_plot   ,


def functionArgs(func):
    plot_args = {
        'title'    : None,
        'x'        : None,
        'y'        : None,
        'xlabel'   : None,
        'ylabel'   : None,
        'figsize'  : None,
        'dpi'      : None,
        'xlim'     : None,
        'ylim'     : None,
        'colormap' : None,
        'despine'  : None,
        'marker'   : '*',
        'snsStyle' : None,
    }
    signature = inspect.signature(func)
    for param in signature.parameters.values():
        plot_args[param.name] = None
    return plot_args

import inspect
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from seaborn.palettes import color_palette

def validate_arguments(func, arguments):
    """
    Validates arguments against the function's signature and specific requirements.
    
    Parameters:
    func (function): The seaborn plotting function.
    arguments (dict): The dictionary of arguments to be passed to the function.
    
    Returns:
    (list, dict): A tuple containing a list of missing arguments and a dictionary of incorrect arguments with error messages.
    """
    signature = inspect.signature(func)
    missing_args = []
    incorrect_args = {}
    
    for param in signature.parameters.values():
        if param.name not in arguments:
            missing_args.append(param.name)
        elif param.name in arguments and arguments[param.name] is not None:
            try:
                # value of the argument
                value = arguments[param.name]

                # Check the important ones
                if param.name == 'data':
                    if not isinstance(value, (pd.DataFrame, np.ndarray, dict, list)):
                        incorrect_args[param.name] = "Expected a pandas DataFrame, numpy ndarray, dictionary, or sequence."
                
                elif param.name in ['x', 'y', 'hue', 'size', 'style', 'units']:
                    if isinstance(arguments.get('data'), pd.DataFrame):
                        if value not in arguments['data'].columns:
                            incorrect_args[param.name] = f"Column '{value}' not found in the DataFrame."
                    elif not isinstance(value, (list, np.ndarray)):
                        incorrect_args[param.name] = f"Expected a vector or key in data for '{param.name}'."
                        
                elif param.name == 'ax':
                    if not isinstance(value, plt.Axes):
                        incorrect_args[param.name] = "Expected a matplotlib.axes.Axes instance."
                        
                elif param.name == 'bins':
                    if not isinstance(value, int):
                        incorrect_args[param.name] = "Expected a float."
                        
                elif param.name == 'binrange':
                    if not isinstance(value, (tuple, list)) or len(value) != 2:
                        incorrect_args[param.name] = "Expected a pair of numbers or a pair of pairs."

                elif param.name == 'binwidth':
                    if not isinstance(value, (int, float, tuple, list)):
                        incorrect_args[param.name] = "Expected a number or pair of numbers."
                        
                elif param.name == 'bw':
                    if not (isinstance(value, str) or isinstance(value, float)):
                        incorrect_args[param.name] = "Expected a string or float."
                        
                elif param.name == 'capsize':
                    if not isinstance(value, float):
                        incorrect_args[param.name] = "Expected a float."
                        
                elif param.name == 'cbar':
                    if not isinstance(value, bool):
                        incorrect_args[param.name] = "Expected a boolean."

                elif param.name == 'cbar_ax':
                    if not isinstance(value, matplotlib.axes.Axes):
                        incorrect_args[param.name] = "Expected a matplotlib.axes.Axes object."

                elif param.name == 'cbar_kws':
                    if not isinstance(value, dict):
                        incorrect_args[param.name] = "Expected a dictionary."
                
                elif param.name == 'ci':
                    if not (isinstance(value, int) or value in ["sd", None]):
                        incorrect_args[param.name] = 'Expected an integer, "sd", or None.'
                
                elif param.name == 'color':
                    if not isinstance(value, (str, tuple)):
                            incorrect_args[param.name] = "Expected a matplotlib color string or tuple."
                            
                elif param.name == 'common_bins':
                    if not isinstance(value, bool):
                        incorrect_args[param.name] = "Expected a boolean."

                elif param.name == 'common_norm':
                    if not isinstance(value, bool):
                        incorrect_args[param.name] = "Expected a boolean."

                elif param.name == 'cumulative':
                    if not isinstance(value, bool):
                        incorrect_args[param.name] = "Expected a boolean."
                            
                elif param.name == 'cut':
                    if not isinstance(value, float):
                        incorrect_args[param.name] = "Expected a float."
                        
                elif param.name == 'dashes':
                    if not (isinstance(value, (bool, list, dict))):
                        incorrect_args[param.name] = "Expected a boolean, list, or dictionary."
                        
                elif param.name == 'discrete':
                    if not isinstance(value, bool):
                        incorrect_args[param.name] = "Expected a boolean."
                
                elif param.name == 'dodge':
                    if not isinstance(value, bool):
                        incorrect_args[param.name] = "Expected a boolean."
                
                elif param.name == 'edgecolor':
                    try:
                        edgecolor = plt.colors.to_rgba(value)
                    except ValueError:
                        if value != "gray":
                            incorrect_args[param.name] = 'Expected a valid matplotlib color or "gray".'
                        
                elif param.name == 'element':
                    if value not in {"bars", "poly", "step"}:
                        incorrect_args[param.name] = f"Expected one of {'bars', 'poly', 'step'}, got {value}."
                        
                elif param.name == 'err_style':
                    if value not in ["band", "bars"]:
                        incorrect_args[param.name] = 'Expected "band" or "bars".'
                
                elif param.name == 'err_kws':
                    if not isinstance(value, dict):
                        incorrect_args[param.name] = "Expected a dictionary."
                
                elif param.name == 'errorbar':
                    if not (isinstance(value, str) or isinstance(value, tuple) or callable(value)):
                        incorrect_args[param.name] = "Expected a string, (string, number) tuple, or callable."
                        
                elif param.name == 'errcolor':
                    try:
                        errcolor = plt.colors.to_rgba(value)
                    except ValueError:
                        incorrect_args[param.name] = "Expected a valid matplotlib color."
                        
                elif param.name == 'errwidth':
                    if not isinstance(value, float):
                        incorrect_args[param.name] = "Expected a float."
                        
                elif param.name == 'estimator':
                    if not (value is None or callable(value) or isinstance(value, str)):
                        incorrect_args[param.name] = "Expected a pandas method name, callable, or None."
                        
                elif param.name == 'fill':
                    if not isinstance(value, bool):
                        incorrect_args[param.name] = "Expected a boolean."
                        
                elif param.name == 'fliersize':
                    if not isinstance(value, float):
                        incorrect_args[param.name] = "Expected a float."
                        
                elif param.name == 'formatter':
                    if not callable(value):
                        incorrect_args[param.name] = "Expected a callable."
                
                elif param.name == 'gridsize':
                    if not isinstance(value, int):
                        incorrect_args[param.name] = "Expected an integer."
                
                elif param.name == 'hue_norm':
                    if not (isinstance(value, tuple) or isinstance(value, Normalize)):
                        incorrect_args[param.name] = "Expected a tuple or matplotlib.colors.Normalize."
                
                elif param.name == 'hue_order' or param.name == 'order':
                    if not isinstance(value, list):
                        incorrect_args[param.name] = "Expected a list of strings."
                        
                elif param.name == 'inner':
                    if value not in ["box", "quartile", "point", "stick", None]:
                        incorrect_args[param.name] = 'Expected "box", "quartile", "point", "stick", or None.'
                        
                elif param.name == 'jitter':
                    if not (isinstance(value, (float, bool))):
                        incorrect_args[param.name] = "Expected a float or boolean."
                        
                elif param.name == 'kde':
                    if not isinstance(value, bool):
                        incorrect_args[param.name] = "Expected a boolean."

                elif param.name == 'kde_kws':
                    if not isinstance(value, dict):
                        incorrect_args[param.name] = "Expected a dictionary."
                        
                elif param.name == 'legend':
                    if value not in ["auto", "brief", "full", False, True]:
                        incorrect_args[param.name] = 'Expected one of "auto", "brief", "full", False, or True.'
                        
                elif param.name == 'line_kws':
                    if not isinstance(value, dict):
                        incorrect_args[param.name] = "Expected a dictionary."
                        
                elif param.name == 'linewidth':
                    if not isinstance(value, (float, int)):
                        incorrect_args[param.name] = "Expected a float or integer."
                
                elif param.name == 'log_scale':
                    if isinstance(value, (bool, int, float, tuple, list)):
                        if isinstance(value, (tuple, list)) and len(value) not in (1, 2):
                            incorrect_args[param.name] = "Expected a boolean or a tuple/list of length 1 or 2."
                    else:
                        incorrect_args[param.name] = "Expected a boolean, int, float, or tuple/list."
                
                elif param.name == 'markers':
                    if not (isinstance(value, (bool, list, dict))):
                        incorrect_args[param.name] = "Expected a boolean, list, or dictionary."
                        
                elif param.name == 'multiple':
                    if value not in {"layer", "dodge", "stack", "fill"}:
                        incorrect_args[param.name] = f"Expected one of {'layer', 'dodge', 'stack', 'fill'}, got {value}."
                
                elif param.name == 'n_boot':
                    if not isinstance(value, int):
                        incorrect_args[param.name] = "Expected an integer."
                        
                elif param.name == 'native_scale':
                    if not isinstance(value, bool):
                        incorrect_args[param.name] = "Expected a boolean."
                
                elif param.name == 'orient':
                    if value not in ["x", "y", "v", "h"]:
                        incorrect_args[param.name] = 'Expected "x", "y", "v", or "h".'
                
                #elif param.name == 'palette':
                #    if not (isinstance(value, (str, list, dict, plt.Colormap)) or callable(value)):
                #        incorrect_args[param.name] = "Expected a string, list, dictionary, or matplotlib.colors.Colormap."

                elif param.name == 'palette':
                    if not isinstance(value, (str, list, dict, matplotlib.colors.Colormap)):
                        incorrect_args[param.name] = "Expected a string, list, dictionary, or matplotlib.colors.Colormap."
                
                elif param.name == 'pmax':
                    if not isinstance(value, (int, float)) and (value is not None):
                        incorrect_args[param.name] = "Expected a number or None."

                elif param.name == 'pthresh':
                    if not isinstance(value, (int, float)) and (value is not None):
                        incorrect_args[param.name] = "Expected a number or None."
                
                elif param.name == 'saturation':
                    if not isinstance(value, float):
                        incorrect_args[param.name] = "Expected a float."
                
                elif param.name == 'scale':
                    if value not in ["area", "count", "width"]:
                        incorrect_args[param.name] = 'Expected "area", "count", or "width".'
                        
                elif param.name == 'scale_hue':
                    if not isinstance(value, bool):
                        incorrect_args[param.name] = "Expected a boolean."
                
                elif param.name == 'seed':
                    if not (isinstance(value, (int, np.random.Generator, np.random.RandomState))):
                        incorrect_args[param.name] = "Expected an int, numpy.random.Generator, or numpy.random.RandomState."
                
                elif param.name == 'shrink':
                    if not isinstance(value, (int, float)):
                        incorrect_args[param.name] = "Expected a number."
                
                elif param.name == 'sizes':
                    if not (isinstance(value, (list, dict, tuple))):
                        incorrect_args[param.name] = "Expected a list, dict, or tuple."
                
                elif param.name == 'size_norm':
                    if not (isinstance(value, tuple) or isinstance(value, Normalize)):
                        incorrect_args[param.name] = "Expected a tuple or matplotlib.colors.Normalize."
                
                elif param.name == 'size_order':
                    if not isinstance(value, list):
                        incorrect_args[param.name] = "Expected a list."
                                
                elif param.name == 'split':
                    if not isinstance(value, bool):
                        incorrect_args[param.name] = "Expected a boolean."
                            
                elif param.name == 'sort':
                    if not isinstance(value, bool):
                        incorrect_args[param.name] = "Expected a boolean."
                        
                elif param.name == 'stat':
                    if value not in ['count', 'frequency', 'probability', 'proportion', 'percent', 'density']:
                        incorrect_args[param.name] = 'Expected "count", "frequency", "probability", "proportion", "percent", or "density".'
                            
                elif param.name == 'style_order':
                    if not isinstance(value, list):
                        incorrect_args[param.name] = "Expected a list."
                        
                elif param.name == 'thresh':
                    if not isinstance(value, (int, float)):
                        incorrect_args[param.name] = "Expected a number or None."
                        
                elif param.name == 'weights':
                    if isinstance(arguments.get('data'), pd.DataFrame):
                        if value not in arguments['data'].columns:
                            incorrect_args[param.name] = f"Column '{value}' not found in the DataFrame."
                    elif not isinstance(value, (list, np.ndarray)):
                        incorrect_args[param.name] = "Expected a vector or key in data."

                elif param.name == 'whis':
                    if not isinstance(value, float):
                        incorrect_args[param.name] = "Expected a float."

                elif param.name == 'width':
                    if not isinstance(value, float):
                        incorrect_args[param.name] = "Expected a float."
                        
            except (TypeError, ValueError) as e:
                incorrect_args[param.name] = f"Expected type {param.annotation}, got {type(arguments[param.name])}. Error: {e}"
    
    return missing_args, incorrect_args

def checkPlotLabels(chatstatus):
    '''
    Parses title, xlabel, and ylabel
    '''
    prompt = chatstatus['prompt']
    patterns = {
        'title': r"title\s*'([^']*)'",
        'xlabel': r"x\s*label\s*'([^']*)'",
        'ylabel': r"y\s*label\s*'([^']*)'"
    }
    for label, pattern in patterns.items():
        match = re.search(pattern, prompt, re.IGNORECASE)
        if match:
            chatstatus['process'][label] = match.group(1)
    return chatstatus

def separate_punctuation_with_spaces(text):
    # Use regular expression to replace each punctuation mark with ' <punctuation> '
    return re.sub(r'([.,!?;:"(){}\[\]])', r' \1 ', text)

def is_valid_colormap(colormap_name):
    return colormap_name in plt.colormaps()

def is_valid_color(color_string):
    try:
        mcolors.to_rgba(color_string)
        return True
    except ValueError:
        return False    

def should_apply_hue(promptWords, data, max_categories=15):
    '''
    Detect if hue should be applied and determine the hue variable based on the prompt and data.

    Parameters:
    - promptWords: List of words parsed from the prompt.
    - data: Pandas DataFrame containing the data.

    Returns:
    - hue_var: String indicating the variable to use as hue, or None if hue is not needed.
    '''
    if 'hue' in promptWords:
        loc = promptWords.index('hue')
        hue_var = promptWords[loc + 1]
        if hue_var in data.columns:
            return hue_var
    
    # Check if there is a categorical column that could be used as hue
    categorical_columns = [col for col in data.columns if pd.api.types.is_categorical_dtype(data[col]) or data[col].dtype == 'object']
    categorical_columns = [col for col in categorical_columns if data[col].nunique() <= max_categories]

    if categorical_columns:
        return random.choice(categorical_columns)  # Return the first categorical column
    
    return None  # No hue variable detected

def checkPlottingConfigurations(chatstatus, df):
    '''
    check if any variables need to be changed
    '''
    prompt = separate_punctuation_with_spaces(chatstatus['prompt']) # we don't mess with the '-' character
    promptWords = prompt.split(' ')
    unwanted_strings = {'', ' ', 'of', 'as', 'use', 'is', 'to', 'by', 'the', ';', '(', '[', '.', ',', '!', '?', ';', ':', '"', '(', ')', '{', '}', '\[', '\]' ']', ')' } # we don't mess with the '-' character
    promptWords = [s for s in promptWords if s not in unwanted_strings]
    # if chatstatus['debug']: logging.info(promptWords)

    if 'dpi' in promptWords:
        loc = promptWords.index('dpi')
        try:
            chatstatus['config']['display']['dpi'] = int(promptWords[loc+1])
        except ValueError:
            print('The dpi value is unclear')
    if 'figsize' in promptWords:
        loc = promptWords.index('figsize')
        try:
            chatstatus['config']['display']['figsize'] = (int(promptWords[loc+1]), int(promptWords[loc+2]))
        except ValueError:
            print('The figsize value is unclear')
    if 'cm' in prompt or 'colormap' in prompt:
        loc = promptWords.index('cm') if 'cm' in promptWords else promptWords.index('colormap')
        if is_valid_colormap(promptWords[loc + 1]):
            chatstatus['config']['display']['colormap'] = promptWords[loc + 1]
        else:
            print('The colormap is unclear')
    if 'ec' in promptWords or 'edgecolor' in prompt or 'edge-color' in prompt:
        if 'ec' in promptWords:
            loc = promptWords.index('ec')
        elif 'edge-color' in prompt:
            loc = promptWords.index('edge-color')
        else:
            loc = promptWords.index('edgecolor')
        if is_valid_color(promptWords[loc + 1]):
            chatstatus['process']['edgecolor'] = promptWords[loc + 1]
        else:
            print('The colormap is unclear')
    if 'markersize' in promptWords:
        loc = promptWords.index('markersize')
        try:
            chatstatus['process']['markersize'] = int(promptWords[loc + 1])
        except ValueError:
            print('The markersize value is unclear')
    if 'linewidth' in promptWords:
        loc = promptWords.index('linewidth')
        try:
            chatstatus['process']['linewidth'] = int(promptWords[loc + 1])
        except ValueError:
            print('The linewidth value is unclear')
    #if 'grid' in promptWords:
    #    chatstatus['process']['grid'] = not chatstatus['config']['display']['grid']
    if 'xlim' in promptWords:
        loc = promptWords.index('xlim')
        try:
            chatstatus['process']['xlim'] = (int(promptWords[loc + 1]), int(promptWords[loc + 2]))
        except ValueError:
            print('The xlim value is unclear')
    if 'ylim' in promptWords:
        loc = promptWords.index('ylim')
        try:
            chatstatus['process']['ylim'] = (int(promptWords[loc + 1]), int(promptWords[loc + 2]))
        except ValueError:
            print('The ylim value is unclear')
    if 'markerfacecolor' in promptWords:
        loc = promptWords.index('markerfacecolor')
        if is_valid_color(promptWords[loc + 1]):
            chatstatus['process']['markerfacecolor'] = promptWords[loc + 1]
        else:
            print('The markerfacecolor is unclear')
    
    if 'legend' in promptWords:
        loc = promptWords.index('legend')
        chatstatus['process']['legend'] = promptWords[loc + 1].lower() == 'true'
    
    if 'fontsize' in promptWords:
        loc = promptWords.index('fontsize')
        try:
            chatstatus['process']['fontsize'] = int(promptWords[loc + 1])
        except ValueError:
            print('The fontsize value is unclear')
            
    if 'bins' in promptWords:
        loc = promptWords.index('bins')
        try:
            chatstatus['process']['bins'] = int(promptWords[loc + 1])
        except ValueError:
            try:
                chatstatus['process']['bins'] = int(promptWords[loc - 1])
            except ValueError:
                print('The bins value is unclear')
    
    if "spine" in prompt.lower():
        chatstatus['process']['despine'] = not chatstatus['process']['despine']

    if "kde" in prompt.lower():
        chatstatus['process']['kde'] = not chatstatus['process']['kde']
            
    hue_var = should_apply_hue(promptWords, df)
    if hue_var:
        chatstatus['process']['hue'] = hue_var
        
    # get string based parameters such as title, xlabel, and ylabel
    chatstatus = checkPlotLabels(chatstatus)

    return chatstatus

'''
This code was automatically generated to call various seaborn functions based on Transformer input.
It defines helper functions for each seaborn function specified in functionArgs.
'''

def scatterplotCaller(args):
    '''Call seaborn scatterplot function based on the Transformer input.'''
    #print(args)
    #print(args['data'].dtypes)

    ax = sns.scatterplot(
        data           = args['data'],
        x              = args['x'],
        y              = args['y'],
        hue            = args['hue'],
        size           = args['size'],
        style          = args['style'],
        palette        = args['palette'],
        hue_order      = args['hue_order'],
        hue_norm       = args['hue_norm'],
        sizes          = args['sizes'],
        size_order     = args['size_order'],
        size_norm      = args['size_norm'],
        markers        = args['markers'],
        style_order    = args['style_order'],
        legend         = args['legend'],
        ax             = args['ax'],
        marker         = args['marker']
    )
    return ax

def lineplotCaller(args):
    '''Call seaborn lineplot function based on the Transformer input.'''
    ax = sns.lineplot(
        data           = args['data'],
        x              = args['x'],
        y              = args['y'],
        hue            = args['hue'],
        size           = args['size'],
        style          = args['style'],
        units          = args['units'],
        palette        = args['palette'],
        hue_order      = args['hue_order'],
        hue_norm       = args['hue_norm'],
        sizes          = args['sizes'],
        size_order     = args['size_order'],
        size_norm      = args['size_norm'],
        dashes         = args['dashes'],
        markers        = args['markers'],
        style_order    = args['style_order'],
        estimator      = args['estimator'],
        errorbar       = args['errorbar'],
        n_boot         = args['n_boot'],
        seed           = args['seed'],
        orient         = args['orient'],
        sort           = args['sort'],
        err_style      = args['err_style'],
        err_kws        = args['err_kws'],
        legend         = args['legend'],
        ci             = args['ci'],
        ax             = args['ax'],
        kwargs         = args['kwargs'],
    )
    return ax

def stripplotCaller(args):
    '''Call seaborn stripplot function based on the Transformer input.'''
    ax = sns.stripplot(
        data           = args['data'],
        jitter         = args['jitter'],
        dodge          = args['dodge'],
        orient         = args['orient'],
        color          = args['color'],
        palette        = args['palette'],
        size           = args['size'],
        edgecolor      = args['edgecolor'],
        linewidth      = args['linewidth'],
        native_scale   = args['native_scale'],
        formatter      = args['formatter'],
        legend         = args['legend'],
        ax             = args['ax'],
        kwargs         = args['kwargs'],
    )
    return ax

def swarmplotCaller(args):
    '''Call seaborn swarmplot function based on the Transformer input.'''
    ax = sns.swarmplot(
        data           = args['data'],
        dodge          = args['dodge'],
        orient         = args['orient'],
        color          = args['color'],
        palette        = args['palette'],
        size           = args['size'],
        edgecolor      = args['edgecolor'],
        linewidth      = args['linewidth'],
        native_scale   = args['native_scale'],
        formatter      = args['formatter'],
        legend         = args['legend'],
        ax             = args['ax'],
        kwargs         = args['kwargs'],
    )
    return ax

def boxplotCaller(args):
    '''Call seaborn boxplot function based on the Transformer input.'''
    ax = sns.boxplot(
        data           = args['data'],
        x              = args['x'],
        y              = args['y'],
        orient         = args['orient'],
        color          = args['color'],
        palette        = None, #args['palette'],
        saturation     = args['saturation'],
        width          = args['width'],
        dodge          = args['dodge'],
        fliersize      = args['fliersize'],
        linewidth      = args['linewidth'],
        whis           = args['whis'],
        ax             = args['ax'],
        kwargs         = args['kwargs'],
    )
    return ax

def violinplotCaller(args):
    '''Call seaborn violinplot function based on the Transformer input.'''
    ax = sns.violinplot(
        data           = args['data'],
        bw             = args['bw'],
        cut            = args['cut'],
        scale          = args['scale'],
        scale_hue      = args['scale_hue'],
        gridsize       = args['gridsize'],
        width          = args['width'],
        inner          = args['inner'],
        split          = args['split'],
        dodge          = args['dodge'],
        orient         = args['orient'],
        linewidth      = args['linewidth'],
        color          = args['color'],
        palette        = args['palette'],
        saturation     = args['saturation'],
        ax             = args['ax'],
    )
    return ax

def barplotCaller(args):
    '''Call seaborn barplot function based on the Transformer input.'''
    print('Debug BarPlot')
    print(args)
    display(args['data'])
    ax = sns.barplot(
        data           = args['data'],
        estimator      = args['estimator'],
        errorbar       = args['errorbar'],
        n_boot         = args['n_boot'],
        units          = args['units'],
        seed           = args['seed'],
        orient         = args['orient'],
        color          = args['color'],
        palette        = args['palette'],
        saturation     = args['saturation'],
        width          = args['width'],
        errcolor       = args['errcolor'],
        errwidth       = args['errwidth'],
        capsize        = args['capsize'],
        dodge          = args['dodge'],
        ax             = args['ax'],
        kwargs         = args['kwargs'],
    )
    return ax

def countplotCaller(args):
    '''Call seaborn countplot function based on the Transformer input.'''
    ax = sns.countplot(
        data           = args['data'],
        orient         = args['orient'],
        color          = args['color'],
        palette        = args['palette'],
        saturation     = args['saturation'],
        dodge          = args['dodge'],
        ax             = args['ax'],
        kwargs         = args['kwargs'],
    )
    return ax

def histplotCaller(args):
    '''Call seaborn histplot function based on the Transformer input.'''
    ax = sns.histplot(
        data           = args['data'],
        hue            = args['hue'],
        weights        = args['weights'],
        stat           = args['stat'],
        bins           = args['bins'],
        binwidth       = args['binwidth'],
        binrange       = args['binrange'],
        discrete       = args['discrete'],
        cumulative     = args['cumulative'],
        common_bins    = args['common_bins'],
        common_norm    = args['common_norm'],
        multiple       = args['multiple'],
        element        = args['element'],
        fill           = args['fill'],
        shrink         = args['shrink'],
        kde            = args['kde'],
        kde_kws        = args['kde_kws'],
        line_kws       = args['line_kws'],
        thresh         = args['thresh'],
        pthresh        = args['pthresh'],
        pmax           = args['pmax'],
        cbar           = args['cbar'],
        cbar_ax        = args['cbar_ax'],
        cbar_kws       = args['cbar_kws'],
        palette        = args['palette'],
        hue_order      = args['hue_order'],
        hue_norm       = args['hue_norm'],
        color          = args['color'],
        log_scale      = args['log_scale'],
        legend         = args['legend'],
        ax             = args['ax'],
    )
    return ax

def kdeplotCaller(args):
    '''Call seaborn kdeplot function based on the Transformer input.'''
    ax = sns.kdeplot(
        data           = args['data'],
        hue            = args['hue'],
        weights        = args['weights'],
        palette        = args['palette'],
        hue_order      = args['hue_order'],
        hue_norm       = args['hue_norm'],
        color          = args['color'],
        fill           = args['fill'],
        multiple       = args['multiple'],
        common_norm    = args['common_norm'],
        common_grid    = args['common_grid'],
        cumulative     = args['cumulative'],
        bw_method      = args['bw_method'],
        bw_adjust      = args['bw_adjust'],
        warn_singular  = args['warn_singular'],
        log_scale      = args['log_scale'],
        levels         = args['levels'],
        thresh         = args['thresh'],
        gridsize       = args['gridsize'],
        cut            = args['cut'],
        clip           = args['clip'],
        legend         = args['legend'],
        cbar           = args['cbar'],
        cbar_ax        = args['cbar_ax'],
        cbar_kws       = args['cbar_kws'],
        ax             = args['ax'],
    )
    return ax

def ecdfplotCaller(args):
    '''Call seaborn ecdfplot function based on the Transformer input.'''
    ax = sns.ecdfplot(
        data           = args['data'],
        hue            = args['hue'],
        weights        = args['weights'],
        stat           = args['stat'],
        complementary  = args['complementary'],
        palette        = args['palette'],
        hue_order      = args['hue_order'],
        hue_norm       = args['hue_norm'],
        log_scale      = args['log_scale'],
        legend         = args['legend'],
        ax             = args['ax'],
    )
    return ax

def rugplotCaller(args):
    '''Call seaborn rugplot function based on the Transformer input.'''
    ax = sns.rugplot(
        data           = args['data'],
        hue            = args['hue'],
        height         = args['height'],
        expand_margins = args['expand_margins'],
        palette        = args['palette'],
        hue_order      = args['hue_order'],
        hue_norm       = args['hue_norm'],
        legend         = args['legend'],
        ax             = args['ax'],
    )
    return ax

def regplotCaller(args):
    '''Call seaborn regplot function based on the Transformer input.'''
    ax = sns.regplot(
        data           = args['data'],
        x_estimator    = args['x_estimator'],
        x_bins         = args['x_bins'],
        x_ci           = args['x_ci'],
        scatter        = args['scatter'],
        fit_reg        = args['fit_reg'],
        ci             = args['ci'],
        n_boot         = args['n_boot'],
        units          = args['units'],
        seed           = args['seed'],
        order          = args['order'],
        logistic       = args['logistic'],
        lowess         = args['lowess'],
        robust         = args['robust'],
        logx           = args['logx'],
        truncate       = args['truncate'],
        label          = args['label'],
        color          = args['color'],
        marker         = args['marker'],
        ax             = args['ax'],
    )
    return ax

def lmplotCaller(args):
    '''Call seaborn lmplot function based on the Transformer input.'''
    ax = sns.lmplot(
        data           = args['data'],
        palette        = args['palette'],
        col_wrap       = args['col_wrap'],
        height         = args['height'],
        aspect         = args['aspect'],
        markers        = args['markers'],
        legend         = args['legend'],
        legend_out     = args['legend_out'],
        x_estimator    = args['x_estimator'],
        x_bins         = args['x_bins'],
        x_ci           = args['x_ci'],
        scatter        = args['scatter'],
        fit_reg        = args['fit_reg'],
        ci             = args['ci'],
        n_boot         = args['n_boot'],
        units          = args['units'],
        seed           = args['seed'],
        order          = args['order'],
        logistic       = args['logistic'],
        lowess         = args['lowess'],
        robust         = args['robust'],
        logx           = args['logx'],
        truncate       = args['truncate'],
        facet_kws      = args['facet_kws'],
    )
    return ax

def residplotCaller(args):
    '''Call seaborn residplot function based on the Transformer input.'''
    ax = sns.residplot(
        data           = args['data'],
        x              = args['x'],
        y              = args['y'],
        lowess         = args['lowess'],
        order          = args['order'],
        robust         = args['robust'],
        dropna         = args['dropna'],
        label          = args['label'],
        color          = args['color'],
        ax             = args['ax'],
    )
    return ax

def heatmapCaller(args):
    '''Call seaborn heatmap function based on the Transformer input.'''
    ax = sns.heatmap(
        data           = args['data'],
        cmap           = args['cmap'],
        center         = args['center'],
        robust         = args['robust'],
        annot          = args['annot'],
        fmt            = args['fmt'],
        annot_kws      = args['annot_kws'],
        linewidths     = args['linewidths'],
        linecolor      = args['linecolor'],
        cbar           = args['cbar'],
        cbar_kws       = args['cbar_kws'],
        cbar_ax        = args['cbar_ax'],
        square         = args['square'],
        mask           = args['mask'],
        ax             = args['ax'],
        kwargs         = args['kwargs'],
    )
    return ax

def clustermapCaller(args):
    '''Call seaborn clustermap function based on the Transformer input.'''
    ax = sns.clustermap(
        data           = args['data'],
        pivot_kws      = args['pivot_kws'],
        method         = args['method'],
        metric         = args['metric'],
        z_score        = args['z_score'],
        standard_scale = args['standard_scale'],
        figsize        = args['figsize'],
        cbar_kws       = args['cbar_kws'],
        mask           = args['mask'],
        cbar_pos       = args['cbar_pos'],
        tree_kws       = args['tree_kws'],
        kwargs         = args['kwargs'],
    )
    return ax

def pairplotCaller(args):
    '''Call seaborn pairplot function based on the Transformer input.'''
    ax = sns.pairplot(
        data           = args['data'],
        hue            = args['hue'],
        hue_order      = args['hue_order'],
        palette        = args['palette'],
        vars           = args['vars'],
        kind           = args['kind'],
        diag_kind      = args['diag_kind'],
        markers        = args['markers'],
        height         = args['height'],
        aspect         = args['aspect'],
        corner         = args['corner'],
        dropna         = args['dropna'],
    )
    return ax

def jointplotCaller(args):
    '''Call seaborn jointplot function based on the Transformer input.'''
    ax = sns.jointplot(
        data           = args['data'],
        hue            = args['hue'],
        kind           = args['kind'],
        height         = args['height'],
        ratio          = args['ratio'],
        space          = args['space'],
        dropna         = args['dropna'],
        color          = args['color'],
        palette        = args['palette'],
        hue_order      = args['hue_order'],
        hue_norm       = args['hue_norm'],
        marginal_ticks = args['marginal_ticks'],
    )
    return ax

def PairGridCaller(args):
    '''Call seaborn PairGrid function based on the Transformer input.'''
    sns.PairGrid(
    )

def JointGridCaller(args):
    '''Call seaborn JointGrid function based on the Transformer input.'''
    sns.JointGrid(
    )

def catplotCaller(args):
    '''Call seaborn catplot function based on the Transformer input.'''
    ax = sns.catplot(
        data           = args['data'],
        col_wrap       = args['col_wrap'],
        estimator      = args['estimator'],
        errorbar       = args['errorbar'],
        n_boot         = args['n_boot'],
        units          = args['units'],
        seed           = args['seed'],
        height         = args['height'],
        aspect         = args['aspect'],
        kind           = args['kind'],
        native_scale   = args['native_scale'],
        formatter      = args['formatter'],
        orient         = args['orient'],
        color          = args['color'],
        palette        = args['palette'],
        hue_norm       = args['hue_norm'],
        legend         = args['legend'],
        legend_out     = args['legend_out'],
        margin_titles  = args['margin_titles'],
        facet_kws      = args['facet_kws'],
        kwargs         = args['kwargs'],
    )
    return ax

def relplotCaller(args):
    '''Call seaborn relplot function based on the Transformer input.'''
    ax = sns.relplot(
        data           = args['data'],
        hue            = args['hue'],
        size           = args['size'],
        style          = args['style'],
        units          = args['units'],
        col_wrap       = args['col_wrap'],
        palette        = args['palette'],
        hue_order      = args['hue_order'],
        hue_norm       = args['hue_norm'],
        sizes          = args['sizes'],
        size_order     = args['size_order'],
        size_norm      = args['size_norm'],
        style_order    = args['style_order'],
        dashes         = args['dashes'],
        markers        = args['markers'],
        legend         = args['legend'],
        kind           = args['kind'],
        height         = args['height'],
        aspect         = args['aspect'],
        facet_kws      = args['facet_kws'],
        kwargs         = args['kwargs'],
    )
    return ax

