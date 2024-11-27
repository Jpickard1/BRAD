"""
The `brad` module serves as the main interface for user interactions, whether through a graphical user interface (GUI), command line, or programmatically.

The `Agent` class creates a single chatbot instance that can be queried in various ways. 

The `AgentFactory` class is a session factory method, used to fetch and maintain Agent sessions from outside the module.  

The `brad.chat` method allows users to initiate a command line chat session without needing to create an `Agent` instance.

Main Methods
-------------

Main Methods:
    
1. `Agent.chat`
    This method creates a chat session where a user and `Agent` can have a conversation with back-and-forth inputs.
2. `Agent.invoke`
    This method responds to an individual user query with a single tool.
3. `AgentFactory.get_agent`
    Method to instantiate a new 

.. _state-schema-section:

State Schema
------------

The `Agent` state is managed within a dictionary called `Agent.state`. This dictionary tracks the agents inputs, outputs, memory, configurations, and more.
To pass this information between the `Agent` and each tool the `state` dictionary is passed as the single input to each tool module. The `state` is structured as:

>>> Agent.state = {
... 'config'            : {
...     <configuration variables>
... },
... 'prompt'            : <user input>,
... 'output'            : <streaming output of Agent>,
... 'memory'            : <agent memory>,
... 'process'           : {
...     'MODULE'        : <Tool module used to respond to user input>,
...     <module specific information>: {
...         ...
...     }
... },
... 'queue'             : [<list of instructions to follow>],
... 'queue pointer'     : <instruction pointer to the queue>,
... 'llm-api-calls'     : <number of LLM calls used by Agent>,
... 'recursion_depth'   : <amount of recursion the Agent is using>
>>> }
    
Class Methods
-------------

The `Agent` class is organized as follows:


"""

# Standard
import pandas as pd
import os
import shutil
from datetime import datetime as dt
import warnings
import json
import logging
import time
from typing import Optional, List
import pickle
import atexit

# Router
from semantic_router.layer import RouteLayer

# RAG
import chromadb

# LangChain
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain.memory import ConversationBufferMemory

# LangChain Core
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM

# Library
from BRAD.planner import *
from BRAD.enrichr import *
from BRAD.scraper import *
from BRAD.router import *
from BRAD.rag import queryDocs, remove_repeats
from BRAD.gene_ontology import *
from BRAD.pythonCaller import *
from BRAD.llms import *
from BRAD.geneDatabaseCaller import geneDBRetriever
from BRAD.planner import planner
from BRAD.coder import code_caller
from BRAD.writer import summarizeSteps, chatReport
from BRAD import log
from BRAD.bradllm import BradLLM
from BRAD.constants import TOOL_MODULES

class Agent():
    """
    This class organizes the agentic capabilities of BRAD. It facilitates interactions 
    with external LLMs, tools, core modules, literature, and other databases while
    managing the chat state and history.
    
    Key functions include:
    
    1. **invoke(user_input)**: Responds to a single user input.
    2. **chat()**: Initiates an interactive session between the user and the BRAD agent.
    
    To address user queries, the agent employs semantic routing to select the appropriate tool module, generates responses using code from the chosen module, and tracks its state throughout the interaction.

    :param model_path: The path to the Llama model file, defaults to '/nfs/turbo/umms-indikar/shared/projects/RAG/models/llama-2-7b-chat.Q8_0.gguf'.
    :type model_path: str, optional
    :param persist_directory: The directory where the literature database is stored, defaults to "/nfs/turbo/umms-indikar/shared/projects/RAG/databases/Transcription-Factors-5-10-2024/".
    :type persist_directory: str, optional
    :param llm: The language model to be used. If None, it will be loaded within the function.
    :type llm: PreTrainedModel, optional
    :param ragvectordb: The RAG vector database to be used. If None, it will prompt the user to load it.
    :type ragvectordb: Chroma, optional
    :param embeddings_model: The embeddings model to be used. If None, it will be loaded within the function.
    :type embeddings_model: HuggingFaceEmbeddings, optional
    :param max_api_calls: The maximum number of api / llm calls BRAD can make
    :type max_api_calls: int, optional
    :param tools: The set of available tool modules. If None, all modules are available for use
    :type tools: list, optional
    :param gui: Indicates if the Agent is used in the GUI
    :type gui: boolean, optional

    :raises FileNotFoundError: If the specified model or database directories do not exist.
    :raises json.JSONDecodeError: If the configuration file contains invalid JSON.
    :raises KeyError: If required keys are missing from the configuration or chat status.
    """

    
    def __init__(self,
        model_path = '/nfs/turbo/umms-indikar/shared/projects/RAG/models/llama-2-7b-chat.Q8_0.gguf',
        persist_directory = "/nfs/turbo/umms-indikar/shared/projects/RAG/databases/DigitalLibrary-10-June-2024/",
        llm=None,
        ragvectordb=None,
        embeddings_model=None,
        restart=None,
        start_path=None,
        tools=None,
        name='BRAD',
        max_api_calls=None, # This prevents BRAD from finding infinite loops and using all your API credits,
        interactive=True,   # This indicates if BRAD is in interactive more or not
        config=None,        # This parameter lets a user specify an additional configuration file that will
                            # overwrite configurations with the same key
        gui=False
    ):
        # Auth: Joshua Pickard
        #       jpic@umich.edu
        # Date: June 4, 2024

        # Dev. Comments:
        # -------------------
        # This function initializes a chat session and the state variable
        #
        # History:
        # - 2024-06-04: wrote 1st draft of this code in the brad.chat() method
        # - 2024-07-10: refactored brad.py file to a class and converted the code
        #               used to initialize the chat session to initialize this class
        # - 2024-07-23: added interactive and max_api_call arguments
        # - 2024-07-29: added config (optional) argument to overwrite the defaults
        # - 2024-10-06: .chatstatus was renamed .state
        # - 2024-10-16: if the restart location doesn't have a log, then a new agent
        #               is created
        # - 2024-11-18: configurations are automatically read/written from created sessions
        #               when turning the Agent back on
        #
        # Issues:
        # - We should change the structure of the classes/modules. In this 1st iteration
        #   state was packed as a class variable and used similar to before, but it
        #   is likely reasonable to split this variable up into separate class variables.
        # - Current configurations only allow either a specific configuration file or a
        #   configuration file from a previous chat session (dominate) - JP

        self.state = self.load_state(config=config)
        self.name       = name.strip()
        self.state['interactive'] = interactive # By default a chatbot is not interactive
        self.state['gui'] = gui
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
        base_dir = os.path.expanduser('~')
        log_dir = os.path.join(base_dir, self.state['config']['log_path'])
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # Check if it will be possible to restart the old session
        if restart is not None:
            log_path = os.path.join(restart, 'log.json')
            if not os.path.exists(log_path):
                restart = None

        if restart is None:
            # making this more human readable
            new_dir_name = start_path if start_path else dt.now().strftime("%B %d, %Y at %I:%M:%S %p")
            
            new_log_dir = os.path.join(log_dir, new_dir_name)
            try:
                os.makedirs(new_log_dir, exist_ok=True)
            except Exception as e:
                print(f"Failed to create directory '{new_log_dir}'. Error: {e}")
                fallback_dir = "C:\\Users\\jpic\\Documents\\BRAD-logs"  # Replace this with a directory that must exist
                print(f"Using fallback directory: {fallback_dir}")
                new_log_dir = fallback_dir
                os.makedirs(new_log_dir, exist_ok=True)  # This should always succeed since the fallback exists
            self.chatname = os.path.join(new_log_dir, 'log.json')
            self.chatlog  = {}
        else:
            new_log_dir = restart
            self.chatname = os.path.join(restart, 'log.json')
            self.chatlog  = json.load(open(self.chatname))

            state_file = os.path.join(new_log_dir, '.agent-state.pkl')
            
            if os.path.exists(state_file):
                try:
                    with open(state_file, 'rb') as f:
                        self.state = pickle.load(f)
                    logging.info(f"Loaded agent state from {state_file}")
                except Exception as e:
                    logging.error(f"Failed to load agent state from {state_file}: {e}")
            else:
                logging.info(f"No existing state file found in {new_log_dir}, starting fresh.")

            # Update from the old configurations
            config_file = os.path.join(new_log_dir, 'config.json')
            if os.path.exists(config_file):
                try:
                    saved_configs = self.load_config(config_file)
                    self.state['config'] = saved_configs
                    logging.info(f"Loaded configuration from {config_file}")
                except Exception as e:
                    logging.error(f"Failed to load configuration from {config_file}: {e}")
            else:
                logging.info(f"No configuration file found at {config_file}.")

        if max_api_calls is None:
            max_api_calls = 1000
        self.max_api_calls = max_api_calls
    
        # Initialize the dictionaries of tables and databases accessible to BRAD
        databases = {} # a dictionary to look up databases
        # tables = {}    # a dictionary to look up tables
    
        # Initialize the RAG database
        if llm is None:

            # By devault we use OpenAI
            llm = load_openai()

        if ragvectordb is None:
            if self.state['interactive']:
                state = log.userOutput('\nWould you like to use a database with ' + self.name + ' [Y/N]?', state=self.state)
                loadDB = input().strip().upper()
                if loadDB == 'Y':
                    ragvectordb, embeddings_model = self.load_literature_db(persist_directory) # load the literature database
                else:
                    ragvectordb, embeddings_model = None, None
            else:
                ragvectordb, embeddings_model = None, None
        
        databases['RAG'] = ragvectordb
        memory = ConversationBufferMemory(ai_prefix="BRAD")
    
        # Initialize the routers from router.py
        self.router = getRouter(available=tools)
    
        # Add other information to state
        # Assign values only if the key does not exist or is None/empty
        if 'llm' not in self.state or not self.state['llm']:
            self.state['llm'] = llm

        if 'memory' not in self.state or not self.state['memory']:
            self.state['memory'] = memory

        if 'databases' not in self.state or not self.state['databases']:
            self.state['databases'] = databases

        if 'output-directory' not in self.state or not self.state['output-directory']:
            self.state['output-directory'] = new_log_dir
    
        # Initialize all modules
        self.module_functions = self.getModules()
    
        # Start loop
        # only log if chat bot is fresh
        if restart is None:
            self.state = log.userOutput('Welcome to BRAD! The output from this conversation will be saved to ' + self.chatname + '. How can I help?', state=self.state)

            # Write an empty chat log
            self.chatlog, self.state = log.logger(self.chatlog, self.state, self.chatname, elapsed_time=0)

        # Ensure that the save_state function is registered to run at program exit
        # atexit.register(self.save_state)


    def save_state(self):
        """
        Saves the agent state to a file named '.agent-state.pkl' in the output directory.
        This method is registered with atexit to ensure it is called when the program exits.
        """
        # Auth: Joshua Pickard
        #       jpic@umich.edu
        # Date: October 15, 2024

        output_directory = self.state['output-directory']
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        state_file = os.path.join(output_directory, '.agent-state.pkl')

        try:
            # Save the log to the json log file
            self.state['prompt'] = None
            self.state['output'] = None
            self.state['process'] = {
                'module' : 'SLEEP'
            }

            # Set the databases to none
            self.state['databases']['RAG'] = None
            self.state['llm'] = None

            # Save the state to a pickle file
            with open(state_file, 'wb') as f:
                pickle.dump(self.state, f)
            logging.info(f"Agent state saved to {state_file}")

            self.chatlog, self.state = log.logger(self.chatlog, self.state, self.chatname, elapsed_time=0)
            logging.info(f"Agent log written for power off")

        except Exception as e:
            logging.error(f"Failed to save agent state: {e}")


    def invoke(self, query):
        """
        Executes a single query using the chatbot, similar to invoking a language model.
    
        This method processes the user input, determines the appropriate routing 
        based on explicit commands or the content of the query, and generates a 
        response using the selected module. It also manages the state of the 
        chatbot throughout the execution.
    
        :param query: The user input to be processed by the chatbot.
        :type query: str
    
        :return: The output generated by the chatbot in response to the input query.
        :rtype: str
    
        :raises Exception: If an error occurs during the execution of the selected module.
    
        :example:
            >>> response = agent.invoke("What's the weather today?")
        
        :note: 
            Special commands recognized include:
            - "exit", "quit", "q", "bye": Ends the session.
            - "help": Displays help information.
            - "/set": Configures settings.
            - "/force": Forces the use of a specified routing function.
    
        This method logs the process and clears memory based on configuration settings.
        """
        # Auth: Joshua Pickard
        #       jpic@umich.edu
        # Date: June 4, 2024

        # Dev. Comments:
        # -------------------
        # This function executes a single user prompt with BRAD
        #
        # History:
        # - 2024-06-04: wrote 1st draft of this code in the brad.chat() method
        # - 2024-07-10: refactored brad.py file to a class and converted the code
        #               used to execute a single prompt into the invoke method

        # start the clock
        start_time = time.time()
        
        # This line packs a query into the state variable we were using previously
        self.state['prompt'] = query
        
        # Handle explicit commands and routing
        if self.state['prompt'].lower() in ['exit', 'quit', 'q', 'bye']:         # check to exit
            return False
        elif self.state['prompt'].lower() == 'help':              # print man to the screen
            self.chatbotHelp()
            return True
            # continue
        # Routing
        elif self.state['prompt'].startswith('/set'):             # set a configuration variable
            self.state = self.reconfig()
            return True
            # continue
        # Continue previous module
        elif self.state.get('continue-module') is not None:
            route = self.state['continue-module'][0]

        # Use router to select correct module
        elif '/force' not in self.state['prompt'].split(' '):     # use the router
            route = self.router(self.state['prompt']).name
            if route is None:
                route = 'RAG'
        # Forced routes
        else:
            route = self.state['prompt'].split(' ')[1]            # use the forced router
            if self.state['config']['ROUTER']['build router db']:
                buildRoutes(self.state['prompt'])
            self.state['prompt'] = " ".join(self.state['prompt'].split(' ')[2:]).strip()

        # Outputs
        self.state = log.userOutput('==================================================', state=self.state)
        self.state = log.userOutput(self.name + ' >> ' + str(len(self.chatlog)) + ': ', state=self.state)

        # select appropriate routeing function
        if route.upper() in self.module_functions.keys():
            routeName = route.upper()
        else:
            routeName = 'RAG'

        log.debugLog(routeName, state=self.state)

        # get the specific module to use
        module = self.module_functions[routeName]

        # Standardize state/logging schema for all modules
        #     see: https://github.com/Jpickard1/RAG-DEV/wiki/Logging
        self.state['process'] = {'module' : routeName,
                                 'steps'  : []
                            }
        self.state['output'] = ""

        # get current output files
        output_files = utils.outputFiles(self.state)
        IP = self.state['queue pointer']
        # if not self.state['interactive']:
        #     self.state['queue'] = [[]] # don't let it us an empty queue if it is not interactive
        
        # Query module
        try:
            self.state = module(self.state)
        except:
            log.debugLog('An error occurred while using a tool.', state=self.state)

        # Remove the item that was executed. We need must do it after running it for the current file naming system.
        log.debugLog('\n\n\nroute\n\n\n', state=self.state)
        log.debugLog(route, state=self.state)
        if len(self.state['queue']) != self.state['queue pointer'] and route != 'PLANNER':
            # log.debugLog(self.state['queue'], state=self.state)
            new_output_files = utils.outputFiles(self.state)
            new_output_files = list(set(new_output_files).difference(set(output_files)))
            if 'output' not in self.state['queue'][IP].keys():
                self.state['queue'][IP]['outputs'] = new_output_files
            self.state = utils.makeNamesConsistent(self.state, new_output_files)
            if self.state['process']['module'] != 'ROUTER':
                self.state['queue pointer'] += 1

        # Clear memory
        if self.state['config']['forgetful']:
            self.state['memory'].clear()

        # stop the clock
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        # Log and reset these values
        self.chatlog, self.state = log.logger(
            self.chatlog,
            self.state,
            self.chatname,
            elapsed_time=elapsed_time
        )
        return self.state['output']

    def chat(self):
        """
        Opens an interactive chat session where users can execute a series of prompts.
    
        This method allows users to engage in a back-and-forth dialogue with the chatbot. 
        Users can input queries, which the chatbot processes and responds to until 
        the session is terminated. It supports both direct user input and queued prompts.
    
        The chat session maintains a record of the conversation and tracks the number 
        of API calls made to language models. It ensures that the session can be exited 
        gracefully and provides feedback about the conversation's context.
    
        :example:
            >>> agent.chat()
        
        :note:
            The session continues until the user explicitly decides to exit by 
            inputting commands like "exit", "quit", or "bye".
    
            If a queue of prompts is available, the chatbot will process 
            them in sequence rather than waiting for user input.
    
            The memory can be temporarily integrated to enrich queries, 
            but its management is handled with care to avoid unintended 
            modifications.
        """
        # Auth: Joshua Pickard
        #       jpic@umich.edu
        # Date: June 4, 2024

        # Dev. Comments:
        # -------------------
        # This function executes a series of prompts in an interactive mannor with BRAD
        #
        # History:
        # - 2024-06-04: wrote 1st draft of this code in the brad.chat() method
        # - 2024-07-10: refactored brad.py file to a class and converted the code
        #               used to execute consecutive prompts into this function
        # - 2024-07-21: added llm-api-calls to chat status to prevent the rerouting/
        #               planner modules from executing unnecessarily long loops.
        self.state['interactive'] = True
        while True:
            # Begin processing a new prompt
            print('==================================================')

            # By default the memory is not messed with in BRAD (but on special occasions it is!)
            resetMemory = False

            # Get the prompt from the user or from the queue
            if len(self.state['queue']) != 0 and self.state['queue pointer'] < len(self.state['queue']) and self.state['queue pointer'] != 0 and not self.state['interactive']:
                query = self.state['queue'][self.state['queue pointer']]['prompt']

                # update memory to use previous points of the pipeline
                if 'inputs' in list(self.state['queue'][self.state['queue pointer']].keys()):
                    print(query)
                    print(self.state['queue'][self.state['queue pointer']]['module'])
                    if self.state['queue'][self.state['queue pointer']]['module'] == 'RAG':
                        print('RAG MODULE')
                        print(f"self.state['queue'].keys()={self.state['queue'].keys()}")
                        inputStages = self.state['queue'][self.state['queue pointer']]['inputs']
                        print(inputStages)
                        query += "**Previous Work**\nYou have generated the following data previously in the pipeline:\n\n"
                        for ISIP in inputStages:
                            fileName  = self.state['queue'][ISIP]['outputs']
                            print(fileName)
                            for file in fileName:
                                file = os.path.join(self.state['output-directory'], file)
                                if not os.path.exists(file):
                                    continue
                                try:
                                    print(file)
                                    dfMemory  = pd.read_csv(file)
                                    if 'state' in df.columns:
                                        query += "Top Ranked Biomarkers:\n"
                                        query += str(df['state'].values[:100])[1:-1]
                                    if 'p_val' in df.columns:
                                        query += "Relevant Biological Processes from Enrichr:"
                                        query += str(df[['path_name']].values[:10])
                                except:
                                    print('this part of the code doesnt work so great, yet!')
            else:
                query = input('Input >> ')  # get query from user
            
            # This line needs to change to check if we should exist the chat session
            output = self.invoke(query)

            # output is false on exit
            if not output:
                break

            # reset memory
            # if resetMemory:
            #    self.resetMemory()
                
            # update llm-api-calls
            newCalls = self.getLLMcalls(self.state['process']['steps'])
            self.state['llm-api-calls'] += newCalls
            log.debugLog(f'current llm calls={self.state["llm-api-calls"]}', state=self.state)
            if self.state['llm-api-calls'] > self.max_api_calls:
                log.debugLog('The maximum number of llm calls has been exceeded', state=self.state)
                break
        self.state['interactive'] = False
        self.state = log.userOutput("Thanks for chatting today! I hope to talk soon, and don't forget that a record of this conversation is available at: " + self.chatname, state=self.state)

    def get_display(self):
        """
        This function returns the history of all inputs/outputs to the agent. This is intended
        for use by the GUI, as it will allow the user to jump between sessions while loading in
        the history of the old session.

        :returns: a list of strings
        :rtype: list
        """
        log.debugLog("Get Display Method", display=True)
        # numIOpairs = len(self.chatlog.keys())
        display = []
        log.debugLog("display = []", display=True)
        for i in self.chatlog.keys():
            if 'module' in self.chatlog[i]['process'].keys() and self.chatlog[i]['process']['module'] == 'SLEEP':
                continue
            elif 'MODULE' in self.chatlog[i]['process'].keys() and self.chatlog[i]['process']['MODULE'] == 'SLEEP':
                continue
            else:
                display.append((self.chatlog[i]['prompt'], None))
                display.append((self.chatlog[i]['output'], self.chatlog[i]))
        log.debugLog(f"{display=}", display=True)
        return display

    @property
    def llm(self):
        """Get the current LLM."""
        return self.state['llm']

    def set_llm(self, llm):
        """Set the LLM and handle any related logic."""
        if not llm:
            raise ValueError("LLM cannot be None")
        
        self.state['llm'] = llm

    def updateMemory(self):
        """
        Thils function lets BRAD reset his memory to focus on specific previous interactions. This is useful
        when BRAD is executing a pipeline and want to manage how input flows from one section to the next.

        .. warning::
            This function may be removed in the near future.
            
        """
        # Auth: Joshua Pickard
        #       jpic@umich.edu
        # Date: July 28, 2024

        log.debugLog('updateMemory is starting', state=state)
        
        # get previous stages that must occur in the input
        inputs = self.state['queue'][self.state['queue pointer']]['input']

        # build a new memory object based upon the previous inputs and outputs of each stage
        memory = ConversationBufferMemory()
        for stage in inputs:
            stageHistory = self.state['queue'][stage]['history']
            memory.add_user_message(stageHistory['input'][-1])
            memory.add_ai_message(stageHistory['output'][-1])

        # Update the state main memory
        state['main memory'] = state['memory']
        state['memory'] = memory
        return
                

    def resetMemory(self):
        """
        Resets the state of the memory by restoring the main memory from the recent messages.
    
        This function undoes the effects of the `updateMemory` method by taking
        the most recent messages from the stage memory and adding them back 
        into the main memory. It is designed to help maintain a consistent 
        memory state throughout the agent's execution.
    
        .. warning::
            This function may be removed in the near future.
    
        :return: None
        :rtype: None
    
        :example:
            >>> agent.resetMemory()
        """
        # Auth: Joshua Pickard
        #       jpic@umich.edu
        # Date: July 28, 2024

        # Update the state main memory
        mainMemory = state['main memory']
        stageMemory = state['main memory']
        
        recent_messages = stageMemory.buffer[-2:]  # Adjust -2 index to control # of recent messages
        for msg in recent_messages:
            mainMemory.add_message(msg['role'], msg['content'])

        state['memory'] = mainMemory
        return
    
    def getLLMcalls(self, steps):
        """
        Counts the number of times the LLM has been called used the agent's execution.
        
        :param steps: A list of steps from the agents log that have been executed
        :type steps: list
        
        :return: The total number of LLM calls made by the agent.
        :rtype: int
        
        :example:
            >>> num_calls = agent.getLLMcalls(steps)
        """
        # Auth: Joshua Pickard
        #       jpic@umich.edu
        # Date: July 2024
        newLLMcalls = 0
        emptyLLMlog = log.llmCallLog()
        for step in steps:
            if all(k in step.keys() for k in emptyLLMlog.keys()):
                newLLMcalls += 1
        return newLLMcalls
    
    def getModules(self):
        """
        Returns a dictionary mapping module names to their corresponding function handles for various tasks.
    
        :param None: This function does not take any parameters.
    
        :raises None: This function does not raise any specific errors.
    
        :return: A dictionary where the keys are module names and the values are function handles for tasks such as querying Enrichr, web scraping, generating seaborn plots, querying documents, and calling Snakemake.
        :rtype: dict
    
        """
        # Auth: Joshua Pickard
        #       jpic@umich.edu
        # Date: June 4, 2024
        
        # this + imports should be the only code someone needs to write to add a new module
        module_functions = {
            'DATABASE'   : geneDBRetriever,     # gget
            'SCRAPE' : webScraping,      # webscrapping
            'RAG'    : queryDocs,        # standard rag
            'PYTHON' : callPython,
            'PLANNER': planner,
            'CODE'   : code_caller,
            'WRITE'  : chatReport, # summarizeSteps,
            'ROUTER' : reroute,
        }
        return module_functions

    def load_config(self, configfile=None):
        """
        Loads the `Agent` configuration settings from a JSON file.
    
        :param None: This function does not take any parameters.
    
        :raises FileNotFoundError: If the configuration file is not found.
        :raises json.JSONDecodeError: If the configuration file contains invalid JSON.
    
        :return: A dictionary containing the configuration settings.
        :rtype: dict
        """
        # Auth: Joshua Pickard
        #       jpic@umich.edu
        # Date: June 4, 2024

        # History:
        # - 2024-06-04: wrote 1st draft of this code
        # - 2024-07-29: added a new argument to allow additional configuration
        #               files to overwrite the defaults

        def deep_update(original, updates):
            """
            Recursively update the original dictionary with the updates dictionary.
            """
            for key, value in updates.items():
                if isinstance(value, dict) and key in original:
                    original[key] = deep_update(original.get(key, {}), value)
                else:
                    original[key] = value
            return original
        
        # Read the default configurations
        current_script_path = os.path.abspath(__file__)
        current_script_dir = os.path.dirname(current_script_path)
        file_path = os.path.join(current_script_dir, 'config', 'config.json')
        with open(file_path, 'r') as f:
            defaultConfigs = json.load(f)

        # print(defaultConfigs)

        # Read the specific configurations
        if configfile:
            with open(configfile, 'r') as f:
                newConfigs = json.load(f)
            defaultConfigs = deep_update(defaultConfigs, newConfigs)

        # print(defaultConfigs)
        return defaultConfigs    
    
    def save_config(self):
        """
        Saves the agent configuration settings to a JSON file.
    
        :param config: A dictionary containing the configuration settings to be saved.
        :type config: dict
    
        :raises FileNotFoundError: If the directory for the configuration file is not found.
        :raises TypeError: If the configuration dictionary contains non-serializable values.    
        """
        # Dev. Comments
        # History:
        # - 2024-11-18: This method modifies the objects configurations but does not change
        #               the configurations of the package. (JP)
        #
        # Auth: Joshua Pickard
        #       jpic@umich.edu
        # Date: June 4, 2024

        # current_script_path = os.path.abspath(__file__)
        # current_script_dir = os.path.dirname(current_script_path)
        # file_path = os.path.join(current_script_dir, 'config', 'config.json')
        file_path = os.path.join(self.chatname[:-8], 'config.json')
        print(f"{file_path=}")
        with open(file_path, 'w') as f:
            json.dump(self.state['config'], f, indent=4)

    def reconfig(self):
        """
        Updates a specific configuration setting based on the given chat status and saves the updated configuration.
    
        :param chat_status: A dictionary containing the current chat status, including the prompt and configuration.
        :type chat_status: dict
    
        :raises KeyError: If the specified configuration key is not found in the chat status.
        :raises ValueError: If the value cannot be converted to an integer or float when applicable.
    
        :return: The updated chat status dictionary.
        :rtype: dict
    
        """
        # Auth: Joshua Pickard
        #       jpic@umich.edu
        # Date: June 4, 2024
        
        prompt = self.state['prompt']
        _, key, value = prompt.split(maxsplit=2)
        if key == 'debug':
            value = (value.lower() == 'true')
        try:
            value = int(value)
        except ValueError:
            try:
                value = float(value)
            except ValueError:
                value = str(value)
        if key in self.state['config']:
            state['config'][key] = value
            self.save_config()
            state = log.userOutput("Configuration " + str(key) + " updated to " + str(value), state=self.state)
        else:
            state = log.userOutput("Configuration " + str(key) + " not found", state=self.state)
        return state

    def load_state(self, config=None):
        """
        Initializes and loads the agent state with default values and configuration settings.
    
        :param None: This function does not take any parameters.
    
        :raises FileNotFoundError: If the configuration file is not found.
        :raises json.JSONDecodeError: If the configuration file contains invalid JSON.
    
        :return: A dictionary representing the chat status with initial values and loaded configuration.
        :rtype: dict
        """
        # Auth: Joshua Pickard
        #       jpic@umich.edu
        # Date: June 4, 2024
    
        state = {
            'config'            : self.load_config(configfile=config),
            'prompt'            : None,
            'output'            : None,
            'process'           : {},
            'current table'     : {'key':None, 'tab':None},
            'current documents' : None,
            'tables'            : {},
            'documents'         : {},
            'plottingParams'    : {},
            'matlabEng'         : None,
            'experiment'        : False,
            'queue'             : [],
            'queue pointer'     : 0,
            'llm-api-calls'     : 0,
            'search'            : {
                'used terms' : [],
            },
            'recursion_depth': 0,
            'continue-module': None, # None (if not continuing in a module) or tuple (with [0] being the module name)
            'gui':None,              # boolean to indicate if the agent is for the gui
            'interactive':False
        }
        return state

    def load_literature_db(
        self,
        persist_directory = "/home/acicalo/BRAD/data/RAG_Database",
        db_name = "DB_cosine_cSize_700_cOver_200"
    ):
        """
        Loads a literature database using specified embedding model and settings.
    
        :param persist_directory: The directory where the database is stored, defaults to "/nfs/turbo/umms-indikar/shared/projects/RAG/databases/Transcription-Factors-5-10-2024/"
        :type persist_directory: str, optional
    
        :raises FileNotFoundError: If the specified directory does not exist or is inaccessible.
        :raises Warning: If the loaded database contains no articles.
    
        :return: A tuple containing the vector database and the embeddings model.
        :rtype: tuple

        The `persist_directory` should point to a directory that has this structure:

        >>> [Oct 16 19:28]  persist_directory
        ... └── [Oct 16 19:28]  DB_cosine_cSize_700_cOver_200
        ...     ├── [Oct 16 19:28]  aaa2c989-0e39-4be8-82b4-139ae2784c00
        ...     │   ├── [Oct 16 19:28]  data_level0.bin
        ...     │   ├── [Oct 16 19:28]  header.bin
        ...     │   ├── [Oct 16 19:28]  length.bin
        ...     │   └── [Oct 16 19:28]  link_lists.bin
        >>>     └── [Oct 16 19:28]  chroma.sqlite3
    
        """
        # Dev. Comments
        # History:
        # - 2024-06-04: 1st version of this was written
        # - 2024-10-17: changes to the pathing were made

        # Auth: Joshua Pickard
        #       jpic@umich.edu
        # Date: June 4, 2024
    
        # load database
        embeddings_model = HuggingFaceEmbeddings(model_name='BAAI/bge-base-en-v1.5')        # Embedding model
        _client_settings = chromadb.PersistentClient(path=os.path.join(persist_directory, db_name))
        vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings_model, client=_client_settings, collection_name=db_name)
        if len(vectordb.get()['ids']) == 0:
            print('The loaded database contains no articles. See the database: ' + str(persist_directory) + ' for details')
            warnings.warn('The loaded database contains no articles. See the database: ' + str(persist_directory) + ' for details')
        return vectordb, embeddings_model

    def chatbotHelp(self):
        """
        Displays a help message to the user with information about the BRAD agents's capabilities and special commands.
        """
        # Auth: Joshua Pickard
        #       jpic@umich.edu
        # Date: May 19, 2024
        help_message = """
        Welcome to our RAG chatbot Help!
        
        You can chat with the llama-2 llm and several scientifici databases including:
        - retrieval augmented generation
        - web scraping
        - search enrichr and gene ontology databases
        - run codes
        - and more!
    
        Special commands:
        /set   - Allows the user to change configuration variables.
        /force - Allows the user to specify which database to use.
    
        For example:
        /set config_variable_name new_value
        --force option_name
    
        Enjoy chatting with the chatbot!
        """
        self.state = log.userOutput(help_message, state=self.state)
        return

    def to_langchain(self):
        """
        This function constructs an object that may be used as an llm in langchain.

        :return: a LangChain compatible LLM instance
        :rtype: `BradLLM`
        """
        # Auth: Joshua Pickard
        #       jpic@umich.edu
        # Date: July 10, 2024
        llm = BradLLM(bot=self)
        return llm


class AgentFactory():
    """
    The AgentFactory mechanism allows us to instantiate, terminate and maintain 
    bot sessions from within the module. Removes the need to have global agents.
    The factory generates a default agent with default parameters if no input is
    given. Based on the session input given instantiates a new agent with that 
    particular session restored

    Provides decoupling of objects used from execution logic. If a new agent class 
    is implemented the get_agent function needs to be updated appropriately.

    Functions:
    1. **get_agent**: instantiates the actual agent and returns the particular agent based on initialization


    :param tools: The set of available tool modules. If None, all modules are available for use
    :type tools: list, optional
    :param session_path: The path to where the bot session is stored. If None, generates a new agent
    :type session_path: str, optional
    :param start_path: The path to where a new bot session can be started. If None, generates a new agent
    :type start_path: str, optional
    :param interactive: Sets BRAD's mode to interactive or non inteactive. Default mode is non Interactive
    :type tools: bool, optional
    """

    def __init__(self, tool_modules=TOOL_MODULES, session_path=None, start_path=None, interactive=False, db_name=None, persist_directory=None, llm_choice=None, gui=None, temperature=0):
        self.interactive = interactive
        suffix = '/log.json'
        if session_path and (session_path.endswith(suffix)):
            session_path = session_path[: -len(suffix)]
        self.session_path = session_path
        self.tool_modules = tool_modules
        self.start_path = start_path
        self.gui=gui
        if db_name:
            self.db_name = db_name
        else:
            self.db_name = None

        if persist_directory:
            self.persist_directory = persist_directory
        else:
            self.persist_directory = None

        self.llm_choice = llm_choice
        self.temperature = temperature
        

    def get_agent(self):
        """
        The agent function for instantiating a new agent or retrieve an existing agent
        """
        if self.session_path:
            agent = Agent(interactive=self.interactive, tools=self.tool_modules, restart=self.session_path, gui=self.gui)
        elif self.start_path:
            agent = Agent(interactive=self.interactive, tools=self.tool_modules, start_path=self.start_path, gui=self.gui)
        else:
            agent = Agent(interactive=self.interactive, tools=self.tool_modules, gui=self.gui)


        if self.db_name and self.persist_directory:
            db, _ = agent.load_literature_db(persist_directory=self.persist_directory, db_name=self.db_name)
            agent.state['databases']['RAG'] = db
        elif self.db_name == None and self.persist_directory==None:
            agent.state['databases']['RAG'] = None

        if self.llm_choice:
            llm = llm_switcher(
                self.llm_choice,
                temperature = self.temperature,
            )
            agent.set_llm(llm)
        return agent