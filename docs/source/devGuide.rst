Interface
=========

The BRAD code offers two distinct interfaces for interaction: a chatbot and a programmatic API. The chatbot provides an interactive experience, accessible via both command line and graphical user interface (GUI). Additionally, the code can be utilized programmatically, similar to other large language model (LLM) tools, and can be integrated with LangChain or similar frameworks. The `Agent` class serves as a cohesive organization for both interaction methods.



Agent Class
-----------

.. automodule:: BRAD.brad
   :members:
   :undoc-members:

Agents as LLMs
--------------

.. automodule:: BRAD.bradllm
   :members:
   :undoc-members:

Graphical User Interface (GUI)
------------------------------

.. toctree::

    gui

Configurations
--------------

.. toctree::

    configs


Core Modules
============

The following modules are the "core" modules of BRAD. These modules contain methods that orchestrate the use of the LLM within modules, standardize logging throughout BRAD, or manage agentic workflows for BRAD.

Large Language Models
---------------------

.. automodule:: BRAD.llms
   :members:
   :undoc-members:

Logging
-------

.. automodule:: BRAD.log
   :members:
   :undoc-members:

Utilities
---------

.. automodule:: BRAD.utils
   :members:
   :undoc-members:

Planner
-------

.. automodule:: BRAD.planner
   :members:
   :undoc-members:

Routing
-------

.. automodule:: BRAD.router
   :members:
   :undoc-members:

Prompt Templates
----------------

.. automodule:: BRAD.promptTemplates
   :members:
   :undoc-members:


Tool Modules
============

The following modules are the "tool" modules of BRAD. These modules contain methods that orchestrate the use of the LLM with different databases, software, or information accessible to the BRAD agents.

Lab Notebook
------------

.. automodule:: BRAD.rag
   :members:
   :undoc-members:

Digital Library
---------------

.. automodule:: BRAD.scraper
   :members:
   :undoc-members:

.. automodule:: BRAD.geneDatabaseCaller
   :members:
   :undoc-members:

.. automodule:: BRAD.enrichr
   :members:
   :undoc-members:

.. automodule:: BRAD.gene_ontology
   :members:
   :undoc-members:

Software
--------

.. automodule:: BRAD.coder
   :members:
   :undoc-members:

.. automodule:: BRAD.pythonCaller
   :members:
   :undoc-members:

