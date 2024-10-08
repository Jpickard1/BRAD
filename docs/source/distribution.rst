Software Installation and Requirements
======================================

This section outlines the steps required to install and configure the software, providing options for installation via `pip`, `conda`, and `Docker`. Below are the software dependencies and corresponding installation instructions for different environments.

Software Requirements
----------------------

The software relies on the following packages and libraries:

- Python 3.11 or higher
- Various Python libraries (listed below)
- `pip` or `conda` for package management
- `Docker` (optional, for containerized deployment)

Python Dependencies
~~~~~~~~~~~~~~~~~~~

The following libraries are required and can be installed using `pip` or `conda`. These are listed in the `requirements.txt` and `environment.yml` files.

**Key Libraries:**
- `transformers`
- `torch>=2.0.1`
- `sentence-transformers`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `gget`
- `biopython`
- `requests`
- `beautifulsoup4`
- `semantic-router`
- `langchain`

Refer to the full list of dependencies in the `requirements.txt` and `environment.yml` files for exact versions.

Pip Installation
----------------

To install the required dependencies using `pip`, follow these steps:

1. **Ensure Python 3.11 or higher is installed.**
   - You can download Python from `python.org <https://www.python.org/downloads/>`_.

2. **Install dependencies from `requirements.txt`:**

   First, clone the repository and navigate to the root directory:

>>> git clone <repo_url>
>>> cd BRAD                                               
>>> pip install -r requirements.txt


This will install all necessary Python packages, including torch, transformers, langchain, and more.

Conda Instillation
------------------

Alternatively, you can install the dependencies using `conda`. A `conda` environment file is provided to easily set up the project environment:

Install Conda: If Conda is not already installed, download and install it from `here <https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html>`_.

In the root directory of the project, run the following command to create an environment named `BRAD-1`


>>> conda env create -f environment.yml
>>> conda list                                               
>>> conda activate BRAD-1

Docker Instillation
-------------------


>>> docker build -t brad:local .
>>> docker run -it brad:local
                                              

                                               
                                               
                                               
