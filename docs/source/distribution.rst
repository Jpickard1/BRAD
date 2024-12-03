Software Installation and Requirements
======================================

This section outlines the requirements and steps required to install the software.
Multiple method of instilation, including `pip`, `conda`, `Docker` and manual instillation
are available, and should be selected according to the users requirements (pip and conda installs are recommended
for development and Docker is recomended for using the tool).
Below are the software dependencies and corresponding installation instructions for different environments.

Software Requirements
----------------------

The software relies on the following packages and libraries:

- Python 3.11 or higher
- Various Python libraries (listed below)
- `pip` or `conda` for package management
- `Docker` (optional, for containerized deployment)

Dependencies
~~~~~~~~~~~~

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

>>> git clone https://github.com/Jpickard1/BRAD/
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
>>> conda activate BRAD-DEV

Docker Instillation
-------------------

Download and install `docker desktop <https://www.docker.com/products/docker-desktop/>`_ and follow the instructions.
Click on download docker and double click the .exe file on windows or .dmg file on mac to start the installation.
You will have to restart the system to get docker desktop running.

Once installed, pull the latest BRAD docker image with the command:

```
docker pull thedoodler/brad:main
```

This will download the BRAD container image and prepare it for deployment. This instillation should take several minutes.

Start the BRAD container using the following command:
```
docker run -e OPENAI_API_KEY=<YOUR_OPENAI_API_KEY> \
           -e PYDANTIC_SKIP_VALIDATING_CORE_SCHEMAS='True' \
           -p 5002:5000 -p 3000:3000 \
           thedoodler/brad:main
```

Replace `<YOUR_OPENAI_API_KEY>` with your OpenAI API key. If using LLMs hosted by NVIDIA, you can include the NVIDIA API key as well:

```
docker run -e OPENAI_API_KEY=<YOUR_OPENAI_API_KEY> \
           -e NVIDIA_API_KEY=<YOUR_NVIDIA_API_KEY> \
           -e PYDANTIC_SKIP_VALIDATING_CORE_SCHEMAS='True' \
           -p 5002:5000 -p 3000:3000 \
           thedoodler/brad:main
```

Once the container is running, open your browser and navigate to `http://localhost:3000 <http://localhost:3000>`_ to access the BRAD GUI.                                              

                                               
                                               
                                               
