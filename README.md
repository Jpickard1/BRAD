# BRAD: Bioinformatics Digital Assistant

Please see the projects main page available [here!](https://brad-bioinformatics-retrieval-augmented-data.readthedocs.io/_/downloads/en/latest/pdf/)

## Architecture
<div align="center">
  <img width="635" alt="brad-dl-vision" src="https://github.com/user-attachments/assets/da7a1722-28ca-44e8-b45f-4350b7b29305">
</div>

## Installation

Brad can be installed directly from pip:

```
pip install -U BRAD-Chat
```
<br>
<br>

## Web Server deployment

BRAD can be deployed as a web server to interact and experiment with its functionalities.

The BRAD web server is written with react.js for the frontend and flask backend.

The recommended method is to use docker.



### Docker instructions
=======
## [BRAD-Examples](https://github.com/Jpickard1/BRAD-Examples/tree/main)

- **GUI Tutorial**  
  A simple tutorial for how to use BRAD's Graphical User Interface.

- [**Hello World**](https://github.com/Jpickard1/BRAD-Examples/blob/main/Hello-World/Example-0.ipynb)  
  A simple "Hello, World!" example to help you understand the basics of using the BRAD chatbot.

- [**Search and Retrieval-Augmented Generation (RAG)**](https://github.com/Jpickard1/BRAD-Examples/blob/main/RAG-SCRAPE/Example-1.ipynb)  
  Demonstrates how to use BRAD to scrape online data and integrate it into a Retrieval-Augmented Generation pipeline.

- [**Using the Scanpy Package with BRAD**](https://github.com/Jpickard1/BRAD-Examples/blob/main/Scanpy/Example-2.ipynb)  
  Explores how BRAD can streamline workflows involving **Scanpy**, including preprocessing and visualization of single-cell data.

- [**Biomarker Selection Pipeline**](https://github.com/Jpickard1/BRAD-Examples/blob/main/DMD-Biomarkers/Example-3.ipynb)  
  Illustrates how BRAD can assist in selecting biomarkers from datasets using machine learning and bioinformatics tools.

## Development

#### Docker
The Docker build can be used to deploy brad without having to install packages manually.
after installing either docker desktop or docker engine [docker intsallation](https://docs.docker.com/desktop/), you can follow one of the following commands to install and run BRAD


1. Use with docker compose:
```
cd deployment
docker compose up -d
```

2. OR with just docker 

```
docker run -e OPENAI_API_KEY=your_open_ai_key -e  PYDANTIC_SKIP_VALIDATING_CORE_SCHEMAS='True' -p 5001:5000 -p 3000:3000  brad:full_frontend
```

3. To build the dockerfile yourself:

```
docker build -t brad:local .
```

Then proceed to http://localhost:3000 to view the frontend
<br>
<br>

### Development Environment

To contribute or make your own modifications to BRAD, A separate development environment needs to be set up. Follow the instructions below to set up your environment

Download the BRAD repository
```
git clone https://github.com/Jpickard1/BRAD.git
cd BRAD
```
<br>

Update ./BRAD/config/config.json to your preference
( specifically update the log_path key to point somewhere on your local system )
<br>

Make sure python is installed and then install the requirements
>Our recommendation is to use conda and setup a separate conda environment to ensure package discrepencies

```
pip install -r requirements_frozen.txt
```  
<br>

Install node=v20.18.0 and npm=10.8.2 from the (website)[https://nodejs.org/en/download/package-manager]  or by using nvm
>Our recommendation is to use nvm
```
https://raw.githubusercontent.com/creationix/nvm/master/install.sh | bash
source ~/.nvm/nvm.sh \
nvm install 20.18.0 \
nvm use 20.18.0

# install the necessary packages
npm install --prefix ./brad-chat
```

Start BRAD backend with 
```
PYDANTIC_SKIP_VALIDATING_CORE_SCHEMAS=True OPENAI_API_KEY=<your_open_api_key> flask --app app run --host=0.0.0.0 --port=5000
```

Start BRAD frontend with 
```
cd brad-chat
npm start
```

This will start BRAD Backend at - http://localhost:3000  
And the BRAD frontend at - http://localhost:5000

<br>
<br>

### Documentation

To build the projects documentation in the ReadTheDocs html formatting, in the `docs/` directory, run the command `make html`. This will populate the `docs/build/html` directory with the webpages. The `docs/build/` directory is excluded from git but *will* automatically be built when pushing to main.

During the build, the `tutorials/` directory is automatically pulled in to the source by copying all jupyter notebooks. These are included in the user guide section of the documentation.

To remove the documentation from `docs/build/` run `make clean` from the same directory where you built it.

<br>

## Cite Us

```
@article{pickard2024bioinformatics,
  title={Language Model Powered Digital Biology},
  author={Pickard, Joshua and Choi, Marc Andrew and Oliven, Natalie and
          Stansbury, Cooper and Cwycyshyn, Jillian and Galioto, Nicholas
          and Gorodetsky, Alex and Velasquez, Alvaro and Rajapakse, Indika},
  journal={arXiv preprint arXiv:2409.02864},
  url={https://arxiv.org/abs/2409.02864},
  year={2024}
}
```
