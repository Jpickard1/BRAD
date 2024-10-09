# RAG-V1

## Installation

### Local Install

### Install with PIP

### Reuirements and conda

## Usage
See `BRAD/llms.py`

### OpenAI's GPT

### NVIDIA's LLMs

### Local LLM's

## Development

### Documentation

To build the projects documentation in the ReadTheDocs html formatting, in the `docs/` directory, run the command `make html`. This will populate the `docs/build/html` directory with the webpages. The `docs/build/` directory is excluded from git but *will* automatically be built when pushing to main.

During the build, the `tutorials/` directory is automatically pulled in to the source by copying all jupyter notebooks. These are included in the user guide section of the documentation.

To remove the documentation from `docs/build/` run `make clean` from the same directory where you built it.

