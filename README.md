# RAG-V1

## Installation

#### Docker
The Docker build can be used to deploy brad without having to install packages manually.
after installing either docker desktop or docker engine [docker intsallation](https://docs.docker.com/desktop/), you can run the following commands to install and run BRAD

```
docker build -t brad:local .
docker compose up -d
```

then proceed to http://localhost:3000


### Local Install

### Install with PIP

## Usage

### Locall LLMs

### NVIDIA Hosted LLMs

## Development

### Documentation

To build the projects documentation in the ReadTheDocs html formatting, in the `docs/` directory, run the command `make html`. This will populate the `docs/build/html` directory with the webpages. The `docs/build/` directory is excluded from git but *will* automatically be built when pushing to main.

During the build, the `tutorials/` directory is automatically pulled in to the source by copying all jupyter notebooks. These are included in the user guide section of the documentation.

To remove the documentation from `docs/build/` run `make clean` from the same directory where you built it.

### Coding

