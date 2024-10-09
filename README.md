# RAG-V1

## Installation

#### Docker
The Docker build can be used to deploy brad without having to install packages manually.
after installing either docker desktop or docker engine [docker intsallation](https://docs.docker.com/desktop/), you can run the following commands to install and run BRAD

To build the dockerfile:

```
docker build -t brad:local .
```

With docker compose:
```
cd deployment
docker compose up -d
```

OR with just docker 

```
docker run -e OPENAI_API_KEY=your_open_ai_key -e  PYDANTIC_SKIP_VALIDATING_CORE_SCHEMAS='True' -p 5001:5000 -p 3000:3000  brad:full_frontend
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

