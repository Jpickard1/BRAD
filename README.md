# BRAD: Bioinformatics Digital Assistant

Please see the projects main page available [here!](https://brad-bioinformatics-retrieval-augmented-data.readthedocs.io/_/downloads/en/latest/pdf/)

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



## Development

<div align="center">
  <img width="635" alt="brad-dl-vision" src="https://github.com/user-attachments/assets/da7a1722-28ca-44e8-b45f-4350b7b29305">
</div>

### Documentation

To build the projects documentation in the ReadTheDocs html formatting, in the `docs/` directory, run the command `make html`. This will populate the `docs/build/html` directory with the webpages. The `docs/build/` directory is excluded from git but *will* automatically be built when pushing to main.

During the build, the `tutorials/` directory is automatically pulled in to the source by copying all jupyter notebooks. These are included in the user guide section of the documentation.

To remove the documentation from `docs/build/` run `make clean` from the same directory where you built it.

### Cite Us

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
