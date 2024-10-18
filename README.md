# BRAD: Bioinformatics Digital Assistant

Pleas see the projects main page available [here!](https://brad-bioinformatics-retrieval-augmented-data.readthedocs.io/_/downloads/en/latest/pdf/)

### Documentation

To build the projects documentation in the ReadTheDocs html formatting, in the `docs/` directory, run the command `make html`. This will populate the `docs/build/html` directory with the webpages. The `docs/build/` directory is excluded from git but *will* automatically be built when pushing to main.

During the build, the `tutorials/` directory is automatically pulled in to the source by copying all jupyter notebooks. These are included in the user guide section of the documentation.

To remove the documentation from `docs/build/` run `make clean` from the same directory where you built it.

