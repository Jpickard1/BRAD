# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

# JP modified to use pdflatex
LATEX         = pdflatex

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Clean target to remove the build directory and source/tutorials/
clean:
	@$(SPHINXBUILD) -M clean "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Custom target for building PDF using pdflatex
latexpdf:
	@$(SPHINXBUILD) -b latex "$(SOURCEDIR)" "$(BUILDDIR)/latex"
	# @cd "$(BUILDDIR)/latex" && $(LATEX) bradbioinformaticsretrievalaugmenteddata.tex
	# @cd "$(BUILDDIR)/latex" && $(LATEX) bradbioinformaticsretrievalaugmenteddata.tex
	# @cd "$(BUILDDIR)/latex" && $(LATEX) bradbioinformaticsretrievalaugmenteddata.tex
	# @cd "$(BUILDDIR)/latex" && $(LATEX) bradbioinformaticsretrievalaugmenteddata.tex

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
