# Minimal makefile for Sphinx documentation
#

VENV_PYTHON   := $(abspath ../.venv/bin/python)
ifeq ($(wildcard $(VENV_PYTHON)),)
PYTHON        ?= python
else
PYTHON        ?= $(VENV_PYTHON)
endif
SPHINXOPTS    ?=
SPHINXBUILD   ?= $(PYTHON) -m sphinx
SOURCEDIR     = .
BUILDDIR      = _build

.PHONY: help clean html

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

clean:
	rm -rf "$(BUILDDIR)" "$(SOURCEDIR)/_autosummary"

html:
	@$(MAKE) clean
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
