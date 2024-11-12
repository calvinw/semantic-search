SHELL:=/bin/bash

SOURCES:= \
SemanticSearch.qmd \
Small-Similarity-Example.qmd \
HandMDresses.qmd

REGULAR_HTML_FILES = $(SOURCES:%.qmd=%.html)
REGULAR_IPYNB_FILES = $(SOURCES:%.qmd=%.ipynb)

all: html ipynb
	@echo All files are now up to date

clean:
	@echo Removing files...
	rm -rf $(REGULAR_HTML_FILES) $(REGULAR_IPYNB_FILES) *_files

html: $(REGULAR_HTML_FILES)

ipynb: $(REGULAR_IPYNB_FILES)

# Rule for regular SOURCES
$(REGULAR_HTML_FILES): %.html: %.qmd
	quarto render $< --to html

$(REGULAR_IPYNB_FILES): %.ipynb: %.qmd
	quarto render $< --to ipynb --no-execute

.PHONY: all clean html ipynb
