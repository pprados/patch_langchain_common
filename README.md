# What is it ?
In this repository you can find : 
- the patch (normalization) of the exisiting Langchain's pdf loaders and parsers
- the new pdf loaders and parsers we added 
- the parser comparator tool to compare the parsing results of different parsers on a set of pdfs

This README explains how to use this repository.
For more information about the work that is currently in process of Pull Request go look into the PR_README.md.
This repository is made for internal use and to test the code before the integration inside langchain.

# Installation

## From git clone

**On Linux**

If you don't have poetry, pre-commit and git lfs on your system please install it :
```
sudo apt-get install git-lfs
pip install pre-commit
pip install poetry
```
You may also need if you don't have it yet the following tools :
```
sudo apt-get install xpdf
sudo apt-get install tesseract-ocr
sudo apt-get install tesseract-ocr-eng
sudo apt-get install tesseract-ocr-fra
```
Finally install the patch :
```
git clone ----
cd patch_langchain_common
make init
```

**On Mac**

If you don't have poetry, pre-commit and git lfs on your system please install it :
```
brew install git-lfs
pip install pre-commit
pip insall poetry
```
You may also need if you don't have it yet the following tools :
```
brew install xpdf
brew install tesseract
sudo port install tesseract-eng
sudo port install tesseract-fra
```
Finally install the patch :
```
git clone ----
cd patch_langchain_common
make init
```

# How to synchronise the public repo with my private repo (enterprise use)
The steps are :
- clone the public repo locally : `git clone https://github.com/pprados/patch_langchain_common.git`
- add the private repo as new remote with : `git remote add <private-repo-name> <private-repo-url>`
- push the public clone to the new private remote : `git push <private-repo-name> <branch-name>`

# Packaging

## Build a wheel file if necessary
`make dist`
The wheel can be found in `dist/patch_*.whl`.

If you add a tag with the invocation, it will indicate a version to the weel.
```
git commit -m "v0.0"
git tag v0.0.0
make dist
```

## Install from a wheel 
`pip install $(ls patch_*.whl)'[pdf, parser-comparator]'`
This command is working on Linux OS. You may have to adapt this command depending on your OS.

# Usage
To use the patch in existing code, replace
```python
from langchain_community.document_loaders.pdf import *
from langchain_unstructured import *
```
with
```python
from patch_langchain_community.document_loaders.pdf import *
from patch_langchain_unstructured import *
```

# How to execute notebooks 
Before running the notebooks don't forget to build the distribution at the root of the project using the command
`make dist`.
You can find the notebooks to play with the components in docs/docs directory, either in how to directory or 
integrations/document_loaders directory.
Here is the list of the available notebooks:
- document_loader.ipynb: to learn the ideas behind the patch and how to use it
- new_pdfrouter.ipynb: to play with the new PDFRouter
- pdfminer.ipynb, pdfplumber.ipynb, pymupdf.ipynb, ...: to play with the different patched pdf parsers
- document_loader_custom.ipynb : to learn how to build custom parser/loader and how to use the Generic Loader.

# How to use the parser comparator tool
The parser comparator tool is a tool using the new PDFMultiParser component letting you compare the parsing results of
different parsers on a set of pdfs. You can evaluate the quality of the different parsings either by checking manually 
the quality of the parsing looking at the output files or using the scores automatically computed by some heuristics.
Be careful, currently implemented heuristiics are very basic and may not be relevant to fully evaluate the parsing
quality. You can easily add your own heuristics in the PDFMultiParser component methods.

If you have not done it yet, install the poetry virtual environment following the instructions in the `install 
from git clone` section.
Make sure you have activated this virtual environment with `source .venv/bin/activate`.

The parser comparator tool can be found in the `parser_comparator` directory.
In this directory you have:
  - **sources_pdf** sub-directory: where you should put the pdfs of which you want to compare the parsings. You can 
    group your pdfs in sub-directories so you can organize your experiments based on a specific set of pdfs.

  - **main.py**: the main script where you can
    - select the parsers you want to test by disabling or enabling the parsers families meta parameters such as
      `USE_OLD_PARSERS`, `USE_NEW_PARSERS` and `USE_ONLINE_PARSERS` and/or by block commenting the parsers you want to 
      exclude 
    - tune their parameters using global parameters at the top of the file (e.g. MODE) and/or directly in the 
      declaration of each parser
    - and run the comparison with the following command (make sure terminal is in the `parser_comparator` directory):
      `python3 main.py your_experiment_name`, or simply running the script with your IDE (in this case the experiment 
      name will be `default`)
    
  - **multi_parsing_results** sub-directory: where you can find for each pdf that is in the sources and for each 
    experiment:
    - a directory `parsings_by_parser` containing for each parser :
      - the resulting parsing as a .md file
      - the extracted properties of the pdf as a properties file
    - an excel file containing the automatically computed scores for each parser on this pdf
    - the resulting parsing as a .md file for the best parser according to the scores




