# Install module
For internal use and test the code before the integration inside langchain.

## From git clone
To create the VENV and install dependencies:
```
poetry install --extras pdf
poetry shell
```

On mac:

```
brew install git-lfs
pipx insall poetry
brew install xpdf
brew install tesseract
sudo port install tesseract-eng
sudo port install tesseract-fra
git clone ...
cd patch_langchain_common
make init
```

# Packaging

## Build a wheel file if necessary
`make dist`
The wheel can be found in `dist/patch_*.whl`

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
You can find the notebooks to play with the components in docs/docs directory.

# How to use the parser comparator tool
If you have not done it yet, install the poetry virtual environment following the instructions in the corresponding 
section.
Make sure you have activated this virtual environment with `source .venv/bin/activate`.

The parser comparator tool can be found in the `parser_comparator` directory.
In this directory you have:
  - sources_pdf sub-directory: where you should put the pdfs of which you want to compare the parsings. You can group your 
    pdfs in sub-directories so you can organize your experiments based on a specific set of pdfs.

  - main.py: the main script where you can
    - select the parsers you want to test by disabling or enabling the parsers families meta parameters such as
      USE_OLD_PARSERS and USE_ONLINE_PARSERS and/or by block commenting the parsers you want to exclude
    - tune their parameters using global parameters at the top of the file (e.g. MODE) and/or directly in the declaration 
      of each parser
    - and run the comparison with the following command (make sure the terminal is in the `parser_comparator` directory):
      `python3 main.py your_experiment_name`, or simply running the script with your IDE (in this case the experiment 
      name will be `default`)
    
  - multi_parsing_results sub-directory: where you can find for each pdf that is in the sources and for each experiment:
      - a directory `parsings_by_parser` containing for each parser :
        - the resulting parsing as a .md file
        - the extracted properties of the pdf as a properties file
      - an excel file containing the automatically computed scores for each parser on this pdf
      - the resulting parsing as a .md file for the best parser according to the scores




