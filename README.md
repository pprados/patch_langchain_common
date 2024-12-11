# Install module
For internal use and test the code before the integration inside langchain.

## From well
After a `make init dist`, copy the file `dist/patch_*.whl` to your project and install it with `pip install $(ls patch_*.whl)'[pdf]'`.

## From git clone
To create the VENV and install dependencies:
```
poetry install --extras pdf
poetry shell
```

# Usage
To use the patch, replace
```python
from langchain_community.document_loaders.pdf import *
from langchain_unstructured import *
```
with
```python
from patch_langchain_community.document_loaders.pdf import *
from patch_langchain_unstructured import *
```

