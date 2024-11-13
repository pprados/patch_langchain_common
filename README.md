Langchain-patch-loader
======================

To use the patch, replace
```python
from langchain_community.document_loaders.pdf import *
from langchain_unstructured import *
```
else
```python
from patch_langchain_community.document_loaders.pdf import *
from patch_langchain_unstructured import *
```
