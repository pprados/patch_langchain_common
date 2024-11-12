from pathlib import Path
from typing import Sequence, Union

import pytest
from patch_langchain_unstructured import document_loaders

from patch_langchain_community.document_loaders import (
    AmazonTextractPDFLoader,
    MathpixPDFLoader,
    PDFMinerLoader,
    PDFMinerPDFasHTMLLoader,
    PDFPlumberLoader,
    PyMuPDF4LLMLoader,
    PyMuPDFLoader,
    PyPDFium2Loader,
    UnstructuredPDFLoader,
)

# PPR : a faire
