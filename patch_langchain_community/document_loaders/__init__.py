"""**Document Loaders**  are classes to load Documents.

**Document Loaders** are usually used to load a lot of Documents in a single run.

**Class hierarchy:**

.. code-block::

    BaseLoader --> <name>Loader  # Examples: TextLoader, UnstructuredFileLoader

**Main helpers:**

.. code-block::

    Document, <name>TextSplitter
"""

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .pdf import (
        AmazonTextractPDFLoader,
        DedocPDFLoader,
        MathpixPDFLoader,
        OnlinePDFLoader,
        PagedPDFSplitter,
        PDFMinerLoader,
        PDFMinerPDFasHTMLLoader,
        PDFPlumberLoader,
        PDFRouterLoader,
        PyMuPDF4LLMLoader,
        PyMuPDFLoader,
        PyPDFDirectoryLoader,
        PyPDFLoader,
        PyPDFium2Loader,
        UnstructuredPDFLoader,
    )


_module_lookup = {
    "AmazonTextractPDFLoader": "langchain_community.document_loaders.pdf",
    "DedocPDFLoader": "langchain_community.document_loaders.pdf",
    "MathpixPDFLoader": "langchain_community.document_loaders.pdf",
    "OnlinePDFLoader": "langchain_community.document_loaders.pdf",
    "PDFMinerLoader": "langchain_community.document_loaders.pdf",
    "PDFMinerPDFasHTMLLoader": "langchain_community.document_loaders.pdf",
    "PDFPlumberLoader": "langchain_community.document_loaders.pdf",
    "PDFRouterLoader": "langchain_community.document_loaders.pdf",
    "PagedPDFSplitter": "langchain_community.document_loaders.pdf",
    "PyMuPDFLoader": "langchain_community.document_loaders.pdf",
    "PyMuPDF4LLMLoader": "langchain_community.document_loaders.pdf",
    "PyPDFDirectoryLoader": "langchain_community.document_loaders.pdf",
    "PyPDFLoader": "langchain_community.document_loaders.pdf",
    "PyPDFium2Loader": "langchain_community.document_loaders.pdf",
    "UnstructuredPDFLoader": "langchain_community.document_loaders.pdf",
}


def __getattr__(name: str) -> Any:
    if name in _module_lookup:
        module = importlib.import_module(_module_lookup[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__} has no attribute {name}")


__all__ = [
    "AmazonTextractPDFLoader",
    "DedocPDFLoader",
    "MathpixPDFLoader",
    "OnlinePDFLoader",
    "PDFMinerLoader",
    "PDFMinerPDFasHTMLLoader",
    "PDFPlumberLoader",
    "PagedPDFSplitter",
    "PDFRouterLoader",
    "PyMuPDFLoader",
    "PyMuPDF4LLMLoader",
    "PyPDFDirectoryLoader",
    "PyPDFLoader",
    "PyPDFium2Loader",
    "UnstructuredPDFLoader",
]
