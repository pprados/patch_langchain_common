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

# ruff: noqa

if TYPE_CHECKING:
    from .new_pdf import (
        LlamaIndexPDFLoader,
        PDFRouterLoader,
        PyMuPDF4LLMLoader,
    )
    from .pdf import (
        AmazonTextractPDFLoader,
        DedocPDFLoader,
        MathpixPDFLoader,
        OnlinePDFLoader,
        PagedPDFSplitter,
        PDFMinerLoader,
        PDFMinerPDFasHTMLLoader,
        PDFPlumberLoader,
        PyMuPDFLoader,
        PyPDFDirectoryLoader,
        PyPDFium2Loader,
        PyPDFLoader,
        UnstructuredPDFLoader,
        ZeroxPDFLoader,
    )


_module_lookup = {
    "AmazonTextractPDFLoader": "patch_langchain_community.document_loaders.pdf",
    "DedocPDFLoader": "patch_langchain_community.document_loaders.pdf",
    "MathpixPDFLoader": "patch_langchain_community.document_loaders.pdf",
    "OnlinePDFLoader": "patch_langchain_community.document_loaders.pdf",
    "PDFMinerLoader": "patch_langchain_community.document_loaders.pdf",
    "PDFMinerPDFasHTMLLoader": "patch_langchain_community.document_loaders.pdf",
    "PDFPlumberLoader": "patch_langchain_community.document_loaders.pdf",
    "PagedPDFSplitter": "patch_langchain_community.document_loaders.pdf",
    "PyMuPDFLoader": "patch_langchain_community.document_loaders.pdf",
    "PyPDFDirectoryLoader": "patch_langchain_community.document_loaders.pdf",
    "PyPDFLoader": "patch_langchain_community.document_loaders.pdf",
    "PyPDFium2Loader": "patch_langchain_community.document_loaders.pdf",
    "ZeroxPDFLoader": "patch_langchain_community.document_loaders.pdf",
    "UnstructuredPDFLoader": "patch_langchain_community.document_loaders.pdf",
    "PyMuPDF4LLMLoader": "patch_langchain_community.document_loaders.new_pdf",
    "LlamaIndexPDFLoader": "patch_langchain_community.document_loaders.new_pdf",
    "PDFRouterLoader": "patch_langchain_community.document_loaders.new_pdf",
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
    "PyMuPDFLoader",
    "PyPDFDirectoryLoader",
    "PyPDFLoader",
    "PyPDFium2Loader",
    "ZeroxPDFLoader",
    "UnstructuredPDFLoader",
    "PyMuPDF4LLMLoader",
    "LlamaIndexPDFLoader",
    "PDFRouterLoader",
]
