import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from patch_langchain_community.document_loaders.parsers.pdf import (
        PDFMinerParser,
        PDFPlumberParser,
        PDFRouterParser,
        PyMuPDF4LLMParser,
        PyMuPDFParser,
        PyPDFium2Parser,
        PyPDFParser,
    )


_module_lookup = {
    "PDFMinerParser": "langchain_community.document_loaders.parsers.pdf",
    "PDFPlumberParser": "langchain_community.document_loaders.parsers.pdf",
    "PyMuPDFParser": "langchain_community.document_loaders.parsers.pdf",
    "PyMuPDF4LLMParser": "langchain_community.document_loaders.parsers.pdf",
    "PyPDFParser": "langchain_community.document_loaders.parsers.pdf",
    "PyPDFium2Parser": "langchain_community.document_loaders.parsers.pdf",
    "PDFRouterParser": "langchain_community.document_loaders.parsers.pdf",
}


def __getattr__(name: str) -> Any:
    if name in _module_lookup:
        module = importlib.import_module(_module_lookup[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__} has no attribute {name}")


__all__ = [
    "PDFMinerParser",
    "PDFPlumberParser",
    "PyMuPDFParser",
    "PyMuPDF4LLMParser",
    "PyPDFParser",
    "PyPDFium2Parser",
    "PDFRouterParser",
]
