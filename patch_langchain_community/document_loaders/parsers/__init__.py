import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from patch_langchain_community.document_loaders.parsers.pdf import (
        PDFMinerParser,
        PDFPlumberParser,
        PyMuPDFParser,
        PyPDFium2Parser,
        PyPDFParser,
    )
from patch_langchain_community.document_loaders.parsers.new_pdf import (
    LlamaIndexPDFParser,
    PDFRouterParser,
    PyMuPDF4LLMParser,
)

_module_lookup = {
    "PDFMinerParser": "patch_langchain_community.document_loaders.parsers.pdf",
    "PDFPlumberParser": "patch_langchain_community.document_loaders.parsers.pdf",
    "PyMuPDFParser": "patch_langchain_community.document_loaders.parsers.pdf",
    "PyPDFParser": "patch_langchain_community.document_loaders.parsers.pdf",
    "PyPDFium2Parser": "patch_langchain_community.document_loaders.parsers.pdf",
    "PyMuPDF4LLMParser": "patch_langchain_community.document_loaders.parsers.new_pdf",
    "LlamaIndexPDFParser": "patch_langchain_community.document_loaders.parsers.new_pdf",
    "PDFRouterParser": "patch_langchain_community.document_loaders.parsers.new_pdf",
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
    "PyPDFParser",
    "PyPDFium2Parser",
    "PyMuPDF4LLMParser",
    "LlamaIndexPDFParser",
    "PDFRouterParser",
]
