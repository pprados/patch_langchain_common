from pathlib import Path

from .test_pdf_parsers import _assert_with_parser
from patch_langchain_community.document_loaders.parsers import PyMuPDF4LLMParser


def test_pymupdf4llm_parse() -> None:
    _assert_with_parser(PyMuPDF4LLMParser(),splits_by_page=False)