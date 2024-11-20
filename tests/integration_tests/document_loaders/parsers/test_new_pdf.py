import unittest
from unittest.mock import Mock

from patch_langchain_community.document_loaders.parsers.new_pdf import PDFMultiParser
from patch_langchain_community.document_loaders.parsers import PyMuPDF4LLMParser

from .test_pdf_parsers import _assert_with_parser


def test_pymupdf4llm_parse() -> None:
    _assert_with_parser(PyMuPDF4LLMParser(), splits_by_page=False)
