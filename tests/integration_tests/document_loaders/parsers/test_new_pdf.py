from pathlib import Path

from patch_langchain_community.document_loaders import PDFRouterLoader
from patch_langchain_community.document_loaders.parsers import PyMuPDF4LLMParser, \
    PyMuPDFParser, PDFPlumberParser, PyPDFium2Parser

from .test_pdf_parsers import _assert_with_parser


# PDFs to test parsers on.
HELLO_PDF = Path(__file__).parent.parent.parent / "examples" / "hello.pdf"

def test_pymupdf4llm_parse() -> None:
    _assert_with_parser(PyMuPDF4LLMParser(), splits_by_page=False)

def test_parser_router_parse() -> None:
    routes = [
        ("Microsoft", {"producer": "Microsoft", "creator": "Microsoft"}, PyMuPDFParser()),
        ("LibreOffice", {"producer": "LibreOffice", }, PDFPlumberParser()),
        ("Xdvipdfmx", {"producer": "xdvipdfmx.*", "page1":"Hello"}, PDFPlumberParser()),
        ("default", {}, PyPDFium2Parser())
    ]
    loader = PDFRouterLoader(HELLO_PDF,routes=routes)
    loader.load()
