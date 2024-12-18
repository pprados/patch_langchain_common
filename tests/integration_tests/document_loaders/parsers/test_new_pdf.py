from pathlib import Path

from patch_langchain_community.document_loaders import PDFRouterLoader
from patch_langchain_community.document_loaders.parsers import PyMuPDF4LLMParser, \
    PyMuPDFParser, PDFPlumberParser, PyPDFium2Parser

from .test_pdf_parsers import _assert_with_parser


# PDFs to test parsers on.
HELLO_PDF = Path(__file__).parent.parent.parent / "examples" / "hello.pdf"
LAYOUT_PARSER_PAPER_PDF = (
    Path(__file__).parent.parent.parent / "examples" / "layout-parser-paper.pdf"
)



def test_pymupdf4llm_parse() -> None:
    _assert_with_parser(PyMuPDF4LLMParser(), splits_by_page=False)


def test_docling_parse() -> None:
    loader = DoclingPDFLoader(file_path=LAYOUT_PARSER_PAPER_PDF,
                              mode="single")
    docs = loader.load()
    pprint(docs)
    # _assert_with_parser(PyMuPDF4LLMParser(), splits_by_page=False)


def test_parser_router_parse() -> None:
    mode = "single"
    routes = [
        (
            "Microsoft",
            {"producer": "Microsoft", "creator": "Microsoft"},
            PyMuPDFParser(mode=mode),
        ),
        (
            "LibreOffice",
            {
                "producer": "LibreOffice",
            },
            PDFPlumberParser(mode=mode),
        ),
        (
            "Xdvipdfmx",
            {"producer": "xdvipdfmx.*", "page1": "Hello"},
            PDFPlumberParser(mode=mode),
        ),
        ("default", {}, PyPDFium2Parser(mode=mode)),
    ]
    _assert_with_parser(PDFRouterParser(routes=routes), splits_by_page=False)
