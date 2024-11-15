import logging
import re
from pathlib import Path
from typing import (
    Any,
    Iterator,
    Literal,
    Optional,
    Union,
    Dict,
)

from langchain_community.document_loaders.blob_loaders import Blob
from langchain_core.document_loaders import BaseBlobParser
from langchain_core.documents import Document

from .parsers.new_pdf import LlamaIndexPDFParser, PDFRouterParser, PyMuPDF4LLMParser
from .parsers.pdf import CONVERT_IMAGE_TO_TEXT, _default_page_delimitor
from .pdf import BasePDFLoader
from patch_langchain_community.document_loaders.pdf import BasePDFLoader

from .parsers.new_pdf import PDFMultiParser, PDFRouterParser, PyMuPDF4LLMParser
from .parsers.pdf import _default_page_delimitor

logger = logging.getLogger(__file__)

class PDFMultiLoader(BasePDFLoader):

    def __init__(
            self,
            file_path: str,
            pdf_multi_parser: PDFMultiParser,
            headers: Optional[Dict] = None,
    ) -> None:
        """Load PDF using a multi parser"""
        super().__init__(file_path, headers=headers)
        self.parser = pdf_multi_parser

    def lazy_load(
        self,
    ) -> Iterator[tuple[list[Document], str]]:
        """Lazy load given path as pages."""
        if self.web_path:
            blob = Blob.from_data(open(self.file_path, "rb").read(), path=self.web_path)  # type: ignore[attr-defined]
        else:
            blob = Blob.from_path(self.file_path)  # type: ignore[attr-defined]
        yield from self.parser.parse(blob)



class PDFRouterLoader(BasePDFLoader):
    """
    Load PDFs using different parsers based on the metadata of the PDF.
    The routes are defined as a list of tuples, where each tuple contains
    the regex pattern for the producer, creator, and page, and the parser to use.
    The parser is used if the regex pattern matches the metadata of the PDF.
    Use the route in the correct order, as the first matching route is used.
    Add a default route (None, None, None, parser) at the end to catch all PDFs.

    Sample:
    ```python
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_community.document_loaders.parsers.pdf import PyMuPDFParser
    from langchain_community.document_loaders.parsers.pdf import PyPDFium2Parser
    from langchain_community.document_loaders.parsers import PDFPlumberParser
    routes = [
        ("Microsoft", "Microsoft", None, PyMuPDFParser()),
        ("LibreOffice", None, None, PDFPlumberParser()),
        (None, None, None, PyPDFium2Parser())
    ]
    loader = PDFRouterLoader(filename, routes)
    loader.load()
    ```
    """

    def __init__(
        self,
        file_path: Union[str, Path],
        *,
        routes: list[
            tuple[
                Optional[Union[re.Pattern, str]],
                Optional[Union[re.Pattern, str]],
                Optional[Union[re.Pattern, str]],
                BaseBlobParser,
            ]
        ],
        password: Optional[str] = None,
    ):
        """Initialize with a file path."""
        try:
            import pypdf  # noqa:F401
        except ImportError:
            raise ImportError(
                "pypdf package not found, please install it with `pip install pypdf`"
            )
        super().__init__(file_path)
        self.parser = PDFRouterParser(routes, password=password)

    def lazy_load(
        self,
    ) -> Iterator[Document]:
        if self.web_path:
            blob = Blob.from_data(
                open(self.file_path, "rb").read(), path=self.web_path
            )  # type: ignore[attr-defined]
        else:
            blob = Blob.from_path(self.file_path)  # type: ignore[attr-defined]
        yield from self.parser.lazy_parse(blob)


class PyMuPDF4LLMLoader(BasePDFLoader):
    """Load `PDF` files using `PyMuPDF`."""

    def __init__(
        self,
        file_path: Union[str, Path],
        *,
        password: Optional[str] = None,
        mode: Literal["single", "paged"] = "single",
        pages_delimitor: str = _default_page_delimitor,
        extract_images: bool = False,  # FIXME
        extract_tables: Optional[Literal["markdown"]] = None,  # FIXME
        **kwargs: Any,
    ) -> None:
        """Initialize with a file path."""
        try:
            import pymupdf4llm  # noqa:F401
        except ImportError:
            raise ImportError(
                "`PyMuPDF4LLM` package not found, please install it with "
                "`pip install pymupdf4llm`"
            )
        super().__init__(file_path)
        # if extract_images:
        #     raise NotImplemented("extract_images is not implemented yet.")
        self.parser = PyMuPDF4LLMParser(
            mode=mode,
            pages_delimitor=pages_delimitor,
            password=password,
            to_markdown_kwargs=kwargs,
        )

    def lazy_load(
        self,
    ) -> Iterator[Document]:
        """Lazily load documents."""
        blob = Blob.from_path(self.file_path)  # type: ignore[attr-defined]
        yield from self.parser.lazy_parse(blob)


class LlamaIndexPDFLoader(BasePDFLoader):
    def __init__(
        self,
        file_path: Union[str, Path],
        *,
        password: Optional[str] = None,
        mode: Literal["single", "paged"] = "single",
        pages_delimitor: str = _default_page_delimitor,
        extract_tables: Literal["markdown"] = "markdown",
        api_key: Optional[str] = None,
        verbose: bool = False,
        language: str = "en",
        extract_images: bool = False,
        images_to_text: CONVERT_IMAGE_TO_TEXT = None,
    ):
        try:
            from llama_parse import LlamaParse  # noqa:F401
        except ImportError:
            raise ImportError(
                "llama_parse package not found, please install it "
                "with `pip install llama_parse`"
            )
        super().__init__(file_path)
        if extract_images:
            logger.info("Ignore extract_images==True in LlamaIndexPDFParser.")
        if extract_tables != "markdown" or images_to_text:
            logger.info("Ignore extract_tables!='markdown' in LlamaIndexPDFParser.")
        self.parser = LlamaIndexPDFParser(
            password=password,
            mode=mode,
            pages_delimitor=pages_delimitor,
            extract_images=extract_images,
            images_to_text=images_to_text,
            extract_tables=extract_tables,
            api_key=api_key,
            verbose=verbose,
            language=language,
        )

    def lazy_load(
        self,
    ) -> Iterator[Document]:
        """Lazily load documents."""
        blob = Blob.from_path(self.file_path)  # type: ignore[attr-defined]
        yield from self.parser.lazy_parse(blob)
