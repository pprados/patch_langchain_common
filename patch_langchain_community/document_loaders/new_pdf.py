import logging
import re
from pathlib import Path
from typing import (
    Any,
    Iterator,
    Literal,
    Optional,
    Union,
)

from langchain_community.document_loaders.blob_loaders import Blob
from langchain_core.document_loaders import BaseBlobParser
from langchain_core.documents import Document

from .parsers.new_pdf import (
    LlamaIndexPDFParser,
    PDFMultiParser,
    PDFRouterParser,
    PyMuPDF4LLMParser,
)
from .parsers.pdf import CONVERT_IMAGE_TO_TEXT, _default_page_delimitor
from .pdf import BasePDFLoader

logger = logging.getLogger(__file__)


class PDFMultiLoader(BasePDFLoader):
    def __init__(
        self,
        file_path: Union[str, Path],
        *,
        headers: Optional[dict] = None,
        parsers: dict[str, BaseBlobParser],
        max_workers: Optional[int] = None,
        continue_if_error: bool = True,
    ) -> None:
        """Load PDF using a multi parser"""
        super().__init__(file_path, headers=headers)
        self.parser = PDFMultiParser(
            parsers=parsers,
            max_workers=max_workers,
            continue_if_error=continue_if_error,
        )

    def lazy_load(
        self,
    ) -> Iterator[Document]:
        """Lazy load given path as pages."""
        if self.web_path:
            blob = Blob.from_data(open(self.file_path, "rb").read(), path=self.web_path)  # type: ignore[attr-defined]
        else:
            blob = Blob.from_path(self.file_path)  # type: ignore[attr-defined]
        yield from self.parser.parse(blob)


class PDFRouterLoader(BasePDFLoader):
    """
    Load PDFs using different parsers based on the metadata of the PDF
    or the body of the first page.
    The routes are defined as a list of tuples, where each tuple contains
    the name, a dictionary of metadata and regex pattern and the parser to use.
    The special key "page1" is to search in the first page with a regexp.
    Use the route in the correct order, as the first matching route is used.
    Add a default route ("default", {}, parser) at the end to catch all PDFs.

    Sample:
    ```python
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_community.document_loaders.parsers.pdf import PyMuPDFParser
    from langchain_community.document_loaders.parsers.pdf import PyPDFium2Parser
    from langchain_community.document_loaders.parsers import PDFPlumberParser
    routes = [
        # Name, keys with regex, parser
        ("Microsoft", {"producer": "Microsoft", "creator": "Microsoft"},
        PyMuPDFParser()),
        ("LibreOffice", {"producer": "LibreOffice", }, PDFPlumberParser()),
        ("Xdvipdfmx", {"producer": "xdvipdfmx.*", "page1":"Hello"}, PDFPlumberParser()),
        ("defautl", {}, PyPDFium2Parser())
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
                str,
                dict[str, Union[re.Pattern | str]],
                BaseBlobParser,
            ]
        ],
        password: Optional[str] = None,
    ):
        """Initialize with a file path."""
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
        mode: Literal["single", "page"] = "single",
        pages_delimitor: str = _default_page_delimitor,
        extract_images: bool = False,  # FIXME: extract_images in PyMuPDF4LLM
        extract_tables: Optional[
            Literal["markdown"]
        ] = None,  # FIXME: extract_tables in in PyMuPDF4LLM
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
        if self.web_path:
            blob = Blob.from_data(
                open(self.file_path, "rb").read(), path=self.web_path
            )  # type: ignore[attr-defined]
        else:
            blob = Blob.from_path(self.file_path)  # type: ignore[attr-defined]
        yield from self.parser.lazy_parse(blob)


class LlamaIndexPDFLoader(BasePDFLoader):
    def __init__(
        self,
        file_path: Union[str, Path],
        *,
        headers: Optional[dict] = None,
        password: Optional[str] = None,
        mode: Literal["single", "page"] = "single",
        pages_delimitor: str = _default_page_delimitor,
        extract_tables: Literal["markdown"] = "markdown",
        api_key: Optional[str] = None,
        verbose: bool = False,
        language: str = "en",
        extract_images: bool = False,
        images_to_text: CONVERT_IMAGE_TO_TEXT = None,
    ):
        super().__init__(file_path, headers=headers)
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
        try:
            from llama_parse import LlamaParse  # noqa:F401
        except ImportError:
            raise ImportError(
                "llama_parse package not found, please install it "
                "with `pip install llama_parse`"
            )
        if self.web_path:
            blob = Blob.from_data(
                open(self.file_path, "rb").read(), path=self.web_path
            )  # type: ignore[attr-defined]
        else:
            blob = Blob.from_path(self.file_path)  # type: ignore[attr-defined]
        yield from self.parser.lazy_parse(blob)


# TODO: https://www.linkedin.com/posts/liorsinclair_nvidia-just-released-a-powerful-pdf-extraction-ugcPost-7267580522359336962-GAQv/?utm_source=share&utm_medium=member_desktop
