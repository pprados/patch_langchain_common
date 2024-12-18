"""Module contains common parsers for PDFs."""

import logging
import os
import re
import sys
from io import BytesIO
from typing import (
    Any,
    BinaryIO,
    Iterator,
    Literal,
    Optional,
    Union,
    cast,
)

from docling_core.types.doc import ImageRefMode

if sys.version_info < (3, 11):  # FIXME: (3,11)
    pass

from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from langchain_community.document_loaders.base import BaseBlobParser
from langchain_community.document_loaders.blob_loaders import Blob
from langchain_core.documents import Document

from .pdf import (
    CONVERT_IMAGE_TO_TEXT,
    ImagesPdfParser,
    PDFMinerParser,
    PyMuPDFParser,
    _default_page_delimitor,
    purge_metadata,
)

logger = logging.getLogger(__name__)


class PDFMultiParser(BaseBlobParser):
    def __init__(
        self,
        parsers: dict[str, BaseBlobParser],
        *,
        max_workers: Optional[int] = None,
        continue_if_error: bool = True,
    ) -> None:
        """"""
        self.parsers = parsers
        self.max_workers = max_workers
        self.continue_if_error = continue_if_error

    def lazy_parse(
        self,
        blob: Blob,
    ) -> Iterator[Document]:
        """Lazily parse the blob. (Fakely because for each parser all Documents
        need to be loaded at once in order to
        compute the global parsing score.)"""
        parsers_results = self.parse_and_evaluate(blob)
        best_parsing_documents = parsers_results[0][1]
        for document in best_parsing_documents:
            document.metadata["parser_name"] = parsers_results[0][0]

        return iter(best_parsing_documents)

    @lru_cache(maxsize=5)
    @staticmethod
    def _thread_pool_executor(max_workers: int) -> ThreadPoolExecutor:
        return ThreadPoolExecutor(max_workers=max_workers)

    def parse_and_evaluate(
        self,
        blob: Blob,
    ) -> list[tuple[str, list[Document], dict[str, float]]]:
        """Parse the blob with all parsers and return the results as a dictionary
        {parser_name: (documents, metrics)}"""
        parsers_results = []
        all_exceptions: dict[str, Exception] = {}
        executor = PDFMultiParser._thread_pool_executor(self.max_workers)
        # Submit each parser's load method to the executor
        futures = {
            executor.submit(parser.parse, blob): parser_name
            for parser_name, parser in self.parsers.items()
        }
        # Collect the results from the futures as they complete
        for future in as_completed(futures):
            parser_name = futures[future]
            try:
                documents = future.result()
                # print(f"{parser_name} \u001B[32m completed \u001B[0m")
                # print(f"documents list for parser {parser_name} :", documents)
                metric_name2score = self.evaluate_parsing_quality(documents)
                parsers_results.append((parser_name, documents, metric_name2score))
            except Exception as e:
                log = f"Parser {parser_name} failed with exception : {e}"
                logger.warning(log)
                all_exceptions[parser_name] = e

        # TODO: si tu ne veux pas que ça continue en cas d'erreur et qu'il y a des
        # erreurs alors exception
        if not self.continue_if_error and all_exceptions:
            raise ExceptionGroup(
                "Some parsers have failed.", list(all_exceptions.values())
            )

        # TODO si tous les parsers sont en erreur, soulever une exception
        if len(all_exceptions) == len(self.parsers):
            raise ExceptionGroup(
                "All parsers have failed.", list(all_exceptions.values())
            )

        # sort parsers results by global score
        parsers_results.sort(key=lambda x: x[2]["global_score"], reverse=True)
        return parsers_results

    def evaluate_parsing_quality(
        self,
        documents_list: list[Document],
    ) -> dict[str, float]:
        """Evaluate the quality of a parsing based on some metrics measured
        by heuristics.
        Return the dictionary {key=metric_name: value=score}"""
        metric_methods = [
            getattr(self, m) for m in dir(self) if m.startswith("metric_")
        ]

        metric_name2score = {}
        concatenated_docs = "\n".join([doc.page_content for doc in documents_list])
        for method in metric_methods:
            metric_name2score[method.__name__.split("metric_")[1]] = method(
                concatenated_docs
            )

        global_score = self.compute_global_parsing_score(metric_name2score)
        metric_name2score["global_score"] = global_score
        return metric_name2score

    def metric_tables(
        self,
        content: str,
    ) -> float:
        """Evaluate the quality of tables identification in a document."""
        tables_score = 0
        patterns = [
            r"(?s)("
            r"(?:(?:[^\n]*\|)\n)"
            r"(?:\|(?:\s?:?---*:?\s?\|)+)\n"
            r"(?:(?:[^\n]*\|)\n)+"
            r")",
            r"(?s)(<table[^>]*>(?:.*?)<\/table>)",
            r"((?:(?:"
            r'(?:"(?:[^"]*(?:""[^"]*)*)"'
            r"|[^\n,]*),){2,}"
            r"(?:"
            r'(?:"(?:[^"]*(?:""[^"]*)*)"'
            r"|[^\n]*))\n){2,})",
        ]

        for pattern in patterns:
            matches = re.findall(pattern, content)
            if matches:
                tables_score += len(matches)
        return tables_score

    def metric_titles(
        self,
        content: str,
    ) -> float:
        """Evaluate the quality of titles identification in a document."""
        title_score = 0
        titles_tags_weights = {
            r"# ": np.exp(0),
            r"## ": np.exp(1),
            r"### ": np.exp(2),
            r"#### ": np.exp(3),
            r"##### ": np.exp(4),
            r"###### ": np.exp(5),
        }

        for title, weight in titles_tags_weights.items():
            pattern = re.compile(rf"^{re.escape(title)}", re.MULTILINE)
            matches = re.findall(pattern, content)
            title_score += len(matches) * weight

        return title_score

    def metric_lists(
        self,
        content: str,
    ) -> float:
        """Evaluate the quality of lists identification in a document."""
        lists_score = 0
        list_regex = re.compile(
            r"^([ \t]*)([-*+•◦▪·o]|\d+([./]|(\\.))) .+", re.MULTILINE
        )

        matches = re.findall(list_regex, content)
        for match in matches:
            indent = match[0]  # get indentation
            level = len(indent)
            lists_score += (
                level + 1
            )  # the more indent the parser identify the more it is rewarded

        return lists_score

    def compute_global_parsing_score(
        self,
        metric_name2score: dict[str, float],
    ) -> np.floating[Any]:
        """Compute the global parsing score based on the scores of each metric."""
        return np.mean(list(metric_name2score.values()))


class PyMuPDF4LLMParser(ImagesPdfParser):
    """Parse `PDF` using `PyMuPDF`."""

    def __init__(
        self,
        *,
        password: Optional[str] = None,
        mode: Literal["single", "page"] = "single",
        pages_delimitor: str = _default_page_delimitor,
        to_markdown_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        """Initialize the parser.

        Args:
            password: Password to open the PDF.
            mode: Extraction mode to use. Either "single" or "page".
            pages_delimitor: Delimiter to use between pages.
                             May be r'\f', '<!--PAGE BREAK -->', ...

            to_markdown_kwargs: Keyword arguments to pass to the PyMuPDF4LLM
             extraction method.
        """
        super().__init__(
            extract_images=False,  # PPR: extract_images will be True
            images_to_text=None,
        )
        if mode not in ["single", "page"]:
            raise ValueError("mode must be single or page")
        self.mode = mode
        self.pages_delimitor = pages_delimitor
        self.password = password or ""
        if not to_markdown_kwargs:
            to_markdown_kwargs = {}
        if "show_progress" not in to_markdown_kwargs:
            to_markdown_kwargs["show_progress"] = False
        _to_markdown_kwargs = cast(dict[str, Any], to_markdown_kwargs or {})
        _to_markdown_kwargs["page_chunks"] = True
        self.to_markdown_kwargs = _to_markdown_kwargs

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:  # type: ignore[valid-type]
        """Lazily parse the blob."""
        try:
            import pymupdf
            import pymupdf4llm
        except ImportError:
            raise ImportError(
                "pymupdf4llm package not found, please install it "
                "with `pip install pymupdf4llm`"
            )
        with PyMuPDFParser._lock:
            with blob.as_bytes_io() as file_path:  # type: ignore[attr-defined]
                if blob.data is None:  # type: ignore[attr-defined]
                    doc = pymupdf.open(file_path)
                else:
                    doc = pymupdf.open(stream=file_path, filetype="pdf")
                if doc.is_encrypted:
                    doc.authenticate(self.password)

                full_text = []
                metadata: dict[str, Any] = {}
                for mu_doc in pymupdf4llm.to_markdown(
                    doc,
                    **self.to_markdown_kwargs,
                ):
                    if self.mode == "single":
                        full_text.append(mu_doc["text"])
                        if not metadata:
                            metadata = mu_doc["metadata"]
                    elif self.mode == "page":
                        yield Document(
                            page_content=mu_doc["text"],
                            metadata=purge_metadata(mu_doc["metadata"]),
                        )
                        # PPR: extraire images ? See PyMuPDFParser
                        # PPR: extraire array ? See PyMuPDFParser
                if self.mode == "single":
                    yield Document(
                        page_content=self.pages_delimitor.join(full_text),
                        metadata=purge_metadata(metadata),
                    )

    _map_key = {"page_count": "total_pages", "file_path": "source"}
    _date_key = ["creationdate", "moddate"]


class DoclingPDFParser(ImagesPdfParser):
    """Parse `PDF` using `PyMuPDF`."""

    def __init__(
            self,
            *,
            password: Optional[str] = None,
            mode: Literal["single", "page"] = "single",
            pages_delimitor: str = _default_page_delimitor,
            images_to_text: CONVERT_IMAGE_TO_TEXT = None,
            extract_tables: Optional[Literal["markdown"]] = "markdown",
    ) -> None:
        """Initialize the parser.

        Args:
            password: Password to open the PDF.
            mode: Extraction mode to use. Either "single" or "page".
            pages_delimitor: Delimiter to use between pages.
                             May be r'\f', '<!--PAGE BREAK -->', ...

        """
        super().__init__(
            extract_images=False,  # PPR: extract_images will be True
            images_to_text=None,
        )
        if mode not in ["single", "page"]:
            raise ValueError("mode must be single or page")
        if password:
            raise ValueError("Password is not implemented")
        if extract_tables != "markdown":
            logger.warning("DoclingPDFParser accept only markdown format")
        self.mode = mode
        self.pages_delimitor = pages_delimitor

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:  # type: ignore[valid-type]
        """Lazily parse the blob."""
        try:
            from docling.document_converter import DocumentConverter
            from docling_core.types.io import DocumentStream
        except ImportError:
            raise ImportError(
                "docling package not found, please install it "
                "with `pip install docling pdfminer`"
            )
        converter = DocumentConverter()

        doc_metadata = purge_metadata(self._get_metadata(blob))
        result = converter.convert(
            DocumentStream(
                name=blob.path or blob.source,
                stream=BytesIO(blob.as_bytes())))
        if self.mode == "single":
            text = result.document.export_to_markdown(
                image_mode=ImageRefMode.EMBEDDED,
                # image_placeholder="<!-- image -->"
            )
            yield Document(page_content=text,
                           metadata=doc_metadata)
        elif self.mode == "page":
            for page_no in range(len(result.pages)):
                text = result.document.export_to_markdown(
                    page_no=page_no+1,
                    image_mode= ImageRefMode.EMBEDDED,
                )
                yield Document(page_content=text,
                               metadata={**doc_metadata,"page":page_no})

    def _get_metadata(
            self,
            blob: Blob,
    ) -> dict[str, Any]:

        from pdfminer.pdfpage import PDFDocument, PDFPage, PDFParser

        with blob.as_bytes_io() as fp:

            # Create a PDF parser object associated with the file object.
            parser = PDFParser(fp)
            # Create a PDF document object that stores the document structure.
            doc = PDFDocument(parser)
            metadata = {}

            for info in doc.info:
                metadata.update(info)
            for k, v in metadata.items():
                try:
                    metadata[k] = PDFMinerParser.resolve_and_decode(v)
                except Exception as e:  # pragma: nocover
                    # This metadata value could not be parsed. Instead of failing the PDF
                    # read, treat it as a warning only if `strict_metadata=False`.
                    logger.warning(
                        '[WARNING] Metadata key "%s" could not be parsed due to '
                        "exception: %s",
                        k,
                        str(e),
                    )

            # Count number of pages.
            metadata["total_pages"] = len(list(PDFPage.create_pages(doc)))

            return metadata


class PDFRouterParser(BaseBlobParser):
    """
    Parse PDFs using different parsers based on the metadata of the PDF.
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
        ("Microsoft", "Excel", None, PyMuPDFParser()),
        ("Microsoft", "Word", None, ZeroxPDFParser()),
        ("LibreOffice", None, None, PDFPlumberParser()),
        (None, None, None, PyPDFium2Parser())
    ]
    loader = PDFRouterLoader(filename, routes)
    loader.load()
    ```
    """

    # {"metadata":r"regex"},
    # doc_regex = r"regex"
    def __init__(
        self,
        routes: list[
            tuple[
                str,
                dict[str, Union[re.Pattern | str]],
                BaseBlobParser,
            ]
        ],
        *,
        password: Optional[str] = None,
    ):
        """Initialize with a file path."""
        try:
            import pypdf  # noqa:F401
        except ImportError:
            raise ImportError(
                "pypdf package not found, please install it with `pip install pypdf`"
            )
        super().__init__()
        self.password = password
        new_routes = []
        for name, matchs, parser in routes:
            new_matchs = {}
            for k, v in matchs.items():
                if isinstance(v, str):
                    v = re.compile(v)
                new_matchs[k] = v
            new_routes.append((name, new_matchs, parser))
        self.routes = new_routes

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:  # type: ignore[valid-type]
        """Lazily parse the blob."""
        try:
            import pypdf  # noqa:F401
        except ImportError:
            raise ImportError(
                "pypdf package not found, please install it with `pip install pypdf`"
            )
        from pypdf import PdfReader

        with blob.as_bytes_io() as pdf_file_obj:  # type: ignore[attr-defined]
            with PdfReader(pdf_file_obj, password=self.password) as reader:
                metadata = purge_metadata(cast(dict[str, Any], reader.metadata))
                page1 = reader.pages[0].extract_text()
                metadata["page1"] = page1
                find = False
                for name, match, parser in self.routes:
                    for k, p in match.items():
                        if k not in metadata or not p.search(metadata[k]):
                            break
                    else:
                        find = True
                        break
                if find:
                    for doc in parser.lazy_parse(blob):
                        doc.metadata["router"] = name
                        yield doc


class LlamaIndexPDFParser(BaseBlobParser):
    """Parse `PDF` using `LlamaIndex`."""

    def __init__(
        self,
        *,
        password: Optional[str] = None,
        mode: Literal["single", "page"] = "single",
        pages_delimitor: str = _default_page_delimitor,
        extract_tables: Literal["markdown"] = "markdown",
        api_key: Optional[str] = None,
        verbose: bool = False,
        language: str = "en",
        extract_images: bool = False,
        images_to_text: CONVERT_IMAGE_TO_TEXT = None,
    ) -> None:
        if mode not in ["single", "page"]:
            raise ValueError("mode must be single or page")
        if extract_images:
            logger.info("Ignore extract_images==True in LlamaIndexPDFParser.")
        if extract_tables != "markdown" or images_to_text:
            logger.info("Ignore extract_tables!='markdown' in LlamaIndexPDFParser.")

        if password:
            logger.info("Ignore password in LlamaIndexPDFParser.")

        self.mode = mode
        self.extract_tables = extract_tables
        self.pages_delimitor = pages_delimitor
        self._llama_parser = LlamaParse(
            api_key=os.environ.get("LLAMAINDEX_API_KEY", api_key),
            result_type="markdown",  # "markdown" and "text" are available
            num_workers=1,
            verbose=verbose,
            language=language,
        )

    def _get_metadata(self, blob: Blob) -> dict[str, Any]:
        with blob.as_bytes_io() as pdf_file_obj:
            doc_metadata = purge_metadata(
                LlamaIndexPDFParser.__get_metadata(pdf_file_obj)
            )
            return blob.metadata | doc_metadata

    @staticmethod
    def __get_metadata(
        fp: BinaryIO,
        password: str = "",
        caching: bool = True,
    ) -> dict[str, Any]:
        from pdfminer.pdfpage import PDFDocument, PDFPage, PDFParser

        # Create a PDF parser object associated with the file object.
        parser = PDFParser(fp)
        # Create a PDF document object that stores the document structure.
        doc = PDFDocument(parser, password=password, caching=caching)
        metadata = {}

        for info in doc.info:
            metadata.update(info)
        for k, v in metadata.items():
            try:
                metadata[k] = PDFMinerParser.resolve_and_decode(v)
            except Exception as e:  # pragma: nocover
                # This metadata value could not be parsed. Instead of failing the PDF
                # read, treat it as a warning only if `strict_metadata=False`.
                logger.warning(
                    '[WARNING] Metadata key "%s" could not be parsed due to '
                    "exception: %s",
                    k,
                    str(e),
                )

        # Count number of pages.
        metadata["total_pages"] = len(list(PDFPage.create_pages(doc)))

        return metadata

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        try:
            import pdfminer  # noqa:F401
            from llama_parse import LlamaParse
        except ImportError:
            raise ImportError(
                "llama_parse package not found, please install it "
                "with `pip install llama_parse pdfminer.six`"
            )
        doc_metadata = self._get_metadata(blob) | {"source": blob.source}
        llama_documents = self._llama_parser.load_data(
            blob.as_bytes(), extra_info={"file_name": blob.source}
        )

        full_text = []
        for page_number, llama_doc in enumerate(llama_documents):
            if self.mode == "single":
                full_text.append(llama_doc.text)
            else:
                yield Document(
                    page_content=llama_doc.text,
                    metadata=doc_metadata | llama_doc.metadata | {"page": page_number},
                )
        if self.mode == "single":
            yield Document(
                page_content=self.pages_delimitor.join(full_text),
                metadata=doc_metadata,
            )


# PPR: https://djajafer.medium.com/document-parsing-with-omniparser-and-gpt4o-vision-5fa222c35ddd
# PPR: https://github.com/QuivrHQ/MegaParse/tree/main
