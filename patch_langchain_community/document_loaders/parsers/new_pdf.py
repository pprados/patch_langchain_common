"""Module contains common parsers for PDFs."""
import logging
import os
import re
import sys
from typing import (
    Any,
    BinaryIO,
    Iterator,
    Literal,
    Optional,
    Union,
    cast,
)

if sys.version_info < (3, 11):  # FIXME: (3,11)
    from exceptiongroup import ExceptionGroup

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

from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

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
        """Lazily parse the blob. (Fakely because for each parser all Documents are loaded at once to compute the global
        parsing score.)
        Memory optimisation at least: In multi-thread process we keep in memory only the current parser Documents list
        and the current best parser Documents list."""
        current_best_score = 0
        current_best_parsing_documents = []
        current_best_parser_name = None
        all_exceptions: dict[str, Exception] = {}

        with ThreadPoolExecutor(
                max_workers=self.max_workers or len(self.parsers)
        ) as executor:
            # Submit each parser's load method to the executor
            futures = {
                executor.submit(self._safe_parse, parser, blob): parser_name
                for parser_name, parser in self.parsers.items()
            }
            # Collect the results from the futures as they complete
            for future in as_completed(futures):
                parser_name = futures[future]
                try:
                    documents = future.result()
                    #print(f"{parser_name} \u001B[32m completed \u001B[0m")
                    #print(f"documents list for parser {parser_name} :", documents)
                    metric_name2score = self.evaluate_parsing_quality(documents)
                    global_score = metric_name2score['global_score']
                    if global_score >= current_best_score:
                        current_best_score = global_score
                        current_best_parsing_documents = documents
                        current_best_parser_name = parser_name

                except Exception as e:
                    log = f"Parser {parser_name} failed with exception : {e}"
                    logger.warning(log)
                    all_exceptions[parser_name]=e

        # si tu ne veux pas que ça continue en cas d'erreur et qu'il y a des erreurs alors exception
        if not self.continue_if_error and all_exceptions:
            raise ExceptionGroup("Some parsers have failed.", list(all_exceptions.values()))
        # si tu ne veux pas que ça continue en cas d'erreur et qu'il n'y a pas d'erreur PASS
        # si tu veux que ça continue en cas d'erreur et qu'il y a des erreurs (PASS elles sont affichées plus haut grâce au logger)
        # si tu veux que ça continue en cas d'erreur et qu'il n'y a pas d'erreur PASS

        # si tous les parsers sont en erreur, soulever une exception
        if len(all_exceptions) == len(self.parsers):
            raise ExceptionGroup("All parsers have failed.", list(all_exceptions.values()))

        current_best_parsing_documents[0].metadata['parser_name'] = current_best_parser_name

        return iter(current_best_parsing_documents)


    @staticmethod  # FIXME
    def _safe_parse(
            parser: BaseBlobParser,
            blob: Blob,
    ) -> list[Document]:
        """Parse function handling errors for logging purposes in the multi thread process"""
        try:
            return parser.parse(blob)
        except Exception as e:
            raise e

    def parse_and_return_all_results(
            self,
            blob: Blob,
    ) :
        parser_name2parser_results:dict[str, tuple[list[Document], dict[str, float]]]
        all_exceptions:dict[str,Exception]= {}
        with ThreadPoolExecutor(max_workers=len(self.parsers)) as executor:
            # Submit each parser's load method to the executor
            futures = {executor.submit(self.safe_parse, parser, blob):
                           parser_name for parser_name, parser in self.parsers.items()
                       }
            # Collect the results from the futures as they complete
            for future in as_completed(futures):
                parser_name = futures[future]
                try:
                    documents = future.result()
                    #print(f"{parser_name} \u001B[32m completed \u001B[0m")
                    #print(f"documents list for parser {parser_name} :", documents)
                    metric_name2score = self.evaluate_parsing_quality(documents)
                    parser_name2parser_results[parser_name] = (documents, metric_name2score)
                except Exception as e:
                    log = f"Parser {parser_name} failed with exception : {e}"
                    logger.warning(log)
                    all_exceptions[parser_name]=log

        # si tu ne veux pas que ça continue en cas d'erreur et qu'il y a des erreurs alors exception
        if not self.continue_if_error and all_exceptions:
            raise ExceptionGroup("Some parsers have failed.", all_exceptions)
        # si tu ne veux pas que ça continue en cas d'erreur et qu'il n'y a pas d'erreur PASS
        # si tu veux que ça continue en cas d'erreur et qu'il y a des erreurs (PASS elles sont affichées plus haut grâce au logger)
        # si tu veux que ça continue en cas d'erreur et qu'il n'y a pas d'erreur PASS

        # si tous les parsers sont en erreur, soulever une exception
        if len(all_exceptions) == len(self.parsers):
            raise ExceptionGroup("All parsers have failed.", list(all_exceptions.values()))

        return parser_name2parser_results


    def evaluate_parsing_quality(
            self,
            documents_list : list[Document],
    ) -> dict[str: float]:
        """Evaluate the quality of a parsing based on some metrics measured by heuristics.
        Return the dictionnary {key=metric_name: value=score}"""


        def evaluate_tables_identification(
                content : str,
        ) -> float:
            """Evaluate the quality of tables identification in a document."""

            nonlocal tables_scores_sum

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
                r"|[^\n]*))\n){2,})"
            ]

            for pattern in patterns:
                matches = re.findall(pattern, content)
                if matches:
                    tables_scores_sum += len(matches)

        def evaluate_titles_identification(
                content : str,
        ) -> float:
            """Evaluate the quality of titles identification in a document."""

            titles_tags_weights = {
                r'# ': np.exp(0),
                r'## ': np.exp(1),
                r'### ': np.exp(2),
                r'#### ': np.exp(3),
                r'##### ': np.exp(4),
                r'###### ': np.exp(5)
            }

            nonlocal title_level_scores_sum

            for title, weight in titles_tags_weights.items():
                pattern = re.compile(rf"^{re.escape(title)}", re.MULTILINE)
                matches = re.findall(pattern, content)
                title_level_scores_sum += len(matches) * weight

            return title_level_scores_sum

        def evaluate_lists_identification(
                content : str,
        ) -> float:
            """Evaluate the quality of lists identification in a document."""

            list_regex = re.compile(r"^([ \t]*)([-*+•◦▪·o]|\d+([./]|(\\.))) .+", re.MULTILINE)

            nonlocal list_level_scores_sum

            matches = re.findall(list_regex, content)
            for match in matches:
                indent = match[0]  # get indentation
                level = len(indent)  # a tab is considered equivalent to one space
                list_level_scores_sum += (level + 1)  # the more indent the parser identify the more it is rewarded



            return list_level_scores_sum

        def compute_global_parsing_score(
                metric_name2score : dict[str, float],
        ) -> float:
            """Compute the global parsing score based on the scores of each metric."""
            return np.mean(list(metric_name2score.values()))


        # Metrics
        title_level_scores_sum = 0
        list_level_scores_sum = 0
        tables_scores_sum = 0

        # Heuristics function used for each metric
        evaluation_functions_dict = {
            "titles": evaluate_titles_identification,
            "lists": evaluate_lists_identification,
            'tables': evaluate_tables_identification,
            # You can add more evaluation functions here
        }

        for doc in documents_list:
            content = doc.page_content
            for func_name, func in evaluation_functions_dict.items():
                func(content)

        metric_name2score = {
            "titles": title_level_scores_sum,
            "lists": list_level_scores_sum,
            "tables": tables_scores_sum,
            # You can add more resulting scores here
        }
        global_score = compute_global_parsing_score(metric_name2score)
        metric_name2score['global_score'] = global_score
        return metric_name2score

class PyMuPDF4LLMParser(ImagesPdfParser):
    """Parse `PDF` using `PyMuPDF`."""

    def __init__(
        self,
        *,
        password: Optional[str] = None,
        mode: Literal["single", "paged"] = "single",
        pages_delimitor: str = _default_page_delimitor,
        to_markdown_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        """Initialize the parser.

        Args:
            password: Password to open the PDF.
            mode: Extraction mode to use. Either "single" or "paged".
            pages_delimitor: Delimiter to use between pages.
                             May be r'\f', '<!--PAGE BREAK -->', ...

            to_markdown_kwargs: Keyword arguments to pass to the PyMuPDF4LLM
             extraction method.
        """
        super().__init__(
            extract_images=False,  # PPR: extract_images will be True
            images_to_text=None,
        )
        if mode not in ["single", "paged"]:
            raise ValueError("mode must be single or paged")
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
            import pymupdf4llm  # noqa:F401
        except ImportError:
            raise ImportError(
                "pymupdf4llm package not found, please install it "
                "with `pip install pymupdf4llm`"
            )
        with (PyMuPDFParser._lock):
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
                    elif self.mode == "paged":
                        yield Document(
                            page_content=mu_doc["text"],
                            metadata=purge_metadata(mu_doc["metadata"]),
                        )
                        # PPR TODO: extraire les images. Voir PyMuPDFParser
                        # PPR TODO: extraire les tableaux ? Voir PyMuPDFParser
                if self.mode == "single":
                    yield Document(
                        page_content=self.pages_delimitor.join(full_text),
                        metadata=purge_metadata(metadata),
                    )

    _map_key = {"page_count": "total_pages", "file_path": "source"}
    _date_key = ["creationdate", "moddate"]


# PPR PDFRouterParser à revoir
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
        ("Microsoft", "Microsoft", None, PyMuPDFParser()),
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
                Optional[Union[re.Pattern, str]],
                Optional[Union[re.Pattern, str]],
                Optional[Union[re.Pattern, str]],
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
        for producer, creator, page, parser in routes:
            if isinstance(producer, str):
                producer = re.compile(producer)
            if isinstance(creator, str):
                creator = re.compile(creator)
            if isinstance(page, str):
                page = re.compile(page)
            new_routes.append((producer, creator, page, parser))
        self.routes = new_routes

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:  # type: ignore[valid-type]
        """Lazily parse the blob."""
        from pypdf import PdfReader

        with blob.as_bytes_io() as pdf_file_obj:  # type: ignore[attr-defined]
            with PdfReader(pdf_file_obj, password=self.password) as reader:
                if reader.metadata:
                    producer, creator = (
                        reader.metadata.producer,
                        reader.metadata.creator,
                    )
                    page1 = reader.pages[0].extract_text()
                for re_producer, re_create, re_page, parser in self.routes:
                    is_producer = not re_producer or re_producer.search(producer)
                    is_creator = not re_create or re_create.search(creator)
                    is_page = not re_page or re_page.search(page1)
                    if is_producer and is_creator and is_page:
                        yield from parser.lazy_parse(blob)


class LlamaIndexPDFParser(BaseBlobParser):
    """Parse `PDF` using `LlamaIndex`."""

    def __init__(
        self,
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
    ) -> None:
        try:
            import pdfminer  # noqa:F401
            from llama_parse import LlamaParse
        except ImportError:
            raise ImportError(
                "llama_parse package not found, please install it "
                "with `pip install llama_parse pdfminer.six`"
            )
        if mode not in ["single", "paged"]:
            raise ValueError("mode must be single or paged")
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
            api_key=os.environ.get("LLAMA_CLOUD_API_KEY", api_key),
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
                    f'[WARNING] Metadata key "{k}" could not be parsed due to '
                    f"exception: {str(e)}"
                )

        # Count number of pages.
        metadata["total_pages"] = len(list(PDFPage.create_pages(doc)))

        return metadata

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        # import pickle  # FIXME: pickle
        # with open("/home/pprados/workspace.bda/patch_pdf_loader/llama-parse.pickle",
        #           "rb") as f:
        #     llama_documents = pickle.load(f)
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
