"""Module contains common parsers for PDFs."""
DEV_DEBUG=False
import re
from typing import (
    Any,
    Iterator,
    Literal,
    Optional,
    Union,
    cast,
)

from langchain_community.document_loaders.base import BaseBlobParser
from langchain_community.document_loaders.blob_loaders import Blob
from langchain_core.documents import Document

from .pdf import ImagesPdfParser, _default_page_delimitor, purge_metadata

from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

import logging
logger = logging.getLogger(__name__)

class PDFMultiParser(BaseBlobParser):

    def __init__(
            self,
            parsers_dict: dict[str : BaseBlobParser],
            debug_mode: bool = False,
    ) -> None:
        """"""
        self.parsers_dict = parsers_dict
        self.debug_mode = debug_mode

    def lazy_parse(
            self,
            blob: Blob,
    ) -> Iterator[Document]:
        parsers_result = {}
        with ThreadPoolExecutor(max_workers=len(self.parsers_dict)) as executor:
            # Submit each parser's load method to the executor
            futures = {executor.submit(self.safe_parse, parser, blob): parser_name for parser_name, parser in self.parsers_dict.items()}
            # Collect the results from the futures as they complete
            for future in as_completed(futures):
                parser_name = futures[future]
                try:
                    documents_list = future.result()
                    #print(f"documents list for parser {parser_name} :", documents_list)
                    scores_dict = self.evaluate_parsing_quality(documents_list)
                    global_score = np.mean(list(scores_dict.values()))
                    scores_dict['global_score']=global_score
                    parsers_result[parser_name] = (documents_list, scores_dict)
                    print(f"{parser_name} \u001B[32m completed \u001B[0m")
                    #print(parsers_result)

                except Exception as e:
                    logger.warning(f"Parser {parser_name} failed with exception : {e}")
                    if DEV_DEBUG:
                        raise e
        if not parsers_result:
            raise RuntimeError("All parsers have failed.")

        best_parser_data = max(parsers_result.items(), key=lambda item: item[1][1]['global_score'])
        best_parser_name = best_parser_data[0]
        if self.debug_mode:
            return list(parsers_result.items()), best_parser_name
        else:
            best_parser_associated_documents_list = best_parser_data[1][0]
            return iter(best_parser_associated_documents_list)


    @staticmethod
    def safe_parse(
            parser: BaseBlobParser,
            blob: Blob,
    ) -> list[Document]:
        """Parse function handling errors for logging purposes in the multi thread process"""
        try:
            return parser.parse(blob)
        except Exception as e:
            raise e


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

        scores_dict = {
            "titles": title_level_scores_sum,
            "lists": list_level_scores_sum,
            "tables": tables_scores_sum,
            # You can add more resulting scores here
        }
        return scores_dict

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
        # self.password = password
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
