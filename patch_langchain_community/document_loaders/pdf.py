import copy
import json
import logging
import os
import re
import tempfile
import time
from abc import ABC
from io import StringIO
from pathlib import Path, PurePath
from typing import (
    TYPE_CHECKING,
    Any,
    BinaryIO,
    Iterator,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Union,
    cast,
)
from urllib.parse import urlparse

import requests
from langchain_community.document_loaders.base import BaseLoader
from langchain_community.document_loaders.blob_loaders import Blob
from langchain_community.document_loaders.dedoc import DedocBaseLoader
from langchain_community.document_loaders.unstructured import UnstructuredFileLoader
from langchain_core._api.deprecation import deprecated, warn_deprecated
from langchain_core.documents import Document
from langchain_core.utils import get_from_dict_or_env

from patch_langchain_community.document_loaders.parsers.pdf import (
    CONVERT_IMAGE_TO_TEXT,
    AmazonTextractPDFParser,
    DocumentIntelligenceParser,
    PDFMinerParser,
    PDFPlumberParser,
    PyMuPDFParser,
    PyPDFium2Parser,
    PyPDFParser,
    _default_page_delimitor,
)

if TYPE_CHECKING:
    from textractor.data.text_linearization_config import TextLinearizationConfig

logger = logging.getLogger(__file__)


@deprecated(
    since="0.3.X",  # TODO: update version 0.3.X
    removal="1.0",
    alternative_import="langchain_unstructured.UnstructuredPDFLoader",
)
class UnstructuredPDFLoader(UnstructuredFileLoader):
    """Load `PDF` files using `Unstructured`.

    You can run the loader in one of two modes: "single" and "elements".
    If you use "single" mode, the document will be returned as a single
    langchain Document object. If you use "elements" mode, the unstructured
    library will split the document into elements such as Title and NarrativeText.
    You can pass in additional unstructured kwargs after mode to apply
    different unstructured settings.

    Examples
    --------
    from langchain_community.document_loaders import UnstructuredPDFLoader

    loader = UnstructuredPDFLoader(
        "example.pdf", mode="elements", strategy="fast",
    )
    docs = loader.load()

    References
    ----------
    https://unstructured-io.github.io/unstructured/bricks.html#partition-pdf
    """

    def _get_elements(self) -> list:
        from unstructured.partition.pdf import partition_pdf

        return partition_pdf(filename=str(self.file_path), **self.unstructured_kwargs)


class BasePDFLoader(BaseLoader, ABC):
    """Base Loader class for `PDF` files.

    If the file is a web path, it will download it to a temporary file, use it, then
        clean up the temporary file after completion.
    """

    def __init__(
        self, file_path: Union[str, PurePath], *, headers: Optional[dict] = None
    ):
        """Initialize with a file path.

        Args:
            file_path: Either a local, S3 or web path to a PDF file.
            headers: Headers to use for GET request to download a file from a web path.
        """
        self.file_path = str(file_path)
        self.web_path = None
        self.headers = headers
        if "~" in self.file_path:
            self.file_path = os.path.expanduser(self.file_path)

        # If the file is a web path or S3, download it to a temporary file,
        # and use that. It's better to use a BlobLoader.
        if not os.path.isfile(self.file_path) and self._is_valid_url(self.file_path):
            self.temp_dir = tempfile.TemporaryDirectory()
            _, suffix = os.path.splitext(self.file_path)
            if self._is_s3_presigned_url(self.file_path):
                suffix = urlparse(self.file_path).path.split("/")[-1]
            temp_pdf = os.path.join(self.temp_dir.name, f"tmp{suffix}")
            self.web_path = self.file_path
            if not self._is_s3_url(self.file_path):
                r = requests.get(self.file_path, headers=self.headers)
                if r.status_code != 200:
                    raise ValueError(
                        "Check the url of your file; returned status code %s"
                        % r.status_code
                    )

                with open(temp_pdf, mode="wb") as f:
                    f.write(r.content)
                self.file_path = str(temp_pdf)
        elif not os.path.isfile(self.file_path):
            raise ValueError("File path %s is not a valid file or url" % self.file_path)

    def __del__(self) -> None:
        if hasattr(self, "temp_dir"):
            self.temp_dir.cleanup()

    @staticmethod
    def _is_valid_url(url: str) -> bool:
        """Check if the url is valid."""
        parsed = urlparse(url)
        return bool(parsed.netloc) and bool(parsed.scheme)

    @staticmethod
    def _is_s3_url(url: str) -> bool:
        """check if the url is S3"""
        try:
            result = urlparse(url)
            if result.scheme == "s3" and result.netloc:
                return True
            return False
        except ValueError:
            return False

    @staticmethod
    def _is_s3_presigned_url(url: str) -> bool:
        """Check if the url is a presigned S3 url."""
        try:
            result = urlparse(url)
            return bool(re.search(r"\.s3\.amazonaws\.com$", result.netloc))
        except ValueError:
            return False

    @property
    def source(self) -> str:
        return self.web_path if self.web_path is not None else self.file_path


@deprecated(
    since="0.3.X",  # TODO: update version 0.3.X
    removal="1.0",
    alternative_import="langchain_unstructured.UnstructuredPDFLoader",
)
class OnlinePDFLoader(BasePDFLoader):
    """Load online `PDF`."""

    def load(self) -> list[Document]:
        """Load documents."""
        loader = UnstructuredPDFLoader(str(self.file_path))
        return loader.load()


class PyPDFLoader(BasePDFLoader):
    """
    PyPDFLoader document loader integration

    Setup:
        Install ``langchain-community``.

        .. code-block:: bash

            pip install -U langchain-community

    Instantiate:
        .. code-block:: python

            from langchain_community.document_loaders import PyPDFLoader

            loader = PyPDFLoader(
                file_path = "./example_data/layout-parser-paper.pdf",
                password = "my-pasword",
                mode = "page",
                extract_images = True,
                # headers = None
                # extraction_mode = "plain",
                # extraction_kwargs = None,
            )

    Lazy load:
        .. code-block:: python

            docs = []
            docs_lazy = loader.lazy_load()

            # async variant:
            # docs_lazy = await loader.alazy_load()

            for doc in docs_lazy:
                docs.append(doc)
            print(docs[0].page_content[:100])
            print(docs[0].metadata)

        .. code-block:: python

            LayoutParser : A Uniﬁed Toolkit for Deep
            Learning Based Document Image Analysis
            Zejiang Shen1( ), R
            {'source': './example_data/layout-parser-paper.pdf', 'page': 0}

    Async load:
        .. code-block:: python

            docs = await loader.aload()
            print(docs[0].page_content[:100])8
            print(docs[0].metadata)

        .. code-block:: python

            LayoutParser : A Uniﬁed Toolkit for Deep
            Learning Based Document Image Analysis
            Zejiang Shen1( ), R
            {'source': './example_data/layout-parser-paper.pdf', 'page': 0}
    """  # noqa: E501

    def __init__(
        self,
        file_path: Union[str, PurePath],
        password: Optional[Union[str, bytes]] = None,
        headers: Optional[dict] = None,
        extract_images: bool = False,
        *,  # Move after the file_path ?
        images_to_text: CONVERT_IMAGE_TO_TEXT = None,
        mode: Literal["single", "page"] = "page",
        pages_delimitor: str = _default_page_delimitor,
        extraction_mode: Literal["plain", "layout"] = "plain",
        extraction_kwargs: Optional[dict] = None,
    ) -> None:
        """Initialize with a file path."""
        try:
            import pypdf  # noqa:F401
        except ImportError:
            raise ImportError(
                "pypdf package not found, please install it with `pip install pypdf`"
            )
        super().__init__(file_path, headers=headers)
        self.parser = PyPDFParser(
            password=password,
            extract_images=extract_images,
            images_to_text=images_to_text,
            mode=mode,
            pages_delimitor=pages_delimitor,
            extraction_mode=extraction_mode,
            extraction_kwargs=extraction_kwargs,
        )

    def lazy_load(
        self,
    ) -> Iterator[Document]:
        """Lazy load given path as pages."""
        if self.web_path:
            blob = Blob.from_data(open(self.file_path, "rb").read(), path=self.web_path)  # type: ignore[attr-defined]
        else:
            blob = Blob.from_path(self.file_path)  # type: ignore[attr-defined]
        yield from self.parser.lazy_parse(blob)


class PyPDFium2Loader(BasePDFLoader):
    """Load `PDF` using `pypdfium2` and chunks at character level."""

    def __init__(
        self,
        file_path: Union[str, PurePath],
        *,
        mode: Literal["single", "page"] = "page",
        pages_delimitor: str = _default_page_delimitor,
        password: Optional[str] = None,
        extract_images: bool = False,
        images_to_text: CONVERT_IMAGE_TO_TEXT = None,
        headers: Optional[dict] = None,
    ):
        """Initialize with a file path."""
        super().__init__(file_path, headers=headers)
        self.parser = PyPDFium2Parser(
            mode=mode,
            password=password,
            extract_images=extract_images,
            images_to_text=images_to_text,
            pages_delimitor=pages_delimitor,
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


@deprecated(
    since="0.3.X",  # TODO: update version 0.3.X
    removal="1.0",
    alternative="langchain_community.document_loaders.generic.GenericLoader",
)
class PyPDFDirectoryLoader(BaseLoader):
    """Load a directory with `PDF` files using `pypdf` and chunks at character level.

    Loader also stores page numbers in metadata.
    """

    def __init__(
        self,
        path: Union[str, PurePath],
        glob: str = "**/[!.]*.pdf",
        silent_errors: bool = False,
        load_hidden: bool = False,
        recursive: bool = False,
        extract_images: bool = False,
        *,
        password: Optional[str] = None,
        mode: Literal["single", "page"] = "page",
        images_to_text: CONVERT_IMAGE_TO_TEXT = None,
        headers: Optional[dict] = None,
        extraction_mode: Literal["plain", "layout"] = "plain",
        extraction_kwargs: Optional[dict] = None,
    ):
        self.password = password
        self.mode = mode
        self.path = path
        self.glob = glob
        self.load_hidden = load_hidden
        self.recursive = recursive
        self.silent_errors = silent_errors
        self.extract_images = extract_images
        self.images_to_text = images_to_text
        self.headers = headers
        self.extraction_mode = extraction_mode
        self.extraction_kwargs = extraction_kwargs

    @staticmethod
    def _is_visible(path: PurePath) -> bool:
        return not any(part.startswith(".") for part in path.parts)

    def load(self) -> list[Document]:
        p = Path(self.path)
        docs = []
        items = p.rglob(self.glob) if self.recursive else p.glob(self.glob)
        for i in items:
            if i.is_file():
                if self._is_visible(i.relative_to(p)) or self.load_hidden:
                    try:
                        loader = PyPDFLoader(
                            str(i),
                            password=self.password,
                            mode=self.mode,
                            extract_images=self.extract_images,
                            images_to_text=self.images_to_text,
                            headers=self.headers,
                            extraction_mode=self.extraction_mode,
                            extraction_kwargs=self.extraction_kwargs,
                        )
                        sub_docs = loader.load()
                        for doc in sub_docs:
                            doc.metadata["source"] = str(i)
                        docs.extend(sub_docs)
                    except Exception as e:
                        if self.silent_errors:
                            logger.warning(e)
                        else:
                            raise e
        return docs


class PDFMinerLoader(BasePDFLoader):
    """Load `PDF` files using `PDFMiner`."""

    def __init__(
        self,
        file_path: Union[str, PurePath],
        *,
        password: Optional[str] = None,
        mode: Literal["single", "page"] = "single",
        pages_delimitor: str = _default_page_delimitor,
        extract_images: bool = False,
        images_to_text: CONVERT_IMAGE_TO_TEXT = None,
        headers: Optional[dict] = None,
        concatenate_pages: Optional[bool] = None,
    ) -> None:
        """Initialize with file path.

        Args:
            extract_images: Whether to extract images from PDF.
            concatenate_pages: If True, concatenate all PDF pages into one a single
                               document. Otherwise, return one document per page.
        """
        try:
            from pdfminer.high_level import extract_text  # noqa:F401
        except ImportError:
            raise ImportError(
                "`pdfminer` package not found, please install it with "
                "`pip install pdfminer.six`"
            )

        super().__init__(file_path, headers=headers)
        self.parser = PDFMinerParser(
            password=password,
            extract_images=extract_images,
            images_to_text=images_to_text,
            concatenate_pages=concatenate_pages,
            mode=mode,
            pages_delimitor=pages_delimitor,
        )

    def lazy_load(
        self,
    ) -> Iterator[Document]:
        """Lazily load documents."""
        if self.web_path:
            blob = Blob.from_data(open(self.file_path, "rb").read(), path=self.web_path)  # type: ignore[attr-defined]
        else:
            blob = Blob.from_path(self.file_path)  # type: ignore[attr-defined]
        yield from self.parser.lazy_parse(blob)


class PDFMinerPDFasHTMLLoader(BasePDFLoader):
    """Load `PDF` files as HTML content using `PDFMiner`.
    Warning, the HTML output is just a positioning of the boxes,
    without being able to interpret the HTML in an LLM.
    """

    def __init__(
        self,
        file_path: Union[str, PurePath],
        *,
        password: Optional[str] = None,
        headers: Optional[dict] = None,
    ):
        """Initialize with a file path."""
        try:
            from pdfminer.high_level import extract_text_to_fp  # noqa:F401
        except ImportError:
            raise ImportError(
                "`pdfminer` package not found, please install it with "
                "`pip install pdfminer.six`"
            )

        super().__init__(file_path, headers=headers)
        self.password = password

    def lazy_load(self) -> Iterator[Document]:
        """Load file."""
        from pdfminer.high_level import extract_text_to_fp
        from pdfminer.layout import LAParams
        from pdfminer.utils import open_filename

        output_string = StringIO()
        with open_filename(self.file_path, "rb") as fp:
            extract_text_to_fp(
                cast(BinaryIO, fp),
                output_string,
                password=self.password or "",
                codec="",
                laparams=LAParams(),
                output_type="html",
            )
        metadata = {
            "source": self.file_path if self.web_path is None else self.web_path
        }
        yield Document(page_content=output_string.getvalue(), metadata=metadata)


class PyMuPDFLoader(BasePDFLoader):
    """Load `PDF` files using `PyMuPDF`."""

    def __init__(
        self,
        file_path: Union[str, PurePath],
        *,
        password: Optional[str] = None,
        mode: Literal["single", "page"] = "page",
        pages_delimitor: str = _default_page_delimitor,
        extract_images: bool = False,
        images_to_text: CONVERT_IMAGE_TO_TEXT = None,
        extract_tables: Union[
            Literal["csv"], Literal["markdown"], Literal["html"], None
        ] = None,
        headers: Optional[dict] = None,
        extract_tables_settings: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize with a file path."""
        try:
            import pymupdf  # noqa:F401
        except ImportError:
            raise ImportError(
                "`PyMuPDF` package not found, please install it with "
                "`pip install pymupdf`"
            )
        if mode not in ["single", "page"]:
            raise ValueError("mode must be single or page")
        super().__init__(file_path, headers=headers)
        self.parser = PyMuPDFParser(
            password=password,
            mode=mode,
            pages_delimitor=pages_delimitor,
            text_kwargs=kwargs,
            extract_images=extract_images,
            images_to_text=images_to_text,
            extract_tables=extract_tables,
            extract_tables_settings=extract_tables_settings,
        )

    def lazy_load(self, **kwargs: Any) -> Iterator[Document]:
        parser = self.parser
        if kwargs:
            warn_deprecated(
                since="0.2.17",
                removal="0.3",
                message=f"Received runtime arguments {kwargs}. Passing runtime args "
                f"to `load` is deprecated. Please pass arguments during "
                f"initialization instead.",
            )
            parser = copy.deepcopy(self.parser)
            parser.text_kwargs.update(kwargs)

        if self.web_path:
            blob = Blob.from_data(open(self.file_path, "rb").read(), path=self.web_path)  # type: ignore[attr-defined]
        else:
            blob = Blob.from_path(self.file_path)  # type: ignore[attr-defined]
        yield from parser.lazy_parse(blob)


class PDFPlumberLoader(BasePDFLoader):
    """Load `PDF` files using `pdfplumber`."""

    def __init__(
        self,
        file_path: Union[str, PurePath],
        *,
        text_kwargs: Optional[Mapping[str, Any]] = {
            "use_text_flow": False,
            "keep_blank_chars": False,
        },
        dedupe: bool = False,
        headers: Optional[dict] = None,
        password: Optional[str] = None,
        mode: Literal["single", "page"] = "page",
        extract_images: bool = False,
        images_to_text: CONVERT_IMAGE_TO_TEXT = None,
        pages_delimitor: str = _default_page_delimitor,
        extract_tables: Optional[
            Literal["csv", "markdown", "html"]
        ] = None,  # FIXME: auto ?
        extract_tables_settings: Optional[dict[str, Any]] = None,
    ) -> None:
        """Initialize with a file path."""
        try:
            import pdfplumber  # noqa:F401
        except ImportError:
            raise ImportError(
                "pdfplumber package not found, please install it with "
                "`pip install pdfplumber`"
            )

        super().__init__(file_path, headers=headers)
        self.parser = PDFPlumberParser(
            password=password,
            mode=mode,
            pages_delimitor=pages_delimitor,
            extract_images=extract_images,
            images_to_text=images_to_text,
            extract_tables=extract_tables,
            text_kwargs=text_kwargs,
            extract_tables_settings=extract_tables_settings,
            dedupe=dedupe,
        )

    def lazy_load(
        self,
    ) -> Iterator[Document]:
        """Lazy load given path as pages."""
        if self.web_path:
            blob = Blob.from_data(open(self.file_path, "rb").read(), path=self.web_path)  # type: ignore[attr-defined]
        else:
            blob = Blob.from_path(self.file_path)  # type: ignore[attr-defined]
        yield from self.parser.lazy_parse(blob)


# PPR: DedocPDFLoader
class DedocPDFLoader(DedocBaseLoader):
    """
    DedocPDFLoader document loader integration to load PDF files using `dedoc`.
    The file loader can automatically detect the correctness of a textual layer in the
        PDF document.
    Note that `__init__` method supports parameters that differ from ones of
        DedocBaseLoader.

    Setup:
        Install ``dedoc`` package.

        .. code-block:: bash

            pip install -U dedoc

    Instantiate:
        .. code-block:: python

            from langchain_community.document_loaders import DedocPDFLoader

            loader = DedocPDFLoader(
                file_path="example.pdf",
                # split=...,
                # with_tables=...,
                # pdf_with_text_layer=...,
                # pages=...,
                # ...
            )

    Load:
        .. code-block:: python

            docs = loader.load()
            print(docs[0].page_content[:100])
            print(docs[0].metadata)

        .. code-block:: python

            Some text
            {
                'file_name': 'example.pdf',
                'file_type': 'application/pdf',
                # ...
            }

    Lazy load:
        .. code-block:: python

            docs = []
            docs_lazy = loader.lazy_load()

            for doc in docs_lazy:
                docs.append(doc)
            print(docs[0].page_content[:100])
            print(docs[0].metadata)

        .. code-block:: python

            Some text
            {
                'file_name': 'example.pdf',
                'file_type': 'application/pdf',
                # ...
            }

    Parameters used for document parsing via `dedoc`
        (https://dedoc.readthedocs.io/en/latest/parameters/pdf_handling.html):

        with_attachments: enable attached files extraction
        recursion_deep_attachments: recursion level for attached files extraction,
            works only when with_attachments==True
        pdf_with_text_layer: type of handler for parsing, available options
            ["true", "false", "tabby", "auto", "auto_tabby" (default)]
        language: language of the document for PDF without a textual layer,
            available options ["eng", "rus", "rus+eng" (default)], the list of
            languages can be extended, please see
            https://dedoc.readthedocs.io/en/latest/tutorials/add_new_language.html
        pages: page slice to define the reading range for parsing
        is_one_column_document: detect number of columns for PDF without a textual
            layer, available options ["true", "false", "auto" (default)]
        document_orientation: fix document orientation (90, 180, 270 degrees) for PDF
            without a textual layer, available options ["auto" (default), "no_change"]
        need_header_footer_analysis: remove headers and footers from the output result
        need_binarization: clean pages background (binarize) for PDF without a textual
            layer
        need_pdf_table_analysis: parse tables for PDF without a textual layer
    """

    def _make_config(self) -> dict:
        from dedoc.utils.langchain import make_manager_pdf_config

        return make_manager_pdf_config(
            file_path=self.file_path,
            parsing_params=self.parsing_parameters,
            split=self.split,
        )


# %% --------- Online pdf loader ---------
# MathpixPDFLoader implementation taken largely from Daniel Gross's:
# https://gist.github.com/danielgross/3ab4104e14faccc12b49200843adab21
class MathpixPDFLoader(BasePDFLoader):
    """Load `PDF` files using `Mathpix` service."""

    def __init__(
        self,
        file_path: Union[str, PurePath],
        processed_file_format: str = "md",
        max_wait_time_seconds: int = 500,
        should_clean_pdf: bool = False,
        extra_request_data: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize with a file path.

        Args:
            file_path: a file for loading.
            processed_file_format: a format of the processed file. Default is "md".
            max_wait_time_seconds: a maximum time to wait for the response from
             the server. Default is 500.
            should_clean_pdf: a flag to clean the PDF file. Default is False.
            extra_request_data: Additional request data.
            **kwargs: additional keyword arguments.
        """
        self.mathpix_api_key = get_from_dict_or_env(
            kwargs, "mathpix_api_key", "MATHPIX_API_KEY"
        )
        self.mathpix_api_id = get_from_dict_or_env(
            kwargs, "mathpix_api_id", "MATHPIX_API_ID"
        )

        # The base class isn't expecting these and doesn't collect **kwargs
        kwargs.pop("mathpix_api_key", None)
        kwargs.pop("mathpix_api_id", None)

        super().__init__(file_path, **kwargs)
        self.processed_file_format = processed_file_format
        self.extra_request_data = (
            extra_request_data if extra_request_data is not None else {}
        )
        self.max_wait_time_seconds = max_wait_time_seconds
        self.should_clean_pdf = should_clean_pdf

    @property
    def _mathpix_headers(self) -> dict[str, str]:
        return {"app_id": self.mathpix_api_id, "app_key": self.mathpix_api_key}

    @property
    def url(self) -> str:
        return "https://api.mathpix.com/v3/pdf"

    @property
    def data(self) -> dict:
        options = {
            "conversion_formats": {self.processed_file_format: True},
            **self.extra_request_data,
        }
        return {"options_json": json.dumps(options)}

    def send_pdf(self) -> str:
        with open(str(self.file_path), "rb") as f:
            files = {"file": f}
            response = requests.post(
                self.url, headers=self._mathpix_headers, files=files, data=self.data
            )
        response_data = response.json()
        if "error" in response_data:
            raise ValueError(f"Mathpix request failed: {response_data['error']}")
        if "pdf_id" in response_data:
            pdf_id = response_data["pdf_id"]
            return pdf_id
        else:
            raise ValueError("Unable to send PDF to Mathpix.")

    def wait_for_processing(self, pdf_id: str) -> None:
        """Wait for processing to complete.

        Args:
            pdf_id: a PDF id.

        Returns: None
        """
        url = self.url + "/" + pdf_id
        for _ in range(0, self.max_wait_time_seconds, 5):
            response = requests.get(url, headers=self._mathpix_headers)
            response_data = response.json()

            # This indicates an error with the request (e.g. auth problems)
            error = response_data.get("error", None)
            error_info = response_data.get("error_info", None)

            if error is not None:
                error_msg = f"Unable to retrieve PDF from Mathpix: {error}"

                if error_info is not None:
                    error_msg += f" ({error_info['id']})"

                raise ValueError(error_msg)

            status = response_data.get("status", None)

            if status == "completed":
                return
            elif status == "error":
                # This indicates an error with the PDF processing
                raise ValueError("Unable to retrieve PDF from Mathpix")
            else:
                print(  # noqa: T201
                    f"Status: {status}, waiting for processing to complete"
                )
                time.sleep(5)
        raise TimeoutError

    def get_processed_pdf(self, pdf_id: str) -> str:
        self.wait_for_processing(pdf_id)
        url = f"{self.url}/{pdf_id}.{self.processed_file_format}"
        response = requests.get(url, headers=self._mathpix_headers)
        return response.content.decode("utf-8")

    def clean_pdf(self, contents: str) -> str:
        """Clean the PDF file.

        Args:
            contents: a PDF file contents.

        Returns:

        """
        contents = "\n".join(
            [line for line in contents.split("\n") if not line.startswith("![]")]
        )
        # replace \section{Title} with # Title
        contents = contents.replace("\\section{", "# ").replace("}", "")
        # replace the "\" slash that Mathpix adds to escape $, %, (, etc.
        contents = (
            contents.replace(r"\$", "$")
            .replace(r"\%", "%")
            .replace(r"\(", "(")
            .replace(r"\)", ")")
        )
        return contents

    def load(self) -> list[Document]:
        pdf_id = self.send_pdf()
        contents = self.get_processed_pdf(pdf_id)
        if self.should_clean_pdf:
            contents = self.clean_pdf(contents)
        metadata = {"source": self.source, "file_path": self.source, "pdf_id": pdf_id}
        return [Document(page_content=contents, metadata=metadata)]


class AmazonTextractPDFLoader(BasePDFLoader):
    """Load `PDF` files from a local file system, HTTP or S3.

    To authenticate, the AWS client uses the following methods to
    automatically load credentials:
    https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html

    If a specific credential profile should be used, you must pass
    the name of the profile from the ~/.aws/credentials file that is to be used.

    Make sure the credentials / roles used have the required policies to
    access the Amazon Textract service.

    Example:
        .. code-block:: python
            from langchain_community.document_loaders import AmazonTextractPDFLoader
            loader = AmazonTextractPDFLoader(
                file_path="s3://pdfs/myfile.pdf"
            )
            document = loader.load()
    """

    def __init__(
        self,
        file_path: Union[str, PurePath],
        textract_features: Optional[Sequence[str]] = None,
        client: Optional[Any] = None,
        credentials_profile_name: Optional[str] = None,
        region_name: Optional[str] = None,
        endpoint_url: Optional[str] = None,
        headers: Optional[dict] = None,
        *,
        linearization_config: Optional["TextLinearizationConfig"] = None,
    ) -> None:
        """Initialize the loader.

        Args:
            file_path: A file, url or s3 path for input file
            textract_features: Features to be used for extraction, each feature
                               should be passed as a str that conforms to the enum
                               `Textract_Features`, see `amazon-textract-caller` pkg
            client: boto3 textract client (Optional)
            credentials_profile_name: AWS profile name, if not default (Optional)
            region_name: AWS region, eg us-east-1 (Optional)
            endpoint_url: endpoint url for the textract service (Optional)
            linearization_config: Config to be used for linearization of the output
                                  should be an instance of TextLinearizationConfig from
                                  the `textractor` pkg
        """
        super().__init__(file_path, headers=headers)

        try:
            import textractcaller as tc
        except ImportError:
            raise ImportError(
                "Could not import amazon-textract-caller python package. "
                "Please install it with `pip install amazon-textract-caller`."
            )
        if textract_features:
            features = [tc.Textract_Features[x] for x in textract_features]
        else:
            features = []

        if credentials_profile_name or region_name or endpoint_url:
            try:
                import boto3

                if credentials_profile_name is not None:
                    session = boto3.Session(profile_name=credentials_profile_name)
                else:
                    # use default credentials
                    session = boto3.Session()

                client_params = {}
                if region_name:
                    client_params["region_name"] = region_name
                if endpoint_url:
                    client_params["endpoint_url"] = endpoint_url

                client = session.client("textract", **client_params)

            except ImportError:
                raise ImportError(
                    "Could not import boto3 python package. "
                    "Please install it with `pip install boto3`."
                )
            except Exception as e:
                raise ValueError(
                    "Could not load credentials to authenticate with AWS client. "
                    "Please check that credentials in the specified "
                    f"profile name are valid. {e}"
                ) from e
        self.parser = AmazonTextractPDFParser(
            textract_features=features,
            client=client,
            linearization_config=linearization_config,
        )

    def load(self) -> list[Document]:
        """Load given path as pages."""
        return list(self.lazy_load())

    def lazy_load(
        self,
    ) -> Iterator[Document]:
        """Lazy load documents"""
        # the self.file_path is local, but the blob has to include
        # the S3 location if the file originated from S3 for multi-page documents
        # raises ValueError when multi-page and not on S3"""

        if self.web_path and self._is_s3_url(self.web_path):
            blob = Blob(path=self.web_path)  # type: ignore[call-arg] # type: ignore[misc]
        else:
            blob = Blob.from_path(self.file_path)  # type: ignore[attr-defined]
            if AmazonTextractPDFLoader._get_number_of_pages(blob) > 1:
                raise ValueError(
                    f"the file {blob.path} is a multi-page document, \
                    but not stored on S3. \
                    Textract requires multi-page documents to be on S3."
                )

        yield from self.parser.parse(blob)

    @staticmethod
    def _get_number_of_pages(blob: Blob) -> int:  # type: ignore[valid-type]
        try:
            import pypdf
            from PIL import Image, ImageSequence

        except ImportError:
            raise ImportError(
                "Could not import pypdf or Pilloe python package. "
                "Please install it with `pip install pypdf Pillow`."
            )
        if blob.mimetype == "application/pdf":  # type: ignore[attr-defined]
            with blob.as_bytes_io() as input_pdf_file:  # type: ignore[attr-defined]
                pdf_reader = pypdf.PdfReader(input_pdf_file)
                return len(pdf_reader.pages)
        elif blob.mimetype == "image/tiff":  # type: ignore[attr-defined]
            num_pages = 0
            img = Image.open(blob.as_bytes())  # type: ignore[attr-defined]
            for _, _ in enumerate(ImageSequence.Iterator(img)):
                num_pages += 1
            return num_pages
        elif blob.mimetype in ["image/png", "image/jpeg"]:  # type: ignore[attr-defined]
            return 1
        else:
            raise ValueError(f"unsupported mime type: {blob.mimetype}")  # type: ignore[attr-defined]


class DocumentIntelligenceLoader(BasePDFLoader):
    """Load a PDF with Azure Document Intelligence"""

    def __init__(
        self,
        file_path: str,
        client: Any,
        model: str = "prebuilt-document",
        headers: Optional[dict] = None,
    ) -> None:
        """
        Initialize the object for file processing with Azure Document Intelligence
        (formerly Form Recognizer).

        This constructor initializes a DocumentIntelligenceParser object to be used
        for parsing files using the Azure Document Intelligence API. The load method
        generates a Document node including metadata (source blob and page number)
        for each page.

        Parameters:
        -----------
        file_path : str
            The path to the file that needs to be parsed.
        client: Any
            A DocumentAnalysisClient to perform the analysis of the blob
        model : str
            The model name or ID to be used for form recognition in Azure.

        Examples:
        ---------
        >>> obj = DocumentIntelligenceLoader(
        ...     file_path="path/to/file",
        ...     client=client,
        ...     model="prebuilt-document"
        ... )
        """

        self.parser = DocumentIntelligenceParser(client=client, model=model)
        super().__init__(file_path, headers=headers)

    def load(self) -> list[Document]:
        """Load given path as pages."""
        return list(self.lazy_load())

    def lazy_load(
        self,
    ) -> Iterator[Document]:
        """Lazy load given path as pages."""
        blob = Blob.from_path(self.file_path)  # type: ignore[attr-defined]
        yield from self.parser.parse(blob)


# Legacy: only for backwards compatibility. Use PyPDFLoader instead
@deprecated(
    since="0.0.30",
    removal="1.0",
    alternative="PyPDFLoader",
)
class PagedPDFSplitter(PyPDFLoader):
    pass
