import logging
from tornado.gen import multi

from langchain_community.document_loaders.parsers import PyPDFParser, PyPDFium2Parser, PDFMinerParser
from langchain_core.document_loaders import BaseBlobParser
from pdf_multi_parser import PDFMultiParser

logger = logging.getLogger(__name__)
print(logger.name)
from langchain_community.document_loaders.pdf import BasePDFLoader, PyPDFLoader, PyPDFium2Loader, PDFMinerLoader
from langchain_core.documents import Document
from langchain_community.document_loaders.blob_loaders import Blob
from pdf_multi_parser import PDFMultiParser
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterator, List, Optional, Dict


# Parser = PyPDFLoader(file_path="/home/mame/PycharmProjects/langchain-perso/test_pdf.pdf")
#
# docs = []
# docs_lazy = Parser.lazy_load()
# print(docs_lazy)
# print(docs_lazy)
# async variant:
# docs_lazy = await Parser.alazy_load()

# for doc in docs_lazy:
#     docs.append(doc)
# print([d.page_content[:100] for d in docs])
#print(docs[0].metadata)

class PDFMultiLoader(BasePDFLoader):

    def __init__(
            self,
            file_path: str,
            pdf_multi_parser: PDFMultiParser,
            headers: Optional[Dict] = None,
    ) -> None:
        """"""
        super().__init__(file_path, headers=headers)
        self.parser = pdf_multi_parser

    def get_best_parsing(
            self,
            iterator_parsings: Iterator[tuple[list[Document], str]]
    ) -> list[Document]:
        """"""
        #for concatenated_documents in concatenated_documents_list:
        # TODO implémenter une méthode de comparaison de qualité de parsing
        # comment comparer en loadant en parallèle ?
        return list(iterator_parsings)[0][0]

    def lazy_load(
        self,
    ) -> Iterator[tuple[list[Document], str]]:
        """Lazy load given path as pages."""
        if self.web_path:
            blob = Blob.from_data(open(self.file_path, "rb").read(), path=self.web_path)  # type: ignore[attr-defined]
        else:
            blob = Blob.from_path(self.file_path)  # type: ignore[attr-defined]
        yield from self.parser.parse(blob)


if __name__ == '__main__':
    file_path = "../test_pdf.pdf"
    parser1 = PyPDFParser()
    parser2 = PyPDFium2Parser()
    parser3 = PDFMinerParser()
    multi_loader = PDFMultiLoader(
        file_path=file_path,
        parsers_list=[parser1, parser2, parser3]
    )
    parsings_results = multi_loader.lazy_load()
    best_parsing = multi_loader.get_best_parsing(parsings_results)
    print(best_parsing)

