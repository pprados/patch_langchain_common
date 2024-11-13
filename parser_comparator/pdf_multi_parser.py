import logging
logger = logging.getLogger(__name__)
print(logger.name)
from langchain_community.document_loaders.base import BaseBlobParser
from langchain_core.documents import Document
from langchain_community.document_loaders.blob_loaders import Blob

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterator, List, Any
import numpy as np
import re

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

class PDFMultiParser(BaseBlobParser):

    def __init__(
            self,
            parsers_dict: dict[str : BaseBlobParser],
            debug_mode: bool = False,
    ) -> None:
        """"""
        self.parsers_dict = parsers_dict
        self.debug_mode = debug_mode

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        parsers_result = {}
        with ThreadPoolExecutor(max_workers=len(self.parsers_dict)) as executor:
            # Submit each parser's load method to the executor
            futures = {executor.submit(self.safe_parse, parser, blob): parser_name for parser_name, parser in self.parsers_dict.items()}
            # Collect the results from the futures as they complete
            for future in as_completed(futures):
                parser_name = futures[future]
                parser = self.parsers_dict[parser_name]
                print(parser_name)
                try:
                    documents_list = future.result()
                    #print(f"documents list for parser {parser_name} :", documents_list)
                    scores_dict = self.evaluate_parsing_quality(documents_list, parser)
                    global_score = np.mean(list(scores_dict.values()))
                    scores_dict['global_score']=global_score
                    parsers_result[parser_name] = (documents_list, scores_dict)
                    #print(parsers_result)

                except Exception as e:
                    print(f"Parser {parser_name} failed with exception : {e}")
                    logger.warning(f"Parser {parser_name} failed with exception : {e}")

        if not parsers_result:
            raise RuntimeError("All parsers have failed.")

        best_parser_data = max(parsers_result.items(), key=lambda item: item[1][1]['global_score'])
        if self.debug_mode:
            return list(parsers_result.items()), best_parser_data
        else:
            best_parser_associated_documents_list = best_parser_data[1][0]
            return iter(best_parser_associated_documents_list)


    @staticmethod
    def safe_parse(parser: BaseBlobParser, blob: Blob) -> List[Document]:
        try:
            return parser.parse(blob)
        except Exception as e:
            raise e


    def evaluate_parsing_quality(
            self,
            documents_list : list[Document],
            parser : BaseBlobParser,
    ) -> dict[str: float]:
        """Return the dictionnary {key=metric_name: value=score}"""
        title_level_scores_sum = 0
        list_level_scores_sum = 0
        tables_scores_sum = 0

        def evaluate_tables_identification(content : str):
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
                    print('found tables :', matches)
                    tables_scores_sum += len(matches)

        def evaluate_titles_identification(content):
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

        def evaluate_lists_identification(content):
            list_regex = re.compile(r"^([ \t]*)([-*+•◦▪·o]|\d+([./]|(\\.))) .+", re.MULTILINE)

            nonlocal list_level_scores_sum

            matches = re.findall(list_regex, content)
            for match in matches:
                indent = match[0]  # get indentation
                level = len(indent)  # a tab is considered equivalent to one space
                list_level_scores_sum += (level + 1)  # the more indent the parser identify the more it is rewarded

            return list_level_scores_sum

        # List of heuristics at the line level to evaluate
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
            # You can add more evaluation functions here
        }
        print('evaluate parsing output', scores_dict)
        return scores_dict


if __name__ == '__main__':
    pass

