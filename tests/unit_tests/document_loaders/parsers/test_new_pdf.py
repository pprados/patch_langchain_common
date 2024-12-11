import sys
import unittest
from unittest.mock import MagicMock, Mock

import pytest

if sys.version_info < (3, 11):
    from exceptiongroup import ExceptionGroup
from patch_langchain_community.document_loaders.parsers.new_pdf import PDFMultiParser


@pytest.mark.skip(reason="no way of currently testing this")
class TestPDFMultiParser(unittest.TestCase):
    def setUp(self) -> None:
        self.parser1 = Mock()
        self.parser2 = Mock()
        self.parser3 = Mock()

        self.multi_parser = PDFMultiParser(
            parsers={
                "parser1_name": self.parser1,
                "parser2_name": self.parser2,
                "parser3_name": self.parser3,
            }
        )

        self.blob = Mock()

    def test_partial_failures_return_log_containing_parsers_exceptions(self) -> None:
        exception_example_1 = Exception("Exception example 1")
        exception_example_2 = Exception("Exception example 2")
        self.parser1.parse.side_effect = exception_example_1
        doc_parser2 = MagicMock()
        doc_parser2.page_content = ""
        self.parser2.parse.return_value = [doc_parser2]
        self.parser3.parse.side_effect = exception_example_2

        with self.assertLogs(level="WARNING") as lo:
            self.multi_parser.lazy_parse(self.blob)

        logger_output = lo.output
        assert len(logger_output) == 2
        concatenated_log = " ".join(list(logger_output))
        assert str(exception_example_1) in concatenated_log
        assert str(exception_example_2) in concatenated_log

    def test_all_failures_raise_one_exception_containing_parsers_exceptions(
        self,
    ) -> None:
        exception_example_3 = Exception("Exception example 3")
        exception_example_4 = Exception("Exception example 4")
        exception_example_5 = Exception("Exception example 5")
        self.parser1.parse.side_effect = exception_example_3
        self.parser2.parse.side_effect = exception_example_4
        self.parser3.parse.side_effect = exception_example_5

        try:
            list(self.multi_parser.lazy_parse(self.blob))
        except ExceptionGroup as eg:
            exceptions = eg.exceptions
            assert exception_example_3 in exceptions
            assert exception_example_4 in exceptions
            assert exception_example_5 in exceptions
        else:
            self.fail("An exception was expected, but none was raised.")
