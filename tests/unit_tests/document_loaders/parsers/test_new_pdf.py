import unittest
from unittest.mock import Mock

from patch_langchain_community.document_loaders.parsers.new_pdf import PDFMultiParser


class TestPDFMultiParser(unittest.TestCase):

    def setUp(self):
        self.parser1 = Mock()
        self.parser2 = Mock()
        self.parser3 = Mock()

        self.multi_parser = PDFMultiParser(parsers={
            "parser1_name": self.parser1,
            "parser2_name": self.parser2,
            "parser3_name": self.parser3,
        })

        self.blob = Mock()

    def test_partial_failures_return_log_containing_parsers_exceptions(self):
        exception_example_1 = "Exception example 1"
        exception_example_2 = "Exception example 2"
        self.parser1.parse.side_effect = Exception(exception_example_1)
        doc_parser2 = Mock()
        doc_parser2.page_content = ""
        self.parser2.parse.return_value = [doc_parser2]
        self.parser3.parse.side_effect = Exception(exception_example_2)

        with self.assertLogs(level='WARNING') as lo:
            self.multi_parser.lazy_parse(self.blob)

        logger_output = lo.output
        self.assertEqual(len(logger_output), 1)
        self.assertIn(exception_example_1, logger_output[0])
        self.assertIn(exception_example_2, logger_output[0])

    def test_all_failures_raise_one_exception_containing_parsers_exceptions(self):
        exception_example_3 = "Exception example 3"
        exception_example_4 = "Exception example 4"
        exception_example_5 = "Exception example 5"
        self.parser1.parse.side_effect = Exception(exception_example_3)
        self.parser2.parse.side_effect = Exception(exception_example_4)
        self.parser3.parse.side_effect = Exception(exception_example_5)

        try:
            list(self.multi_parser.lazy_parse(self.blob))
        except Exception as e:
            exception = str(e)
            self.assertIn(exception_example_3, exception)
            self.assertIn(exception_example_4, exception)
            self.assertIn(exception_example_5, exception)
        else:
            self.fail("An exception was expected, but none was raised.")